from torch.nn.parallel import DistributedDataParallel as DDP
import importlib
import argparse
import gc
import math
import os
import random
import time
import json
import toml
from itertools import cycle
from multiprocessing import Value

from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from torch.utils.tensorboard import SummaryWriter

import library.train_util as train_util
from library.train_util import (
    DreamBoothDataset,
)
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import apply_snr_weight

"""
temp config place
"""
train_data_dir_2 = "/home/chenkaizheng/data/diffusion_data/dreambooth_3/lora_data/with_tag/nohead_jiaocha_newnew" 
reg_data_dir_2 = ""
logging_dir_2 = "log_locon2"
log_prefix_2 = ""
kld_weight_value = 0 # kl散度的系数
orth_te_weight = 1 # 正交loss的系数
orth_unet_weight = 1
model_path_2 = "/home/chenkaizheng/codes/vscode/waifu/webui_new/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.ckpt"
network_dim = 128
network_alpha = 128
# 暂时不支持locon / currently not supported locon yet
network_module = "networks.lora"
# 是否固定网络1
freeze_network_1= False
# 半固定指的是前半部分只训练衣服lora，后半部分两个lora一起微调
semi_freeze_network_1 = True
# 基准的人物lora的路径
pretrained_lora_path = "/home/chenkaizheng/codes/vscode/waifu/webui_new/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/yetAnotherNahidaGenshin_nahidav3.safetensors"
# pretrained_lora_path_2 = "/home/chenkaizheng/codes/vscode/waifu/webui_new/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/joint_lora_fixn1_sep_5e3-000012_2.safetensors"
# pretrained_lora_path = ""
pretrained_lora_path_2 = ""

# # 如果存在预训练路径2，并且lora1没有半固定
# if pretrained_lora_path_2 is not None and not semi_freeze_network_1:
#   ANNEAL_EPOCH = 0
#   HALF_ANNEAL_EPOCH = 0
# # 如果从头开训或者network1半固定
# else:
#   ANNEAL_EPOCH = 20 # 20轮前进行正常训练
#   HALF_ANNEAL_EPOCH = 11 #超过这个轮次后，对network_2添加正交损失
#   print("第{}轮开始全退火".format(ANNEAL_EPOCH))


def collate_fn(examples):
  return examples[0]


# TODO 他のスクリプトと共通化する
def generate_step_logs(args: argparse.Namespace, current_loss, avr_loss, lr_scheduler):
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if args.network_train_unet_only:
        logs["lr/unet"] = float(lr_scheduler.get_last_lr()[0])
    elif args.network_train_text_encoder_only:
        logs["lr/textencoder"] = float(lr_scheduler.get_last_lr()[0])
    else:
        logs["lr/textencoder"] = float(lr_scheduler.get_last_lr()[0])
        logs["lr/unet"] = float(lr_scheduler.get_last_lr()[-1])  # may be same to textencoder

    if args.optimizer_type.lower() == "DAdaptation".lower():  # tracking d*lr value of unet.
        logs["lr/d*lr"] = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]

    return logs

def prepare_accelerator(args: argparse.Namespace):
  if args.logging_dir_2 is None:
    log_with = None
    logging_dir = None
  else:
    log_with = "tensorboard"
    log_prefix = "" if args.log_prefix_2 is None else args.log_prefix_2
    logging_dir = args.logging_dir_2 + "/" + log_prefix + time.strftime('%Y%m%d%H%M%S', time.localtime())

  accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision,
                            log_with=log_with, logging_dir=logging_dir)

  # accelerateの互換性問題を解決する
  accelerator_0_15 = True
  try:
    accelerator.unwrap_model("dummy", True)
    print("Using accelerator 0.15.0 or above.")
  except TypeError:
    accelerator_0_15 = False

  def unwrap_model(model):
    if accelerator_0_15:
      return accelerator.unwrap_model(model, True)
    return accelerator.unwrap_model(model)

  return accelerator, unwrap_model


# Monkeypatch newer get_scheduler() function overridng current version of diffusers.optimizer.get_scheduler
# code is taken from https://github.com/huggingface/diffusers diffusers.optimizer, commit d87cc15977b87160c30abaace3894e802ad9e1e6
# Which is a newer release of diffusers than currently packaged with sd-scripts
# This code can be removed when newer diffusers version (v0.12.1 or greater) is tested and implemented to sd-scripts

from typing import Optional, Union
from torch.optim import Optimizer
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

def get_scheduler_fix(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
):
    """
    Unified API to get any scheduler from its name.
    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def train(args):
  # todo 是否重建session id
  session_id = random.randint(0, 2**32)
  training_started_at = time.time()
  train_util.verify_training_args(args)
  train_util.prepare_dataset_args(args, True)

  cache_latents = args.cache_latents
  use_dreambooth_method = args.in_json is None
  use_user_config = args.dataset_config is not None

  if args.seed is not None:
    set_seed(args.seed)

  # 准备两个tokenizer
  tokenizer1 = train_util.load_tokenizer(args)
  tokenizer2 = train_util.load_tokenizer(args)

  # データセットを準備する
  blueprint_generator_1 = BlueprintGenerator(ConfigSanitizer(True, True, True))
  blueprint_generator_2 = BlueprintGenerator(ConfigSanitizer(True, True, True))
  # TODO:暂时不需要 / currently not support this one
  if use_user_config:
    print(f"Load dataset config from {args.dataset_config}")
    user_config = config_util.load_user_config(args.dataset_config)
    ignored = ["train_data_dir", "reg_data_dir", "in_json"]
    if any(getattr(args, attr) is not None for attr in ignored):
        print(
            "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                ", ".join(ignored)
            )
        )
  else:
      if use_dreambooth_method:
          print("Use DreamBooth method.")
          user_config_1 = {
              "datasets": [
                  {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}
              ]
          }
          user_config_2 = {
            "datasets": [
                  {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir_2, args.reg_data_dir_2)}
              ]
          }
      else:
          print("Train with captions.")
          user_config = {
              "datasets": [
                  {
                      "subsets": [
                          {
                              "image_dir": args.train_data_dir,
                              "metadata_file": args.in_json,
                          }
                      ]
                  }
              ]
          }
  blueprint_1 = blueprint_generator_1.generate(user_config_1, args, tokenizer=tokenizer1)
  train_dataset_group_1 = config_util.generate_dataset_group_by_blueprint(blueprint_1.dataset_group)
  blueprint_2 = blueprint_generator_2.generate(user_config_2, args, tokenizer=tokenizer2)
  train_dataset_group_2 = config_util.generate_dataset_group_by_blueprint(blueprint_2.dataset_group)

  current_epoch = Value('i',0)
  current_step = Value('i',0)
  ds_for_collater_1 = train_dataset_group_1 if args.max_data_loader_n_workers == 0 else None
  collater_1 = train_util.collater_class(current_epoch,current_step, ds_for_collater_1)
  ds_for_collater_2 = train_dataset_group_2 if args.max_data_loader_n_workers == 0 else None
  collater_2 = train_util.collater_class(current_epoch,current_step, ds_for_collater_2)
  
  if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group_1)
        train_util.debug_dataset(train_dataset_group_2)
        return
  if len(train_dataset_group_1) == 0:
      print(
          "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
      )
      return
  if len(train_dataset_group_2) == 0:
      print(
          "No data found. Please verify arguments (train_data_dir_2 must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
      )
      return

  if cache_latents:
      assert (
          train_dataset_group_1.is_latent_cacheable()
      ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
      assert (
          train_dataset_group_2.is_latent_cacheable()
      ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"


  # acceleratorを準備する
  print("prepare accelerator")
  accelerator_1, unwrap_model_1 = train_util.prepare_accelerator(args)
  accelerator_2, unwrap_model_2 = prepare_accelerator(args)
  is_main_process = accelerator_2.is_main_process

  # mixed precisionに対応した型を用意しておき適宜castする
  # 可以共用
  weight_dtype, save_dtype = train_util.prepare_dtype(args)

  # モデルを読み込む
  # 分别读取模型
  text_encoder_1, vae_1, unet_1, _ = train_util.load_target_model(args, weight_dtype)
  text_encoder_2, vae_2, unet_2, _ = train_util.load_target_model(args, weight_dtype)

  # モデルに xformers とか memory efficient attention を組み込む
  train_util.replace_unet_modules(unet_1, args.mem_eff_attn, args.xformers)
  train_util.replace_unet_modules(unet_2, args.mem_eff_attn, args.xformers)

  # 学習を準備する
  # vae cannot be reused in some circumstances
  if cache_latents:
    vae_1.to(accelerator_1.device, dtype=weight_dtype)
    vae_1.requires_grad_(False)
    vae_1.eval()
    with torch.no_grad():
      train_dataset_group_1.cache_latents(vae_1)
    vae_1.to("cpu")
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    vae_2.to(accelerator_1.device, dtype=weight_dtype)
    vae_2.requires_grad_(False)
    vae_2.eval()
    with torch.no_grad():
      train_dataset_group_2.cache_latents(vae_2)
    vae_2.to("cpu")
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    gc.collect()

  # prepare network
  print("import network module:", args.network_module)
  # can be used publicly
  network_module = importlib.import_module(args.network_module)

  net_kwargs = {}
  if args.network_args is not None:
    for net_arg in args.network_args:
      key, value = net_arg.split('=')
      net_kwargs[key] = value

  # if a new network is added in future, add if ~ then blocks for each network (;'∀')
  # 创建了两个lora / create two lora blocks
  network_1 = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae_1, text_encoder_1, unet_1, **net_kwargs)
  network_2 = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae_1, text_encoder_2, unet_2, **net_kwargs)

  if network_1 is None or network_2 is None:
    return

  # 将network_1设置为冻结模型 / set network_1 freezed
  if args.network_weights != "" and (args.freeze_network_1 or args.semi_freeze_network_1):
    print("load network_1 weights from:", args.network_weights)
    network_1.load_weights(args.network_weights)
    if args.network_weights_2 != "":
      print("load network_2 weights from:", args.network_weights_2)
      network_2.load_weights(args.network_weights_2)

  train_unet = not args.network_train_text_encoder_only
  train_text_encoder = not args.network_train_unet_only
  network_1.apply_to(text_encoder_1, unet_1, train_text_encoder, train_unet)
  network_2.apply_to(text_encoder_2, unet_2, train_text_encoder, train_unet)


  if args.gradient_checkpointing:
    unet_1.enable_gradient_checkpointing()
    text_encoder_1.gradient_checkpointing_enable()
    network_1.enable_gradient_checkpointing()                   # may have no effect
    unet_2.enable_gradient_checkpointing()
    text_encoder_2.gradient_checkpointing_enable()
    network_2.enable_gradient_checkpointing()

  # 学習に必要なクラスを準備する
  print("prepare optimizer, data loader etc.")

  # 8-bit Adamを使う
  if args.use_8bit_adam:
    try:
      import bitsandbytes as bnb
    except ImportError:
      raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
    print("use 8-bit Adam optimizer")
    optimizer_class = bnb.optim.AdamW8bit
  else:
    optimizer_class = torch.optim.AdamW

  trainable_params_1 = network_1.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
  trainable_params_2 = network_2.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

  optimizer_name_1, optimizer_args_1, optimizer_1 = train_util.get_optimizer(args, trainable_params_1)
  optimizer_name_2, optimizer_args_2, optimizer_2 = train_util.get_optimizer(args, trainable_params_2)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)      # cpu_count-1 ただし最大で指定された数まで
  
  train_dataloader_1 = torch.utils.data.DataLoader(
      train_dataset_group_1,
      batch_size=1,
      shuffle=True, 
      collate_fn=collater_1, 
      num_workers=n_workers,
      persistent_workers=args.persistent_data_loader_workers
  )
  
  train_dataloader_2 = torch.utils.data.DataLoader(
      train_dataset_group_2, 
      batch_size=1, 
      shuffle=True, 
      collate_fn=collater_2, 
      num_workers=n_workers,
      persistent_workers=args.persistent_data_loader_workers
  )

  # 学習ステップ数を計算する
  if args.max_train_epochs is not None:
    # print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")
    args.max_train_steps = args.max_train_epochs * math.ceil(len(train_dataloader_1) / accelerator_1.num_processes / args.gradient_accumulation_steps)
    if is_main_process:
        print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

  lr_scheduler_1 = train_util.get_scheduler_fix(args, optimizer_1, accelerator_1.num_processes)
  lr_scheduler_2 = train_util.get_scheduler_fix(args, optimizer_2, accelerator_2.num_processes)

  # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
  if args.full_fp16:
    assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    print("enable full fp16 training.")
    network_1.to(weight_dtype)
    network_2.to(weight_dtype)

  # acceleratorがなんかよろしくやってくれるらしい
  if train_unet and train_text_encoder:
    unet_1, text_encoder_1, network_1, optimizer_1, train_dataloader_1, lr_scheduler_1 = accelerator_1.prepare(
        unet_1, text_encoder_1, network_1, optimizer_1, train_dataloader_1, lr_scheduler_1)
    unet_2, text_encoder_2, network_2, optimizer_2, train_dataloader_2, lr_scheduler_2 = accelerator_2.prepare(
        unet_2, text_encoder_2, network_2, optimizer_2, train_dataloader_2, lr_scheduler_2)
  elif train_unet:
    unet_1, network_1, optimizer_1, train_dataloader_1, lr_scheduler_1 = accelerator_1.prepare(
        unet_1, network_1, optimizer_1, train_dataloader_1, lr_scheduler_1)
    unet_2, network_2, optimizer_2, train_dataloader_2, lr_scheduler_2 = accelerator_2.prepare(
        unet_2, network_2, optimizer_2, train_dataloader_2, lr_scheduler_2)
  elif train_text_encoder:
    text_encoder_1, network_1, optimizer_1, train_dataloader_1, lr_scheduler_1 = accelerator_1.prepare(
        text_encoder_1, network_1, optimizer_1, train_dataloader_1, lr_scheduler_1)
    text_encoder_2, network_2, optimizer_2, train_dataloader_2, lr_scheduler_2 = accelerator_2.prepare(
        text_encoder_2, network_2, optimizer_2, train_dataloader_2, lr_scheduler_2)
  else:
    network_1, optimizer_1, train_dataloader_1, lr_scheduler_1 = accelerator_1.prepare(
        network_1, optimizer_1, train_dataloader_1, lr_scheduler_1)
    network_2, optimizer_2, train_dataloader_2, lr_scheduler_2 = accelerator_2.prepare(
        network_2, optimizer_2, train_dataloader_2, lr_scheduler_2)

  # 将网络固定住 \ freeze the network parameters
  unet_1.requires_grad_(False)
  unet_1.to(accelerator_1.device, dtype=weight_dtype)
  unet_2.requires_grad_(False)
  unet_2.to(accelerator_2.device, dtype=weight_dtype)
  text_encoder_1.requires_grad_(False)
  text_encoder_1.to(accelerator_1.device, dtype=weight_dtype)
  text_encoder_2.requires_grad_(False)
  text_encoder_2.to(accelerator_2.device, dtype=weight_dtype)
  # 固定或者半固定都会在训练初始化的时候将网络固定住 \
  #  Both freezed and semi-freezed will freeze the network during training initialization
  if args.freeze_network_1 or args.semi_freeze_network_1:
    network_1.requires_grad_(False)
  if args.gradient_checkpointing:                       # according to TI example in Diffusers, train is required
    unet_1.train()
    text_encoder_1.train()
    unet_2.train()
    text_encoder_2.train()

    # set top parameter requires_grad = True for gradient checkpointing works
    if type(text_encoder_1) == DDP:
      text_encoder_1.module.text_model.embeddings.requires_grad_(True)
    else:
      text_encoder_1.text_model.embeddings.requires_grad_(True)
    if type(text_encoder_2) == DDP:
      text_encoder_2.module.text_model.embeddings.requires_grad_(True)
    else:
      text_encoder_2.text_model.embeddings.requires_grad_(True)
  else:
    unet_1.eval()
    unet_2.eval()
    text_encoder_1.eval()
    text_encoder_2.eval()
  
  # support DistributedDataParallel
  if type(text_encoder_1) == DDP:
    text_encoder_1 = text_encoder_1.module
    unet_1 = unet_1.module
    network_1 = network_1.module
  if type(text_encoder_2) == DDP:
    text_encoder_2 = text_encoder_2.module
    unet_2 = unet_2.module
    network_2 = network_2.module

  if args.freeze_network_1 or args.semi_freeze_network_1:
    print("network_1冻住啦 / network_1 is freezed")
    pass
  else:
    network_1.prepare_grad_etc(text_encoder_1, unet_1)
  network_2.prepare_grad_etc(text_encoder_2, unet_2)


  if not cache_latents:
    vae_1.requires_grad_(False)
    vae_1.eval()
    vae_1.to(accelerator_1.device, dtype=weight_dtype)
    vae_2.requires_grad_(False)
    vae_2.eval()
    vae_2.to(accelerator_2.device, dtype=weight_dtype)

  # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
  if args.full_fp16:
    train_util.patch_accelerator_for_fp16_training(accelerator_1)
    train_util.patch_accelerator_for_fp16_training(accelerator_2)

  # resumeする
  if args.resume is not None:
    print(f"resume training from state: {args.resume}")
    accelerator_1.load_state(args.resume)
    accelerator_2.load_state(args.resume)

  # epoch数を計算する
  # 软限定，两个训练模型的图片数量需要一致，这样的话此参数可以共用
  num_update_steps_per_epoch = math.ceil(len(train_dataloader_1) / args.gradient_accumulation_steps)
  num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
  if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
    args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

  # 学習する
  total_batch_size = args.train_batch_size * accelerator_1.num_processes * args.gradient_accumulation_steps
  if is_main_process:
      print("running training / 学習開始")
      print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group_1.num_train_images}")
      print(f"  num reg images / 正則化画像の数: {train_dataset_group_1.num_reg_images}")
      print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader_1)}")
      print(f"  num epochs / epoch数: {num_train_epochs}")
      print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group_2.datasets])}")
      print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group_2.num_train_images}")
      print(f"  num reg images / 正則化画像の数: {train_dataset_group_2.num_reg_images}")
      print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader_2)}")
      print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group_2.datasets])}")
      # print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
      print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
      print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

  # TODO refactor metadata creation and move to util
  metadata_1 = {
      "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
      "ss_training_started_at": training_started_at,  # unix timestamp
      "ss_output_name": args.output_name,
      "ss_learning_rate": args.learning_rate,
      "ss_text_encoder_lr": args.text_encoder_lr,
      "ss_unet_lr": args.unet_lr,
      "ss_num_train_images": train_dataset_group_1.num_train_images,
      "ss_num_reg_images": train_dataset_group_1.num_reg_images,
      "ss_num_batches_per_epoch": len(train_dataloader_1),
      "ss_num_epochs": num_train_epochs,
      "ss_gradient_checkpointing": args.gradient_checkpointing,
      "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
      "ss_max_train_steps": args.max_train_steps,
      "ss_lr_warmup_steps": args.lr_warmup_steps,
      "ss_lr_scheduler": args.lr_scheduler,
      "ss_network_module": args.network_module,
      "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
      "ss_network_alpha": args.network_alpha,  # some networks may not use this value
      "ss_mixed_precision": args.mixed_precision,
      "ss_full_fp16": bool(args.full_fp16),
      "ss_v2": bool(args.v2),
      "ss_clip_skip": args.clip_skip,
      "ss_max_token_length": args.max_token_length,
      "ss_cache_latents": bool(args.cache_latents),
      "ss_seed": args.seed,
      "ss_lowram": args.lowram,
      "ss_noise_offset": args.noise_offset,
      "ss_training_comment": args.training_comment,  # will not be updated after training
      "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
      "ss_optimizer": optimizer_name_1 + (f"({optimizer_args_1})" if len(optimizer_args_1) > 0 else ""),
      "ss_max_grad_norm": args.max_grad_norm,
      "ss_caption_dropout_rate": args.caption_dropout_rate,
      "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
      "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
      "ss_face_crop_aug_range": args.face_crop_aug_range,
      "ss_prior_loss_weight": args.prior_loss_weight,
  }

  metadata_2 = {
      "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
      "ss_training_started_at": training_started_at,  # unix timestamp
      "ss_output_name": args.output_name,
      "ss_learning_rate": args.learning_rate,
      "ss_text_encoder_lr": args.text_encoder_lr,
      "ss_unet_lr": args.unet_lr,
      "ss_num_train_images": train_dataset_group_2.num_train_images,
      "ss_num_reg_images": train_dataset_group_2.num_reg_images,
      "ss_num_batches_per_epoch": len(train_dataloader_2),
      "ss_num_epochs": num_train_epochs,
      "ss_gradient_checkpointing": args.gradient_checkpointing,
      "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
      "ss_max_train_steps": args.max_train_steps,
      "ss_lr_warmup_steps": args.lr_warmup_steps,
      "ss_lr_scheduler": args.lr_scheduler,
      "ss_network_module": args.network_module,
      "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
      "ss_network_alpha": args.network_alpha,  # some networks may not use this value
      "ss_mixed_precision": args.mixed_precision,
      "ss_full_fp16": bool(args.full_fp16),
      "ss_v2": bool(args.v2),
      "ss_clip_skip": args.clip_skip,
      "ss_max_token_length": args.max_token_length,
      "ss_cache_latents": bool(args.cache_latents),
      "ss_seed": args.seed,
      "ss_lowram": args.lowram,
      "ss_noise_offset": args.noise_offset,
      "ss_training_comment": args.training_comment,  # will not be updated after training
      "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
      "ss_optimizer": optimizer_name_2 + (f"({optimizer_args_2})" if len(optimizer_args_2) > 0 else ""),
      "ss_max_grad_norm": args.max_grad_norm,
      "ss_caption_dropout_rate": args.caption_dropout_rate,
      "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
      "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
      "ss_face_crop_aug_range": args.face_crop_aug_range,
      "ss_prior_loss_weight": args.prior_loss_weight,
  }

  if use_user_config:
      # save metadata of multiple datasets
      # NOTE: pack "ss_datasets" value as json one time
      #   or should also pack nested collections as json?
      datasets_metadata = []
      tag_frequency = {}  # merge tag frequency for metadata editor
      dataset_dirs_info = {}  # merge subset dirs for metadata editor

      for dataset in train_dataset_group.datasets:
          is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
          dataset_metadata = {
              "is_dreambooth": is_dreambooth_dataset,
              "batch_size_per_device": dataset.batch_size,
              "num_train_images": dataset.num_train_images,  # includes repeating
              "num_reg_images": dataset.num_reg_images,
              "resolution": (dataset.width, dataset.height),
              "enable_bucket": bool(dataset.enable_bucket),
              "min_bucket_reso": dataset.min_bucket_reso,
              "max_bucket_reso": dataset.max_bucket_reso,
              "tag_frequency": dataset.tag_frequency,
              "bucket_info": dataset.bucket_info,
          }

          subsets_metadata = []
          for subset in dataset.subsets:
              subset_metadata = {
                  "img_count": subset.img_count,
                  "num_repeats": subset.num_repeats,
                  "color_aug": bool(subset.color_aug),
                  "flip_aug": bool(subset.flip_aug),
                  "random_crop": bool(subset.random_crop),
                  "shuffle_caption": bool(subset.shuffle_caption),
                  "keep_tokens": subset.keep_tokens,
              }

              image_dir_or_metadata_file = None
              if subset.image_dir:
                  image_dir = os.path.basename(subset.image_dir)
                  subset_metadata["image_dir"] = image_dir
                  image_dir_or_metadata_file = image_dir

              if is_dreambooth_dataset:
                  subset_metadata["class_tokens"] = subset.class_tokens
                  subset_metadata["is_reg"] = subset.is_reg
                  if subset.is_reg:
                      image_dir_or_metadata_file = None  # not merging reg dataset
              else:
                  metadata_file = os.path.basename(subset.metadata_file)
                  subset_metadata["metadata_file"] = metadata_file
                  image_dir_or_metadata_file = metadata_file  # may overwrite

              subsets_metadata.append(subset_metadata)

              # merge dataset dir: not reg subset only
              # TODO update additional-network extension to show detailed dataset config from metadata
              if image_dir_or_metadata_file is not None:
                  # datasets may have a certain dir multiple times
                  v = image_dir_or_metadata_file
                  i = 2
                  while v in dataset_dirs_info:
                      v = image_dir_or_metadata_file + f" ({i})"
                      i += 1
                  image_dir_or_metadata_file = v

                  dataset_dirs_info[image_dir_or_metadata_file] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}

          dataset_metadata["subsets"] = subsets_metadata
          datasets_metadata.append(dataset_metadata)

          # merge tag frequency:
          for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
              # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
              # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
              # なので、ここで複数datasetの回数を合算してもあまり意味はない
              if ds_dir_name in tag_frequency:
                  continue
              tag_frequency[ds_dir_name] = ds_freq_for_dir

      metadata["ss_datasets"] = json.dumps(datasets_metadata)
      metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
      metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
  else:
      # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
      assert (
          len(train_dataset_group_1.datasets) == 1
      ), f"There should be a single dataset but {len(train_dataset_group_1.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

      dataset_1 = train_dataset_group_1.datasets[0]
      dataset_2 = train_dataset_group_2.datasets[0]


      dataset_dirs_info_1 = {}
      reg_dataset_dirs_info_1 = {}
      dataset_dirs_info_2 = {}
      reg_dataset_dirs_info_2 = {}
      if use_dreambooth_method:
          for subset in dataset_1.subsets:
              info = reg_dataset_dirs_info_1 if subset.is_reg else dataset_dirs_info_1
              info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
          for subset in dataset_2.subsets:
              info = reg_dataset_dirs_info_2 if subset.is_reg else dataset_dirs_info_2
              info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
      else:
          for subset in dataset_1.subsets:
              dataset_dirs_info_1[os.path.basename(subset.metadata_file)] = {
                  "n_repeats": subset.num_repeats,
                  "img_count": subset.img_count,
              }
          for subset in dataset_2.subsets:
              dataset_dirs_info_2[os.path.basename(subset.metadata_file)] = {
                  "n_repeats": subset.num_repeats,
                  "img_count": subset.img_count,
              }

      metadata_1.update(
          {
              "ss_batch_size_per_device": args.train_batch_size,
              "ss_total_batch_size": total_batch_size,
              "ss_resolution": args.resolution,
              "ss_color_aug": bool(args.color_aug),
              "ss_flip_aug": bool(args.flip_aug),
              "ss_random_crop": bool(args.random_crop),
              "ss_shuffle_caption": bool(args.shuffle_caption),
              "ss_enable_bucket": bool(dataset_1.enable_bucket),
              "ss_bucket_no_upscale": bool(dataset_1.bucket_no_upscale),
              "ss_min_bucket_reso": dataset_1.min_bucket_reso,
              "ss_max_bucket_reso": dataset_1.max_bucket_reso,
              "ss_keep_tokens": args.keep_tokens,
              "ss_dataset_dirs": json.dumps(dataset_dirs_info_1),
              "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info_1),
              "ss_tag_frequency": json.dumps(dataset_1.tag_frequency),
              "ss_bucket_info": json.dumps(dataset_1.bucket_info),
          }
      )

      metadata_2.update(
          {
              "ss_batch_size_per_device": args.train_batch_size,
              "ss_total_batch_size": total_batch_size,
              "ss_resolution": args.resolution,
              "ss_color_aug": bool(args.color_aug),
              "ss_flip_aug": bool(args.flip_aug),
              "ss_random_crop": bool(args.random_crop),
              "ss_shuffle_caption": bool(args.shuffle_caption),
              "ss_enable_bucket": bool(dataset_2.enable_bucket),
              "ss_bucket_no_upscale": bool(dataset_2.bucket_no_upscale),
              "ss_min_bucket_reso": dataset_2.min_bucket_reso,
              "ss_max_bucket_reso": dataset_2.max_bucket_reso,
              "ss_keep_tokens": args.keep_tokens,
              "ss_dataset_dirs": json.dumps(dataset_dirs_info_1),
              "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info_1),
              "ss_tag_frequency": json.dumps(dataset_2.tag_frequency),
              "ss_bucket_info": json.dumps(dataset_2.bucket_info),
          }
      )

  # add extra args
  if args.network_args:
      metadata_1["ss_network_args"] = json.dumps(net_kwargs)
      metadata_2["ss_network_args"] = json.dumps(net_kwargs)
      # for key, value in net_kwargs.items():
      #   metadata["ss_arg_" + key] = value

  # model name and hash
  if args.pretrained_model_name_or_path is not None:
      sd_model_name_1 = args.pretrained_model_name_or_path
      sd_model_name_2 = args.pretrained_model_name_or_path_2
      if os.path.exists(sd_model_name_1):
          metadata_1["ss_sd_model_hash"] = train_util.model_hash(sd_model_name_1)
          metadata_1["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name_1)
          sd_model_name_1 = os.path.basename(sd_model_name_1)
      metadata_1["ss_sd_model_name"] = sd_model_name_1
      if os.path.exists(sd_model_name_2):
          metadata_2["ss_sd_model_hash"] = train_util.model_hash(sd_model_name_2)
          metadata_2["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name_2)
          sd_model_name_2 = os.path.basename(sd_model_name_2)
      metadata_2["ss_sd_model_name"] = sd_model_name_2

  if args.vae is not None:
      vae_name = args.vae
      if os.path.exists(vae_name):
          metadata_1["ss_vae_hash"] = train_util.model_hash(vae_name)
          metadata_1["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
          vae_name = os.path.basename(vae_name)
      metadata_1["ss_vae_name"] = vae_name

  metadata_1 = {k: str(v) for k, v in metadata_1.items()}
  metadata_2 = {k: str(v) for k, v in metadata_2.items()}

  # make minimum metadata for filtering
  minimum_keys = ["ss_network_module", "ss_network_dim", "ss_network_alpha", "ss_network_args"]
  minimum_metadata = {}
  for key in minimum_keys:
      if key in metadata_1:
          minimum_metadata[key] = metadata_1[key]

  progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator_1.is_local_main_process, desc="steps")
  global_step = 0

  noise_scheduler_1 = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  num_train_timesteps=1000, clip_sample=False)
  noise_scheduler_2 = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  num_train_timesteps=1000, clip_sample=False)

  if accelerator_1.is_main_process:
    accelerator_1.init_trackers("network_train_1")
  if accelerator_2.is_main_process:
    accelerator_2.init_trackers("network_train_2")

  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")
    metadata_1["ss_epoch"] = str(epoch+1)
    metadata_2["ss_epoch"] = str(epoch+1)

    # 进行退火算法
    if epoch < args.anneal_epoch:
      kld_weight = 0
      orth_weight_unet = 0
      orth_weight_te = 0
      if epoch > args.half_anneal_epoch:
        print("network2添加正交loss啦！")
        kld_weight = args.kld_coefficient
        orth_weight_unet = args.orth_unet_coefficient
        orth_weight_te = args.orth_te_coefficient
    else:
      kld_weight = args.kld_coefficient
      orth_weight_unet = args.orth_unet_coefficient
      orth_weight_te = args.orth_te_coefficient
      if args.semi_freeze_network_1:
        print("network_1解禁啦！")
        network_1.requires_grad_(True)
        network_1.prepare_grad_etc(text_encoder_1, unet_1)

    network_1.on_epoch_start(text_encoder_1, unet_1)
    network_2.on_epoch_start(text_encoder_2, unet_2)

    loss_total_1 = 0
    loss_total_2 = 0
    kld_txt = 0
    kld_unet_mid = 0
    epoch_length = max(len(train_dataloader_1), len(train_dataloader_2))
    for step, batch in enumerate(zip(cycle(train_dataloader_1), cycle(train_dataloader_2))):
      # Terminate the infinite loop
      if step == epoch_length:
         break
      with accelerator_1.accumulate(network_1):
        with accelerator_2.accumulate(network_2):
          with torch.no_grad():
            if "latents" in batch[0] and batch[0]["latents"] is not None:
              latents_1 = batch[0]["latents"].to(accelerator_1.device)
            else:
              # latentに変換
              latents_1 = vae_1.encode(batch[0]["images"].to(dtype=weight_dtype)).latent_dist.sample()
            latents_1 = latents_1 * 0.18215
            if "latents" in batch[1] and batch[1]["latents"] is not None:
              latents_2 = batch[1]["latents"].to(accelerator_2.device)
            else:
              # latentに変換
              latents_2 = vae_1.encode(batch[1]["images"].to(dtype=weight_dtype)).latent_dist.sample()
            latents_2 = latents_2 * 0.18215
          b_size = latents_1.shape[0]

          with torch.set_grad_enabled(train_text_encoder):
            # Get the text embedding for conditioning
            input_ids_1 = batch[0]["input_ids"].to(accelerator_1.device)
            encoder_hidden_states_1 = train_util.get_hidden_states(args, input_ids_1, tokenizer1, text_encoder_1, weight_dtype)
            input_ids_2 = batch[1]["input_ids"].to(accelerator_2.device)
            encoder_hidden_states_2 = train_util.get_hidden_states(args, input_ids_2, tokenizer2, text_encoder_2, weight_dtype)


          # Sample noise that we'll add to the latents
          # TODO:添加给latent的噪声是否需要一致？
          # 应该是不一致的，因为两个网络需要独立训练，只是在loss上有联系
          noise_1 = torch.randn_like(latents_1, device=latents_1.device)
          noise_2 = torch.randn_like(latents_2, device=latents_2.device)


          # Sample a random timestep for each image
          # 这个随机时间步选的很有哲学，需要注意统一
          timesteps = torch.randint(0, noise_scheduler_1.config.num_train_timesteps, (b_size,), device=latents_1.device)
          timesteps = timesteps.long()

          # Add noise to the latents according to the noise magnitude at each timestep
          # (this is the forward diffusion process)
          noisy_latents_1 = noise_scheduler_1.add_noise(latents_1, noise_1, timesteps)
          noisy_latents_2 = noise_scheduler_2.add_noise(latents_2, noise_2, timesteps)

          # Predict the noise residual
          noise_pred_1 = unet_1(noisy_latents_1, timesteps, encoder_hidden_states_1).sample
          noise_pred_2 = unet_2(noisy_latents_2, timesteps, encoder_hidden_states_2).sample

          if args.v_parameterization:
            # 感觉没咋用
            # v-parameterization training
            target_1 = noise_scheduler_1.get_velocity(latents_1, noise_1, timesteps)
            target_2 = noise_scheduler_2.get_velocity(latents_2, noise_2, timesteps)
          else:
            target_1 = noise_1
            target_2 = noise_2
          

          # kl_d = torch.nn.functional.kl_div(
          #   encoder_hidden_states_1.float().softmax(dim=-1).log(),# 非target分布需要加log
          #   encoder_hidden_states_2.float().softmax(dim=-1),
          #   reduction="batchmean"
          #   )
          unet_hidden_1 = network_1.lora_unet_mid_block_attentions_0_proj_out.hidden_dict["lora_unet_mid_block_attentions_0_proj_out"]
          unet_hidden_2 = network_2.lora_unet_mid_block_attentions_0_proj_out.hidden_dict["lora_unet_mid_block_attentions_0_proj_out"]
          
          # 对text encoder的输出求散度
          hidden_1 = network_1.lora_te_text_model_encoder_layers_9_mlp_fc2.hidden_dict[
            'lora_te_text_model_encoder_layers_9_mlp_fc2'] # N*C*H 1*77*768
          hidden_2 = network_2.lora_te_text_model_encoder_layers_9_mlp_fc2.hidden_dict[
            'lora_te_text_model_encoder_layers_9_mlp_fc2']
          
          # 对隐藏层求kl散度
          kl_d_txt = torch.nn.functional.kl_div(
            hidden_1.float().softmax(dim=-1).log(),# 非target分布需要加log
            hidden_2.float().softmax(dim=-1),
            reduction="batchmean"
            )
          
          kl_d_unet = torch.nn.functional.kl_div(
            unet_hidden_1.float().softmax(dim=-1).log(),# 非target分布需要加log
            unet_hidden_2.float().softmax(dim=-1),
            reduction="batchmean"
            )

          
          # text_encoder正交正则项
          organal_squad = torch.bmm(hidden_1, hidden_2.permute(0,2,1))
          # ones = torch.ones(hidden_1.shape[0], hidden_1.shape[1], hidden_1.shape[1],dtype=torch.float32,device='cuda')
          diag = torch.eye(hidden_1.shape[1], dtype=torch.float32,device='cuda')
          loss_orth_te = ((organal_squad * diag.to("cuda"))**2).mean()
          # # 修改text_encoder正交正则项求法
          # organal_squad = torch.bmm(hidden_1.permute(0,2,1), hidden_2)
          # # ones = torch.ones(hidden_1.shape[0], hidden_1.shape[1], hidden_1.shape[1],dtype=torch.float32,device='cuda')
          # diag = torch.eye(hidden_1.shape[2], dtype=torch.float32,device='cuda')
          # loss_orth_te = ((organal_squad * diag.to("cuda"))**2).mean()
          # TODO:下一次就是尝试直接按照图像的正交去处理特征了




          # unet正交正则项, 隐向量是4维向量(1, 1280, 8, 8)
          organal_squad_unet = torch.bmm(
            unet_hidden_1.view(unet_hidden_1.shape[0]*unet_hidden_1.shape[1], unet_hidden_1.shape[2], unet_hidden_1.shape[3]),
            unet_hidden_2.view(unet_hidden_1.shape[0]*unet_hidden_1.shape[1], unet_hidden_1.shape[2], unet_hidden_1.shape[3]).permute(0,2,1))# 
          ones_unet = torch.ones(unet_hidden_1.shape[0]*unet_hidden_1.shape[1], unet_hidden_1.shape[2], unet_hidden_1.shape[3],dtype=torch.float32,device='cuda')
          diag_unet = torch.eye(unet_hidden_1.shape[2], dtype=torch.float32,device='cuda')
          loss_orth_unet = ((organal_squad_unet * (ones_unet - diag_unet))**2).mean()
          # loss_orth_unet = ((organal_squad_unet * diag_unet)**2).mean()

          # 删除kld
          loss_1 = torch.nn.functional.mse_loss(noise_pred_1.float(), target_1.float(), reduction="none")
          loss_1 = loss_1.mean([1, 2, 3]) + orth_weight_te * loss_orth_te + orth_weight_unet * loss_orth_unet
          loss_2 = torch.nn.functional.mse_loss(noise_pred_2.float(), target_2.float(), reduction="none")
          loss_2 = loss_2.mean([1, 2, 3]) + orth_weight_te * loss_orth_te + orth_weight_unet * loss_orth_unet

          loss_weights_1 = batch[0]["loss_weights"]                      # 各sampleごとのweight
          loss_1 = loss_1 * loss_weights_1

          loss_1 = loss_1.mean()                # 平均なのでbatch_sizeで割る必要なし

          loss_weights_2 = batch[1]["loss_weights"]                      # 各sampleごとのweight
          loss_2 = loss_2 * loss_weights_2

          loss_2 = loss_2.mean()

          accelerator_1.backward(loss_1, retain_graph=True)
          if accelerator_1.sync_gradients:
            params_to_clip_1 = network_1.get_trainable_params()
            accelerator_1.clip_grad_norm_(params_to_clip_1, 1.0)  # args.max_grad_norm)
          
          # backward second network
          accelerator_2.backward(loss_2)
          if accelerator_2.sync_gradients:
            params_to_clip_2 = network_2.get_trainable_params()
            accelerator_2.clip_grad_norm_(params_to_clip_2, 1.0)

          optimizer_1.step()
          lr_scheduler_1.step()
          optimizer_1.zero_grad(set_to_none=True)
          optimizer_2.step()
          lr_scheduler_2.step()
          optimizer_2.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      # 检查
      if accelerator_1.sync_gradients and accelerator_2.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss_1 = loss_1.detach().item()
      current_loss_2 = loss_2.detach().item()
      current_kld_txt = kl_d_txt.detach().item()
      current_kld_unet = kl_d_unet.detach().item()

      loss_total_1 += current_loss_1
      loss_total_2 += current_loss_2
      kld_txt += current_kld_txt
      kld_unet_mid += current_kld_unet
      
      avr_loss_1 = loss_total_1 / (step+1)
      avr_loss_2 = loss_total_2 / (step+1)
      avr_kld_txt = kld_txt / (step+1)
      avr_kld_unet = kld_unet_mid / (step+1)
      logs = {
        "loss_1": avr_loss_1,
        "loss_2": avr_loss_2,
        "kld_txt": avr_kld_txt,
        "kld_unet": avr_kld_unet
        }  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)

      if args.logging_dir is not None:
        logs = generate_step_logs(args, current_loss_1, avr_loss_1, lr_scheduler_1)
        accelerator_1.log(logs, step=global_step)
        accelerator_2.log(logs, step=global_step)
        # for name, param in network.named_parameters():
          # locon_writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch+1)
          # locon_writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch+1)

      if global_step >= args.max_train_steps:
        break

    if args.logging_dir is not None:
      logs = {
        "loss_1/epoch": loss_total_1 / len(train_dataloader_1),
        "loss_2/epoch": loss_total_2 / len(train_dataloader_2),
        "kld_txt/epoch": kld_txt / len(train_dataloader_1),
        "kld_unet/epoch": kld_unet_mid / len(train_dataloader_1),
        }
      accelerator_1.log(logs, step=epoch+1)
      accelerator_2.log(logs, step=epoch+1)
      # for name, param in network.named_parameters():
        # locon_writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch+1)
        # locon_writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch+1)
      

    accelerator_1.wait_for_everyone()
    accelerator_2.wait_for_everyone()

    if args.save_every_n_epochs is not None:
      # name of the saving checkpoint
      model_name = train_util.DEFAULT_EPOCH_NAME if args.output_name is None else args.output_name
      
      # 因为要保存两个模型，所以要改写一下
      def save_func():
        ckpt_name_1 = train_util.EPOCH_FILE_NAME.format(model_name, epoch + 1) + "_1" + '.' + args.save_model_as
        ckpt_name_2 = train_util.EPOCH_FILE_NAME.format(model_name, epoch + 1) + "_2" + '.' + args.save_model_as
        ckpt_file_1 = os.path.join(args.output_dir, ckpt_name_1)
        ckpt_file_2 = os.path.join(args.output_dir, ckpt_name_2)
        print(f"saving checkpoint: {ckpt_file_1} and {ckpt_file_2}")
        unwrap_model_1(network_1).save_weights(ckpt_file_1, save_dtype, None if args.no_metadata else metadata_1)
        unwrap_model_2(network_2).save_weights(ckpt_file_2, save_dtype, None if args.no_metadata else metadata_2)

      def remove_old_func(old_epoch_no):
        old_ckpt_name = train_util.EPOCH_FILE_NAME.format(model_name, old_epoch_no) + '.' + args.save_model_as
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
          print(f"removing old checkpoint: {old_ckpt_file}")
          os.remove(old_ckpt_file)

      saving = train_util.save_on_epoch_end(args, save_func, remove_old_func, epoch + 1, num_train_epochs)
      if saving and args.save_state:
        train_util.save_state_on_epoch_end(args, accelerator_1, model_name, epoch + 1)


    # end of epoch

  metadata_1["ss_epoch"] = str(num_train_epochs)
  metadata_2["ss_epoch"] = str(num_train_epochs)

  is_main_process = accelerator_1.is_main_process
  if is_main_process:
    network_1 = unwrap_model_1(network_1)
    network_2 = unwrap_model_2(network_2)

  accelerator_1.end_training()
  accelerator_2.end_training()

  if args.save_state:
    train_util.save_state_on_train_end(args, accelerator_1)
    train_util.save_state_on_train_end(args, accelerator_2)

  del accelerator_1
  del accelerator_2                        # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = train_util.DEFAULT_LAST_OUTPUT_NAME if args.output_name is None else args.output_name
    ckpt_name_1 = model_name + "_1" + '.' + args.save_model_as
    ckpt_name_2 = model_name + "_1" + '.' + args.save_model_as
    ckpt_file_1 = os.path.join(args.output_dir, ckpt_name_1)
    ckpt_file_2 = os.path.join(args.output_dir, ckpt_name_2)

    print(f"save trained model to {ckpt_file_1} and {ckpt_file_2}")
    network_1.save_weights(ckpt_file_1, save_dtype, None if args.no_metadata else metadata_1)
    network_2.save_weights(ckpt_file_2, save_dtype, None if args.no_metadata else metadata_2)
    print("model saved.")
  

# 疯狂debug
def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")

    parser.add_argument("--network_weights", type=str, default=pretrained_lora_path, help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_weights_2", type=str, default=pretrained_lora_path_2, help="pretrained weights for network2 / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=network_module, help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument(
        "--network_dim", type=int, default=network_dim, help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）"
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=network_alpha,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*", help="additional argmuments for network (key=value) / ネットワークへの追加の引数"
    )
    parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument(
        "--network_train_text_encoder_only", action="store_true", help="only training Text Encoder part / Text Encoder関連部分のみ学習する"
    )
    parser.add_argument(
        "--training_comment", type=str, default=None, help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列"
    )
    parser.add_argument(
      "--logging_dir_2", type=str, default=None, help="针对第二个训练网络的日志记录 / log during training of the second network"
    )
    parser.add_argument(
      "--log_prefix_2", type=str, default=None, help="针对第二个网络的prefix / prefix for network2"
    )
    parser.add_argument(
      "--pretrained_model_name_or_path_2", type=str, default=model_path_2,
      help="pretrained base model for network_2 to train, directory to Diffusers model or StableDiffusion checkpoint / 针对第二个网络的模型预加载位置"
    )
    # 为了不修改其他文件，就在这里加train_data_dir
    # In order to avoid modifying other files, add new parameters here for the second network:
    parser.add_argument(
      "--train_data_dir_2", type=str, default=train_data_dir_2, help="directory for train images of network2 /  针对第二个网络的训练数据"
    )
    parser.add_argument(
      "--reg_data_dir_2", type=str, default=reg_data_dir_2, help="directory for regularization images of network2 / 针对第二个网络的正则数据"
    )
    # have no effect after experiments
    parser.add_argument(
      "--kld_coefficient", type=str, default=0, help="the coefficient of the KL divergence"
    )
    parser.add_argument(
      "--orth_te_coefficient", type=int, default=1, help="the coefficient of the orthogonal loss between the two outputs from text encoder"
    )
    parser.add_argument(
      "--orth_unet_coefficient", type=int, default=1, help="the coefficient of the orthogonal loss between the two outputs from unet_mid"
    )
    parser.add_argument(
      "--freeze_network_1", type=bool, default=False, help="固定network1，只训练network2 / Freeze network1 and only train network2."
    )
    parser.add_argument(
      "--semi_freeze_network_1", type=bool, default=False, help="先固定network1，只训练network2，之后在特定轮次后允许训练network1，但是这样会比较耗费显存 /\
        First, freeze network1 and train only network2. Later, after a specific number of epochs,\
          allow training network1 as well. However, this can be memory-intensive"
    )
    parser.add_argument(
      "--anneal_epoch", type=int, default=0, help="在该轮次开始，对network1和network2都在loss上加上正则项进行训练 / \
        Starting from this epoch, train network1 and network2 with a regularization term added to the loss"
    )
    parser.add_argument(
      "--half_annel_epoch", type=int, default=0, help="在该轮次开始后，固定network1网络，但是network2的loss需要加上正则项进行训练 / \
        After this epoch, keep network1 freezed and add a regularization term to the loss of network2 during training."
    )
    return parser





if __name__ == '__main__':
  parser = setup_parser()
  args = parser.parse_args()
  args = train_util.read_config_from_file(args, parser)
  train(args)