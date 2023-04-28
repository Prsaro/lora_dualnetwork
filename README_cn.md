# LoRA Dualnetworks Train
- 基于Stable Diffusion v1.5的LoRA模型训练脚本
- 使用双网络进行LoRA模型训练
- 基于kohya的[sd-scripts](https://github.com/kohya-ss/sd-scripts)改写

# 概述
受到[lora-block-weight](https://raw.githubusercontent.com/hako-mikan/sd-webui-lora-block-weight/)对LoRA进行分层控制的启发，开发了这个脚本。SD的LoRA模型在进行两个模型以及多个模型叠加时会产生特征融合以及崩坏，这大大限制了LoRA模型的使用场景，为了减缓这个问题，我们提出了新的LoRA训练方法，让两个LoRA模型同时训练，这样可以在训练的时候让两个LoRA互相约束，从而让了两个LoRA的特征在高维空间分离，减弱特征耦合的现象。

## 依赖
与原[sd-scripts](https://github.com/kohya-ss/sd-scripts)一致
These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.)
The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

## 项目原理
在原始DDPM的loss上添加了新的正交损失，如下面的公式所示
$$
L = (\epsilon - \epsilon_\theta)^2 + \lambda (h_1h_2^T\bigodot(1-I))^2
$$
其中h为两个独立LoRA网络相互对应的隐藏层。
![](/img/dualnetwork.jpg)
在训练的时候推荐预训练好network1，冻结network1后再训练network2，这样可以缓解LoRA模型之间的特征融合。


## 网络训练
参数配置与[sd-scripts](https://github.com/kohya-ss/sd-scripts)一致
目前只支持使用Dreambooth的训练方式，暂时不支持finetune的in-json方式训练,也不支持.toml加载数据集。
```
accelerate launch --num_cpu_threads_per_process 1 train_dualnetwork.py
    --pretrained_model_name_or_path=<network1的基础模型.ckpt, .safetensor或者Diffusers支持的模型>
    --pretrained_model_name_or_path=<network2的基础模型.ckpt, .safetensor或者Diffusers支持的模型>
    --output_dir=<模型训练的输出路径>  
    --output_name=<模型训练的输出名字> 
    --save_model_as=safetensors
    --prior_loss_weight=1.0 
    --max_train_steps=2000
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --save_every_n_epochs=1 
    --network_module=networks.lora
    --network_dim=64
    --network_alpha=64
    --train_data_dir=<>
    --reg_data_dir=<>
    --train_data_dir_2=<>
    --reg_data_dir_2=<>
    --orth_te_coefficient=1
    --orth_unet_coefficient=1
    --freeze_network_1=False
    --semi_freeze_network_1=False
    --anneal_epoch=0
    --half_annel_epoch=0
```