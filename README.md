<div align="center">


# LoRA Dualnetworks
**Style Fusion Through Dual Path Low Rank Adaptation**


______________________________________________________________________

<p align="center">
  <a href="">Arxiv</a> •
  <a href="#dependency">Dependency</a> •
<a href="#training">Training</a> •
<a href="#demo">Testing</a> •
  <a href="#pre-trained-models-and-results">Logs</a> •
  <a href="#citation">Citation</a><br>
 </p>

[![python](https://img.shields.io/badge/python-%20%203.9-blue.svg)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/duanyiqun/DiffusionDepth/blob/main/LICENSE)



</div>

Taking inspiration from the hierarchical control of LoRA using [lora-block-weight](https://raw.githubusercontent.com/hako-mikan/sd-webui-lora-block-weight/), we developed this script. In the SD LoRA model, feature fusion and collapse can occur when multiple LoRA models are stacked, greatly limiting the use cases of the LoRA model. To mitigate this problem, we propose a new LoRA training method that involves simultaneously training two LoRA models. This allows the two LoRA models to mutually constrain each other during training, thus separating the features of the two LoRA models in high-dimensional space and reducing the coupling of features.

______________________________________________________________________


## Features 
- Training script for the LoRA model based on Stable Diffusion v1.5
- Training the LoRA model using dual networks
- Base on kohya's [sd-scripts](https://github.com/kohya-ss/sd-scripts)


## Dependency
Same as [sd-scripts](https://github.com/kohya-ss/sd-scripts)
These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.)
The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

## Method Brief
We added a new orthogonal loss to the original DDPM loss, as shown in the following formula
$$L = (\epsilon - \epsilon_\theta)^2 + \lambda (h_1h_2^T\bigodot(1-I))^2$$

Here, h refers to the hidden layer of two independent LoRA networks that correspond to each other.
![](/img/dualnetwork_theory.png)
We select the hidden layers from the output of text encoder and the mid-layer of Unet as our chosen hidden layers h. The influence of the mid-layer of Unet is more significant in this case.
![](/img/dualnetwork.jpg)

It is recommended to pretrain network1 and then freeze it before training network2. This can alleviate the feature fusion between the LoRA models during training.

## Train
Parameter configuration is same as [sd-scripts](https://github.com/kohya-ss/sd-scripts).
Currently, only training with Dreambooth is supported. Fine-tuning using in-json training mode and loading datasets with .toml format are not tested.
```
accelerate launch --num_cpu_threads_per_process 1 train_dualnetwork.py
    --pretrained_model_name_or_path=<base model of network1 in .ckpt, .safetensor, or models supported by Diffusers>
    --pretrained_model_name_or_path=<base model of network2 in .ckpt, .safetensor, or models supported by Diffusers>
    --output_dir=<output path for model training>  
    --output_name=<output name for model training>
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

## Demo & Checkpoints
Used the same prompt and fixed the seed, with the only difference being the addition of Hanfu lora model or not.
### Checkpoints: 
We release the hanfu LoRA trained with Nahida LoRA.
[hanfu_nahida](https://civitai.com/models/55161/hanfunahidadualnetworktrain)

We finetuned a LoRA from a popular [hanfu LoRA](https://civitai.com/models/8029/elegant-hanfu-ruqun-style) and uploaded it into civitai:[hanfu_ruqun](https://civitai.com/models/68524?modelVersionId=73210).

### Demo 
<div align="center">
<img src="img/nihida_base.png" width = "200" height =  alt="图片名称" align=center/><img src="img/nahida_common_loss.jpeg" width = "200" height =  alt="图片名称" align=center/><img src="img/nihida_hanfu.png" width = "200" height =  alt="图片名称" align=center/>
</div>

From left to right, the pictures are Nahida without LoRA, Nahida with hanfu LoRA trained with common loss, Nahida with hanfu LoRA trained with orthogonal loss. Can easily transform the clothes of Nahida into hanfu.
<div align="center">
<img src="img/sangyan_base.jpeg" width = "200" height =  alt="图片名称" align=center/><img src="img/sangyan_common_loss.jpeg" width = "200" height =  alt="图片名称" align=center/><img src="img/sangyan_hanfu.png" width = "200" height =  alt="图片名称" align=center/>
</div>
From left to right, the pictures are Sangyan without LoRA, Sangyan with hanfu LoRA trained with common loss, Sangyan with hanfu LoRA trained with orthogonal loss.
Changed the character's clothing without affecting their appearance and movements.

<div align="center">
<img src="img/15736-2460371662-hanfu, (ru_qun), tree, pool, white hair, green eyes, nahida_genshin, child, masterpiece, best quality.png" width = "200" height =  alt="图片名称" align=center/><img src="img/15735-2460371662-hanfu, (ru_qun), tree, pool, white hair, green eyes, nahida_genshin, child, masterpiece, best quality.png" width = "200" height =  alt="图片名称" align=center/>
</div>
We finetuned a popular hanfu LoRA model in civitai to test effectiveness.The left one is before finetuned, and the right one is after finetuned(same seed and prompts).

@ kkworld