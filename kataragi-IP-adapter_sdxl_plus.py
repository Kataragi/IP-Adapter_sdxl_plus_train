import torchvision
torchvision.disable_beta_transforms_warning()

import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from tqdm import tqdm

import torch
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# 独自の追加ライブラリ
from torch.utils.tensorboard import SummaryWriter # tensorboadを実装
from safetensors.torch import load_file as safe_load_file # safetensorsモデルを読み込めるようにしておく
from ip_adapter.resampler import Resampler # resamplerをインポート
from schedulefree import AdamWScheduleFree # 追加のスケジューラー
from torch.optim.lr_scheduler import LambdaLR # 
from torch.cuda.amp import autocast
import torch.nn as nn

def debug_noise_scheduler(noise_scheduler):
    print("Noise Scheduler Debug Information:")
    # print(f"Betas: {noise_scheduler.betas}")
    print(f"Number of training timesteps: {noise_scheduler.config.num_train_timesteps}")

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
    
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        
        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }
    
    def __len__(self):
        return len(self.data)

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        # チェックポイントパスが指定されている場合、モデルをロード
        if ckpt_path is not None:
            if ckpt_path.endswith(".safetensors"):
                # safetensors形式の読み込み
                print(f"{ckpt_path} を safetensors形式でロードします...")
                self.load_from_checkpoint(ckpt_path, format="safetensors")
            elif ckpt_path.endswith(".bin"):
                # bin形式の読み込み
                print(f"{ckpt_path} を bin形式でロードします...")
                self.load_from_checkpoint(ckpt_path, format="bin")
            else:
                raise ValueError(f"未対応のチェックポイント形式: {ckpt_path}")

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_proj_model = self.image_proj_model.to(device)  # image_proj_modelをデバイスに移動
        self.adapter_modules.to(device)  # adapter_modulesをデバイスに移動

        # encoder_hidden_states の次元数をデバッグ出力
        print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}")

        # image_embeds の次元数をデバッグ出力
        print(f"image_embeds.shape: {image_embeds.shape}")

        # image_proj_model の構造とパラメータキー情報をデバッグ出力
        # print("image_proj_model structure:")
        # for name, param in self.image_proj_model.named_parameters():
            # print(f"  - {name}: shape {param.shape}, requires_grad={param.requires_grad}")

        # adapter_modules の構造とパラメータキー情報をデバッグ出力
        # print("adapter_modules structure:")
        # for name, param in self.adapter_modules.named_parameters():
            # print(f"  - {name}: shape {param.shape}, requires_grad={param.requires_grad}")

        # noisy_latents の次元数をデバッグ出力
        # print(f"noisy_latents.shape: {noisy_latents.shape}")

        # timesteps のデバッグ情報を表示 (テンソルの形状と値の一部)
        # print(f"timesteps.shape: {timesteps.shape}, timesteps[:5]: {timesteps[:5]}")

        # nn.Linear を使用して encoder_hidden_states の次元を拡張
        hidden_dim_expansion = torch.nn.Linear(768, 1280).to(encoder_hidden_states.device)

        # encoder_hidden_states の特徴次元を 768 -> 1280 に変換
        encoder_hidden_states = hidden_dim_expansion(encoder_hidden_states)
        print(f"encoder_hidden_states after expansion: {encoder_hidden_states.shape}")

        # image_proj_model を通じて ip_tokens を生成
        ip_tokens = self.image_proj_model(image_embeds)
        print(f"ip_tokens.shape before processing: {ip_tokens.shape}")

        # 次元を結合 (トークン次元 dim=1 で結合)
        #  => dimension 2 (特徴次元) が同じ 1280 なので、ここは問題なく結合できる
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1) # 1→2
        # encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=2) # 1→2
        print(f"encoder_hidden_states after concatenation: {encoder_hidden_states.shape}")

        noise_pred = self.unet(
               sample=noisy_latents,
               timestep=timesteps,
               encoder_hidden_states=encoder_hidden_states,
               added_cond_kwargs=unet_added_cond_kwargs
           ).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str, format: str):
        # モデルの重み変更を確認するための元のチェックサムを計算
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # チェックポイントのロード
        if format == "safetensors":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = safe_load_file(ckpt_path, device=device)  # safetensorsからロード
        elif format == "bin":
            state_dict = torch.load(ckpt_path, map_location="cpu")  # binからロード
        else:
            raise ValueError(f"未対応のフォーマット: {format}")

        if "image_proj" in state_dict and "ip_adapter" in state_dict:
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
            print("image_proj と ip_adapter の重みを正常にロードしました")
        else:
            raise KeyError("チェックポイントに必要なキー 'image_proj' または 'ip_adapter' が見つかりません")

        # 新しいチェックサムを計算し、変更があったか確認
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        assert orig_ip_proj_sum != new_ip_proj_sum, "image_proj_model の重みに変更がありません"
        assert orig_adapter_sum != new_adapter_sum, "adapter_modules の重みに変更がありません"

        # 正常にロードされたことをデバッグメッセージで確認
        print(f"チェックポイント {ckpt_path} から重みを正常にロードしました")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.betas = torch.linspace(0.0001, 0.9999, noise_scheduler.config.num_train_timesteps)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # デバッグ出力
    # debug_noise_scheduler(noise_scheduler)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)

    #ip-adapter-sdxl-plus
    num_tokens = 16
    image_proj_model = Resampler(
        # dim=1280, # unet.config.cross_attention_dim
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        # embedding_dim=image_encoder.config.hidden_size,
        # output_dim=unet.config.cross_attention_dim,
        # ff_mult=4
    )

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        print("mixed_precisionはfp16で実行されます")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        print("mixed_precisionはbf16で実行されます")
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())

    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_training_steps = len(train_dataloader) * args.num_train_epochs # 総step数を計算
    optimizer = AdamWScheduleFree(params_to_opt, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, warmup_steps=num_training_steps // 2)

    # 学習率スケジューラーの定義
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # ウォームアップ期間: 1e-7 から learning_rate まで線形増加
            return 1e-7 / args.learning_rate + (current_step / warmup_steps) * (1 - 1e-7 / args.learning_rate)
        # ウォームアップ後は一定
        return 1.0

    warmup_steps = num_training_steps // 2  # ウォームアップステップを全体の50%
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ログを保存するディレクトリ
    log_dir = './output_model/log'
    writer = SummaryWriter(log_dir)
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    global_step = 0
    total_steps = len(train_dataloader) * args.num_train_epochs

    with tqdm(total=total_steps, desc="Training Progress", unit="step") as pbar:
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                # 現在の学習率を確認（デバッグ用）
                current_lr = optimizer.param_groups[0]['lr']

                # Optimizer をトレーニングモードに設定
                optimizer.train()

                # IP-Adapter モデルの学習を加速器 (Accelerator) を用いて行う
                with accelerator.accumulate(ip_adapter):
                    # 画像を潜在空間 (latent space) に変換
                    with autocast():  # 混合精度 (Mixed Precision) による高速化
                        # VAE を用いて画像をエンコード
                        latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    # 潜在変数をスケーリング (VAE の設定に基づくスケーリング係数を掛ける)
                    latents = latents * vae.config.scaling_factor
                    # 潜在変数を指定されたデータ型に変換してデバイスに転送
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                    # 潜在変数に加えるノイズをサンプリング
                    noise = torch.randn_like(latents)  # 潜在変数と同じ形状のランダムなノイズを生成
                    if args.noise_offset:  # ノイズのオフセットが設定されている場合
                        noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(
                            accelerator.device, dtype=weight_dtype
                        )

                    # バッチサイズを取得
                    bsz = latents.shape[0]
                    # 各サンプルにランダムなタイムステップを割り当て
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)

                    # 潜在変数にノイズを加える
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    clip_images = []
                    for clip_image, drop_image_embed in zip(batch["clip_images"], batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            clip_images.append(torch.zeros_like(clip_image))
                        else:
                            clip_images.append(clip_image)
                    clip_images = torch.stack(clip_images, dim=0)

                    # 画像埋め込みをエンコード
                    with torch.no_grad():  # 埋め込み生成中は勾配計算を無効化
                        image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]

                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                   # Encode text embeddings
                    with torch.no_grad():
                        encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                        text_embeds = encoder_output.hidden_states[-2]
                        encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                        pooled_text_embeds = encoder_output_2[0]
                        text_embeds_2 = encoder_output_2.hidden_states[-2]
                        text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)

                    # Add conditional inputs
                    add_time_ids = torch.cat([
                        batch["original_size"].to(accelerator.device),
                        batch["crop_coords_top_left"].to(accelerator.device),
                        batch["target_size"].to(accelerator.device)
                    ], dim=1).to(accelerator.device, dtype=weight_dtype)
                    unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                    # IP-Adapter による順伝播
                    noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds)
                    # ノイズ予測の損失 (MSE) を計算
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    # 損失を収集して平均損失を計算
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                    # TensorBoard に損失と学習率を記録
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + step)
                    writer.add_scalar('LearningRate', current_lr, epoch * len(train_dataloader) + step)

                    # 逆伝播を実行
                    accelerator.backward(loss)

                    # optimizer.step()  # スケジューラー不要のため直接ステップ実行
                    optimizer.zero_grad()  # 勾配をリセット
                    scheduler.step()

                # tqdm の進行状況を更新
                pbar.set_postfix(loss=avg_loss)
                pbar.update(1)

                global_step += 1

                # 定期的にモデルを保存
                if global_step % args.save_steps == 0:
                    optimizer.eval()  # チェックポイント保存時は評価モードに設定
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path, safe_serialization=False)
                    print(f"{global_step} ステップ後にモデルのチェックポイントが {save_path} に保存されました。")

if __name__ == "__main__":
    main()    
