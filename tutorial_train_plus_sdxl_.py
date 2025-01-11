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

# 追加のライブラリ
from safetensors.torch import load_file as safe_load_file
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from schedulefree import AdamWScheduleFree # 追加のスケジューラー
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast
from ip_adapter.resampler import Resampler

import torchvision
import warnings
warnings.filterwarnings("ignore")


def debug_noise_scheduler(scheduler):
    print("Debug: Noise Scheduler Configuration")
    print(f"  Beta End: {scheduler.betas[-1].item() if hasattr(scheduler, 'betas') else 'Not Defined'}")
    print(f"  Beta Schedule: {scheduler.config.beta_schedule if hasattr(scheduler, 'config') and 'beta_schedule' in scheduler.config else 'Unknown'}")
    print(f"  Variance Type: {scheduler.config.variance_type if hasattr(scheduler, 'config') and 'variance_type' in scheduler.config else 'Unknown'}")
    print("-" * 50)

def debug_image_proj_model_dtype(model, message=""):
    """
    デバッグ用: image_proj_model の全パラメータの dtype を表示する。
    :param model: チェック対象の image_proj_model。
    :param message: 任意のデバッグメッセージ。
    """
    if model is not None:
        print(f"\n[DEBUG] image_proj_model dtype check: {message}")
        for name, param in model.named_parameters():
            print(f"  - {name}: {param.dtype}")
    else:
        print(f"\n[DEBUG] image_proj_model is None: {message}")

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

        # チェックポイントパスが指定されている場合、モデルの重みをロードする
        if ckpt_path is not None:
            if ckpt_path.endswith(".safetensors"):
                # safetensors ファイルのロード
                print(f"{ckpt_path}からモデルをGPUにロードしました")
                self.load_from_checkpoint(ckpt_path)
            else:
                # それ以外の形式（例: .ckpt, .pth）は torch.load() を使用
                print(f"Loading model weights from {ckpt_path} using torch.")
                self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # チェックポイントのロード
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = safe_load_file(ckpt_path, device=device)

        # 不要なキーをリストアップ
        keys_to_remove = [
            "image_proj.proj.bias",
            "image_proj.norm.weight",
            "image_proj.proj.weight",
            "image_proj.norm.bias",
        ]

        # state_dict から不要なキーを削除
        for key in keys_to_remove:
            if key in state_dict:
                # print(f"Removing unused parameter: {key}")
                del state_dict[key]

        # --------------------------------------------------------------------
        # 不一致パラメータを “部分的にコピー＆パディング” で対応する関数
        def resize_tensor(param: torch.Tensor, expected_shape: torch.Size) -> torch.Tensor:
            # 2D（線形レイヤ weight 等）なら [out_features, in_features]
            if len(param.shape) == 2:
                old_out, old_in = param.shape
                new_out, new_in = expected_shape

                # 新しいテンソルを 0 で作成（dtype を param に揃える -> fp16）
                resized_param = param.new_zeros(new_out, new_in).to(param.dtype)

                # コピーする範囲を決定 (小さい方に合わせる)
                out_to_copy = min(old_out, new_out)
                in_to_copy = min(old_in, new_in)

                # 必要分だけコピー
                resized_param[:out_to_copy, :in_to_copy] = param[:out_to_copy, :in_to_copy]

                return resized_param

            # 1D（bias 等）なら [out_features]
            elif len(param.shape) == 1:
                old_len = param.shape[0]
                new_len = expected_shape[0]

                if new_len == old_len:
                    return param
                elif new_len < old_len:
                    # 超過分を切り捨て
                    return param[:new_len]
                else:
                    # 足りない分を 0 でパディング
                    extra = new_len - old_len
                    pad_zeros = param.new_zeros(extra).to(param.dtype)
                    return torch.cat([param, pad_zeros], dim=0)

            else:
                # 今回の例では 2D, 1D 以外は想定外
                raise ValueError(f"Unsupported tensor shape: {param.shape}")

        # --------------------------------------------------------------------
        # 1. "image_proj_model" 用の state_dict を整形
        state_dict_image_proj = {
            key.replace("image_proj.", ""): value
            for key, value in state_dict.items()
            if key.startswith("image_proj.")
        }

        # debug_image_proj_model_dtype(self.image_proj_model, "Before load_with_resize")
        # 2. リサイズ付きロード処理
        def load_with_resize(model, ckpt_dict):
            model_dict = model.state_dict()
            new_state_dict = {}

            for k, v in ckpt_dict.items():
                if k in model_dict:
                    expected_param = model_dict[k]
                    expected_shape = expected_param.shape

                    # まず dtype をモデル側 (fp16) に合わせる
                    v = v.to(expected_param.dtype)

                    # 形状が一致していればそのまま使用
                    if v.shape == expected_shape:
                        new_state_dict[k] = v
                    else:
                        try:
                            # print(f"Resizing {k}: {tuple(v.shape)} -> {tuple(expected_shape)}")
                            resized_param = resize_tensor(v, expected_shape)
                            new_state_dict[k] = resized_param
                        except Exception as e:
                            print(f"Skipping {k}: resize failed with error {e}")
                else:
                    # モデルに存在しないパラメータはスキップ
                    print(f"Skipping {k}: not found in model state_dict")

            # strict=False でロード
            model.load_state_dict(new_state_dict, strict=False)

            # すべてのパラメータを fp16 に変換
            model.half()

        # image_proj_model に対して読み込み
        load_with_resize(self.image_proj_model, state_dict_image_proj)
        # debug_image_proj_model_dtype(self.image_proj_model, "Before load_with_resize")

        # --------------------------------------------------------------------
        # 3. "adapter_modules" 用の state_dict を整形
        state_dict_adapter = {
            key.replace("ip_adapter.", ""): value
            for key, value in state_dict.items()
            if key.startswith("ip_adapter.")
        }

        # adapter_modules に対して読み込み
        load_with_resize(self.adapter_modules, state_dict_adapter)

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

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
    parser.add_argument("--num_tokens", type=int, default=4, help="Number of image tokens for Resampler.")
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
    debug_noise_scheduler(noise_scheduler)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    #ip-adapter
    num_tokens = args.num_tokens  # コマンドライン引数から取得
    image_proj_model = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    image_proj_model = image_proj_model.half()

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
    # params = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())

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
    # optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = AdamWScheduleFree(params_to_opt, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, warmup_steps=num_training_steps // 2)

    # スケジューラーを定義
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=num_training_steps,  # 学習が終了するまでのステップ数
    #     eta_min=5e-5  # 最小学習率
    # )

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

                with accelerator.accumulate(ip_adapter):
                    # Convert images to latent space
                    with autocast():
                        # VAE のエンコード
                        latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                    # Sample noise to add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)

                    # Add noise to the latents
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Encode image embeddings
                    with torch.no_grad():
                        image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                    image_embeds_ = [
                        torch.zeros_like(image_embed) if drop_image_embed == 1 else image_embed
                        for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"])
                    ]
                    image_embeds = torch.stack(image_embeds_).half()
                    
                    with torch.autocast("cuda", dtype=torch.float16):
                        image_tokens = image_proj_model(image_embeds)

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
                    
                    cross_attention_inputs = torch.cat([text_embeds, image_tokens], dim=1) # テキストと画像トークンを結合

                    # Forward pass through IP-Adapter
                    noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    # Gather and calculate average loss
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                    # Record loss and learning rate in TensorBoard
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + step)
                    writer.add_scalar('LearningRate', current_lr, epoch * len(train_dataloader) + step)

                    # Backpropagation
                    accelerator.backward(loss)
                    optimizer.zero_grad()  # 勾配をリセット
                    scheduler.step()

                # tqdm の進行状況を更新
                pbar.set_postfix(loss=avg_loss)
                pbar.update(1)

                global_step += 1

                # 定期的にモデルを保存
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path, safe_serialization=False)
                    print(f"{global_step} ステップ後にモデルのチェックポイントが {save_path} に保存されました。")

if __name__ == "__main__":
    main()    
