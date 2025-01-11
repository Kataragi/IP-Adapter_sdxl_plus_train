import torch
from torchvision.transforms import Resize

def flatten_keys(state_dict, parent_key=""):
    """
    ネストされた辞書をフラットに展開する関数。
    """
    items = []
    for k, v in state_dict.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):  # ネストされた辞書の場合
            items.extend(flatten_keys(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

ckpt = r"/home/akishiro/IP-Adapter/output_model/checkpoint-50/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu", weights_only=True)

# フラットな辞書を作成
image_proj_sd = {}
ip_sd = {}

# 除外するキー
excluded_keys = [
    "image_proj.proj.bias",
    "image_proj.norm.weight",
    "image_proj.proj.weight",
    "image_proj.norm.bias"
]

for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        flat_image_proj = flatten_keys({k.replace("image_proj_model.", ""): sd[k]})
        for key, value in flat_image_proj.items():
            if key not in excluded_keys:
                image_proj_sd[key] = value
    elif k.startswith("adapter_modules"):
        flat_ip_adapter = flatten_keys({k.replace("adapter_modules.", ""): sd[k]})
        ip_sd.update(flat_ip_adapter)

# 展開された辞書を保存
torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter_plus_test_.bin")
