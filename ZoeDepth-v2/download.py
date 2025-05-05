import torch

repo = "isl-org/ZoeDepth"
# Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Zoe_K
model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)

# Zoe_NK
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)