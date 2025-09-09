# import torch
# from flash_attn import flash_attn_func
#
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
#
# # Tiny test case
# q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
# k = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
# v = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
#
# out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

# make sure this is set before Python starts

# python - <<'EOF'
import torch
from flash_attn import flash_attn_func

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0))

q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)

out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
print("FlashAttention out:", out.shape)
# EOF
