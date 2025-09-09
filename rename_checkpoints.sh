cd /picassox/intelligent-cpfs/pixocial/jin.huang/models/Wan2.2-T2V-A14B

for f in diffusion_pytorch_model-*-bf16.safetensors; do
    mv "$f" "${f/-bf16/}"
done
