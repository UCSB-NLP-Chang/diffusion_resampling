# Download pre-trained classifier
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
# Move to checkpoints/ADM_classifier
mkdir -p checkpoints/ADM_classifier
mv 64x64_classifier.pt checkpoints/ADM_classifier

# Download pre-trained denoising model
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
# Move to checkpoints/pretrained_score
mkdir -p checkpoints/pretrained_score
mv edm-imagenet-64x64-cond-adm.pkl checkpoints/pretrained_score
mv edm-ffhq-64x64-uncond-vp.pkl checkpoints/pretrained_score

# Download stats files
wget https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz
wget https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz
# Move to stats
mkdir -p stats
mv imagenet-64x64.npz stats
mv ffhq-64x64.npz stats
