#!/bin/bash

# Run our particle filtering method
# $1: dataset name
# $2: diffusion sampler
# $3: number of particles

echo "Running on $1"
echo "Diffusion sampler: $2"
echo "Number of particles: $3"

num_images=$(($3 * 50000 - 1))
resample_inds=-1

# Check the dataset name
if [ "$1" == "imagenet" ]; then
    net=checkpoints/pretrained_score/edm-imagenet-64x64-cond-adm.pkl
    discriminator=checkpoints/discriminator/discriminator_imagenet.pt
    cond=1
    steps=64

    # Check the diffusion sampler
    if [ "$2" == "restart" ]; then
        S_max=1.0
        S_min=0.01
        S_churn=0.0
    elif [ "$2" == "edm" ]; then
        S_max=50
        S_min=0.05
        S_churn=$(echo "scale=2; 64 / 256.0 * 40" | bc)
    else
        echo "Diffusion sampler must be either 'restart' or 'edm'"
        exit 1
    fi
elif [ "$1" == "ffhq" ]; then
    net=checkpoints/pretrained_score/edm-ffhq-64x64-uncond-vp.pkl
    discriminator=checkpoints/discriminator/discriminator_ffhq.pt
    cond=0
    steps=32

    if [ "$2" == "restart" ]; then
        S_max=0.0
        S_min=0.01
        S_churn=0.0
    elif [ "$2" == "edm" ]; then
        S_max=50
        S_min=0.05
        S_churn=$(echo "scale=2; 64 / 256.0 * 10" | bc)
    else
        echo "Diffusion sampler must be either 'restart' or 'edm'"
        exit 1
    fi
else
    echo "Dataset must be either 'imagenet' or 'ffhq'"
    exit 1
fi

# Generate samples
python3 generate.py --network ${net} --outdir=samples/ds/${1}_${2}_${3} \
    --sampler $2 --method none --cond=${cond} --dg_weight_1st_order=0.0 \
    --discriminator_ckpt=${discriminator} \
    --restart_info='18; {"0": [3, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 4, 0.59, 1.09], "3": [4, 1, 0.30, 0.59], "4": [4, 4, 0.06, 0.30]}' \
    --steps ${steps} --S_churn=${S_churn} --S_min=${S_min} --S_max=${S_max} --S_noise=1.003 \
    --num_particles $3 --seeds 0-${num_images} --batch 100 --resample_inds=${resample_inds}

# Select best image
python select_top_k.py --indir samples/ds/${1}_${2}_${3} --outdir samples/ds/${1}_${2}_${3}/best --best_of_n $3
