#!/bin/bash

# Run our particle filtering method
# $1: dataset name
# $2: diffusion sampler
# $3: num_steps or restart config index

echo "Running on $1"
echo "Diffusion sampler: $2"

# Check the dataset name
if [ "$1" == "imagenet" ]; then
    net=checkpoints/pretrained_score/edm-imagenet-64x64-cond-adm.pkl
    discriminator=checkpoints/discriminator/discriminator_imagenet.pt
    cond=1

    # Check the diffusion sampler
    if [ "$2" == "restart" ]; then
        S_max=1.0
        S_min=0.01
        S_churn=0.0
        dg_weight=1.3
        restart_info=$(sed "${3}q;d" restart_params_imagenet.txt)
    elif [ "$2" == "edm" ]; then
        S_max=50
        S_min=0.05
        S_churn=((64 / 256.0 * 40))
        dg_weight=1.1
    else
        echo "Diffusion sampler must be either 'restart' or 'edm'"
        exit 1
    fi
elif [ "$1" == "ffhq" ]; then
    net=checkpoints/pretrained_score/edm-ffhq-64x64-uncond-vp.pkl
    discriminator=checkpoints/discriminator/discriminator_ffhq.pt
    cond=0
    dg_weight=1.0

    if [ "$2" == "restart" ]; then
        S_max=0.0
        S_min=0.01
        S_churn=0.0
        restart_info=$(sed "${3}q;d" restart_params_ffhq.txt)
    elif [ "$2" == "edm" ]; then
        S_max=50
        S_min=0.05
        S_churn=((64 / 256.0 * 10))
    else
        echo "Diffusion sampler must be either 'restart' or 'edm'"
        exit 1
    fi
else
    echo "Dataset must be either 'imagenet' or 'ffhq'"
    exit 1
fi

# Generate samples
if [ "$2" == "restart" ]; then
    python3 generate.py --network ${net} --outdir=samples/dg/${1}_${2}_${3} \
        --sampler $2 --method dg --cond=${cond} --dg_weight_1st_order=${dg_weight} \
        --discriminator_ckpt=${discriminator} --restart_info='${restart_info}' \
        --S_churn=${S_churn} --S_min=${S_min} --S_max=${S_max} --S_noise=1.003 \
        --num_particles 1 --seeds 0-49999 --batch 100 --resample_inds=-1 --boosting 1
else
    python3 generate.py --network ${net} --outdir=samples/dg/${1}_${2}_${3} \
        --sampler $2 --method dg --cond=${cond} --dg_weight_1st_order=${dg_weight} \
        --discriminator_ckpt=${discriminator} --steps $3 \
        --S_churn=${S_churn} --S_min=${S_min} --S_max=${S_max} --S_noise=1.003 \
        --num_particles 1 --seeds 0-49999 --batch 100 --resample_inds=-1 --boosting 1
fi
