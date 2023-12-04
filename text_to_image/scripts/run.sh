#!/bin/bash

# Run our particle filtering method
# $1: dataset name
# $2: diffusion sampler
# $3: method to use, pf-hybrid, pf-discriminator, or none
# $4: number of particles

echo "Running on $1"
echo "Diffusion sampler: $2"
echo "Method: $3"
echo "Number of particles: $4"

# Generate samples
if [ "$3" == "none" ]; then
    python generate.py --prompt_file data/${1}.json --sampler $2 --method $3 \
        --diffusion_model_id stabilityai/stable-diffusion-2-1-base --num_generation $4 \
        --batch_size $4 --figure_dir figures/${1}_${2}_${3}_${4}
else
    python generate.py --prompt_file data/${1}.json --sampler $2 --method $3 \
        --diffusion_model_id stabilityai/stable-diffusion-2-1-base --num_generation $4 \
        --batch_size $4 --figure_dir figures/${1}_${2}_${3}_${4} --resample_inds 10 13 16 19
fi

# Select best image
python select_best.py --prompt_file data/${1}.json --sampler $2 --method $3 \
    --diffusion_model_id stabilityai/stable-diffusion-2-1-base --num_generation $4 \
    --input_figure_dir figures/${1}_${2}_${3}_${4} --output_figure_dir figures/${1}_${2}_${3}_${4}_out
