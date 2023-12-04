# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for selecting best image among generated samples."""

import os
import click
import numpy as np
from tqdm import tqdm
from glob import glob
import random

import io
from torchvision.utils import make_grid, save_image
import torch

#----------------------------------------------------------------------------

def select_topk(input_dir, output_dir, best_of_n=10, topk=1):

    files = glob(os.path.join(input_dir, 'samples*.npz'))
    random.shuffle(files)
    count = 0

    for file_ind, filei in tqdm(enumerate(files)):
        try:
            data = np.load(filei)
        except:
            # remove file
            os.remove(filei)
            continue
        images, ll_ratio = data["samples"], data["ll_ratio"]
        if 'label' in data.files:
            class_labels = data["label"]

        if images.shape[0] % best_of_n != 0:
            images = images[:images.shape[0] // best_of_n * best_of_n]
            ll_ratio = ll_ratio[:ll_ratio.shape[0] // best_of_n * best_of_n]
        images = images.reshape(images.shape[0] // best_of_n, best_of_n, *images.shape[1:])
        ll_ratio = ll_ratio.reshape(ll_ratio.shape[0] // best_of_n, best_of_n)
        
        all_inds = np.argsort(ll_ratio, axis=1)
        inds = all_inds[:, -topk]
        images = images[np.arange(images.shape[0]), inds]
        # save the best sample
        with open(os.path.join(output_dir, f"samples_{file_ind}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=images)
            fout.write(io_buffer.getvalue())
        
        nrow = int(np.sqrt(images.shape[0]))
        image_grid = make_grid(torch.tensor(images).permute(0, 3, 1, 2) / 255., nrow, padding=2)
        with open(os.path.join(output_dir, f"sample_{file_ind}.png"), "wb") as fout:
            save_image(image_grid, fout)
        
        count = count + images.shape[0]
        if count >= 50000:
            break

    print(count)


#----------------------------------------------------------------------------
@click.command()
@click.option('--indir', 'input_dir', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--outdir', 'output_dir', help='Dataset reference statistics ', metavar='NPZ|URL',  type=str, required=True)
@click.option('--best_of_n',            help='Pick best out of n images', metavar='INT',          type=click.IntRange(min=1), default=4)

def main(input_dir, output_dir, best_of_n):
    os.makedirs(output_dir, exist_ok=True)
    select_topk(input_dir, output_dir, best_of_n=best_of_n)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
