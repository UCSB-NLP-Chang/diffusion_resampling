import json
from PIL import Image
import shutil
import os
import argparse

import torch
from torch import nn
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np

from utils import get_np_indices, load_spacy_stopwords, load_discriminator, load_diffusion_model
from discriminator_lib import get_likelihood_ratio
import discriminator_lib
from resampling import calculate_caption_prob_ratio


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Text prompts
    parser.add_argument("--prompt_file", type=str, default='data/coco_object.json',
        help="File for text prmopts")
    parser.add_argument("--prompt_start", type=int, default=0,
        help="Start prompt index of the experiment")
    parser.add_argument("--prompt_end", type=int, default=-1,
        help="End prompt index of the experiment")
    
    # Sampling parameters
    parser.add_argument("--sampler", type=str, default='restart', choices=['restart', 'edm'],
        help="Which sampler to use.")
    parser.add_argument("--method", type=str, default='pf-hybrid', choices=['pf-hybrid', 'pf-discriminator', 'none'],
        help="Which method to use.")
    parser.add_argument("--diffusion_model_id", type=str, default='stabilityai/stable-diffusion-2-1-base',
        help="Which diffusion model to use.")
    parser.add_argument("--num_generation", type=int, default=10,
        help="Number of samples per caption")
    
    # Particle filter parameters
    parser.add_argument("--c0_prior", type=float, default=0.2,
        help="q(Oc = 0)")
    parser.add_argument("--use_obj_discriminator", action="store_true",
        help="Whether to use object occurrence as discriminator")
    
    # Other experiment parameters
    parser.add_argument("--input_figure_dir", type=str, default='figures',
        help="Directory of input figures")
    parser.add_argument("--output_figure_dir", type=str, default='figures_out',
        help="Directory of output figures")
    parser.add_argument("--device", type=str, default='cuda',
        help="Device to use")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    
    # Captions
    with open(args.prompt_file, 'r') as f:
        texts = json.load(f)
        texts = [text.strip() for text in texts]
        texts = [text for text in texts if len(text) > 0]
        dataset_name = args.prompt_file.split('/')[-1].split('.')[0]
        print(f"Loaded {len(texts)} captions from {args.prompt_file}")
        if args.prompt_end == -1:
            args.prompt_end = len(texts)

    # Diffusion model
    pipe = load_diffusion_model(args)
    vae_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    # Discriminator
    if not args.method == 'none':
        discriminator = load_discriminator(args)
        vpsde = discriminator_lib.vpsde(scaled_linear=True)
    
    if not args.method == 'pf-discriminator':
        nlp, stopwords_list = load_spacy_stopwords()
        # Object detector
        model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(args.device)
        model.eval()
        detector_transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Object occurrence probability
        prob_file = f'stats/{dataset_name}_probs_{args.sampler}.npy'
        object_occur_prob = torch.from_numpy(np.load(prob_file)).to(args.device)
        print(f"=============== Loaded object occurrence probability from {prob_file} ===============")
    
    with torch.no_grad():
        os.makedirs(args.output_figure_dir, exist_ok=True)
        
        for i, text in tqdm(enumerate(texts[args.prompt_start:args.prompt_end], start=args.prompt_start)):
            imgs = [Image.open(f'{args.input_figure_dir}/{i}/{k}_final.png').convert('RGB') for k in range(args.num_generation)]
            
            scores = 0
            if args.method != 'none':
                vae_imgs = [vae_transform(img).unsqueeze(0) for img in imgs]
                vae_imgs = torch.cat(vae_imgs, dim=0).to(args.device)
                latents = pipe.vae.encode(vae_imgs).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                prompt_embeds = pipe._encode_prompt(
                    text,
                    args.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True
                )
                # Discriminator
                discriminator_cond = prompt_embeds[[1]] if args.method == 'pf-discriminator' else prompt_embeds[[0]]
                discriminator.set_text_embed(discriminator_cond)
                sigma_cur = torch.tensor(1e-4, device=args.device)
                l_ratio, _ = get_likelihood_ratio(discriminator, vpsde, latents, sigma_cur, None, None)
                l_ratio = l_ratio.log().reshape(-1)
                scores += l_ratio
            
            if args.method != 'pf-discriminator':
                imgs = [detector_transform(img).unsqueeze(0).to(args.device) for img in imgs]
                imgs = torch.cat(imgs, 0)
                outputs = model(imgs)
                _, _, coco_indices = get_np_indices(text, nlp, stopwords_list)
                coco_indices = torch.tensor(coco_indices, device=args.device)
                
                log_probs = nn.functional.log_softmax(outputs['pred_logits'], -1)
                obj_logprobs = log_probs[:, :, coco_indices] # B, 100, N
                obj_logprobs = obj_logprobs.max(dim=1)[0] # B, N
                if args.method == 'pf-hybrid':
                    log_caption_prob_ratio = calculate_caption_prob_ratio([obj_logprobs], args, object_occur_prob[-1][coco_indices])
                elif args.method == 'none':
                    log_caption_prob_ratio = obj_logprobs.sum(dim=1)
                scores += log_caption_prob_ratio
            
            best_ind = scores.argmax().item()
            shutil.copy(f"{args.input_figure_dir}/{i}/{best_ind}_final.png", f"{args.output_figure_dir}/{i}.png")


if __name__ == "__main__":
    main()
