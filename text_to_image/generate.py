import json
import os
import argparse

from torchvision import transforms
import numpy as np
import torch
from transformers import set_seed
from tqdm import tqdm

from utils import get_np_indices, load_spacy_stopwords, load_diffusion_model, load_discriminator
from resampling import calculate_weights, resample
import discriminator_lib


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Text prompts
    parser.add_argument("--prompt_file", type=str, default='data/objects.json',
        help="File for text prmopts")
    parser.add_argument("--prompt_start", type=int, default=0,
        help="Start prompt index of the experiment")
    parser.add_argument("--prompt_end", type=int, default=-1,
        help="End prompt index of the experiment")
    
    # Sampling parameters
    parser.add_argument("--method", type=str, default='pf-hybrid', choices=['pf-hybrid', 'pf-discriminator', 'none'],
        help="Which method to use for generation.")
    parser.add_argument("--sampler", type=str, default='restart', choices=['restart', 'edm'],
        help="Which sampler to use.")
    parser.add_argument("--diffusion_model_id", type=str, default='stabilityai/stable-diffusion-2-1-base',
        help="Which diffusion model to use.")
    parser.add_argument("--num_generation", type=int, default=10,
        help="Number of samples per caption")
    parser.add_argument("--num_step", type=int, default=24,
        help="Number of steps for denoising")
    parser.add_argument("--rho", type=float, default=7.0,
        help="rho for sampler")
    parser.add_argument("--S_noise", type=float, default=1.003,
        help="S_noise for sampler")
    parser.add_argument("--restart_info_ind", type=int, default=0,
        help="Which Restart configuration to use")
    parser.add_argument("--batch_size", type=int, default=-1,
        help="batch_size for generation")
    
    # Particle filter parameters
    parser.add_argument("--c0_prior", type=float, default=0.2,
        help="q(Oc = 0)")
    parser.add_argument("--use_obj_discriminator", action="store_true",
        help="Whether to use object occurrence as discriminator")
    parser.add_argument("--resample_inds", type=int, nargs='+', default=[],
        help="Time steps to do resampling in EDM sampler")
    
    # Other experiment parameters
    parser.add_argument("--figure_dir", type=str, default='figures',
        help="Directory to store generated images")
    parser.add_argument("--seed", type=int, default=0,
        help="Seed of the experiment")
    parser.add_argument("--device", type=str, default='cuda',
        help="Device to use")
    args = parser.parse_args()
    
    if args.batch_size == -1:
        args.batch_size = args.num_generation
    
    return args


def main():
    args = parse_args()
    
    # Sampling parameters
    if args.sampler == 'restart':
        gamma = 0.05
        t_min, t_max = 0.01, 1.0
    elif args.sampler == 'edm':
        gamma = 80 / 256
        t_min, t_max = 0.05, 11.0
    
    # Captions
    with open(args.prompt_file, 'r') as f:
        texts = json.load(f)
    texts = [text for text in texts if len(text) > 0]
    if args.prompt_end == -1:
        args.prompt_end = len(texts)
    
    if args.method == 'pf-hybrid':
        dataset_name = args.prompt_file.split('/')[-1].split('.')[0]
        object_occurrence_file = f'stats/{dataset_name}_probs_{args.sampler}.npy'
        object_occur_prob = torch.from_numpy(np.load(object_occurrence_file)).to(args.device)
        print(f"=============== Loaded object occurrence probability from {object_occurrence_file} ===============")
    else:
        object_occur_prob = None
    
    # Restart configurations
    if args.sampler == 'restart':
        with open('restart_params.txt', 'r') as fp:
            infos = fp.readlines()
            restart_info = infos[args.restart_info_ind]
            num_step, restart_list = restart_info.split('\t')
            args.num_step = int(num_step)
            restart_list = json.loads(restart_list.strip())
    
    # Diffusion model
    pipe = load_diffusion_model(args)
    
    # Denoise time steps
    step_indices = torch.arange(args.num_step+1, dtype=torch.float32, device=args.device)
    sigma_steps = (pipe.scheduler.init_noise_sigma ** (1 / args.rho) + step_indices / args.num_step * \
        (pipe.scheduler.sigma_min ** (1 / args.rho) - pipe.scheduler.init_noise_sigma ** (1 / args.rho))) ** args.rho
    sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
    
    if args.sampler == 'restart':
        restart_list = {int(torch.argmin(abs(sigma_steps - i[2]), dim=0)): i for i in restart_list}
        print(f"=============== Restart list: {restart_list} ===============")
    
    # Object detector
    detector_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(args.device)
    detr.eval()
    
    # Discriminator
    discriminator = load_discriminator(args)
    vpsde = discriminator_lib.vpsde(scaled_linear=True)
    
    nlp, stopwords_list = load_spacy_stopwords()
    
    def get_steps(min_t, max_t, num_steps, rho):
        step_indices = torch.arange(num_steps, dtype=torch.float, device=args.device)
        t_steps = (max_t ** (1 / rho) + step_indices / (num_steps - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
        return t_steps
            
    # Default height and width to unet
    height = width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    
    # Generate for each text
    with torch.no_grad():
        for text_ind, text_desc in enumerate(texts[args.prompt_start: args.prompt_end], start=args.prompt_start):
            
            os.makedirs(f'{args.figure_dir}/{text_ind}', exist_ok=True)
            exist_files = os.listdir(f'{args.figure_dir}/{text_ind}')
            if len(exist_files) == args.num_generation:
                print(f"Skipping {text_ind}")
                continue

            _, _, coco_indices = get_np_indices(text_desc, nlp, stopwords_list)
            coco_indices = torch.tensor(coco_indices, device=args.device)
            # Encode input prompt
            prompt_embeds = pipe._encode_prompt(
                text_desc,
                args.device,
                num_images_per_prompt=args.batch_size,
                do_classifier_free_guidance=True
            )
            if not args.use_obj_discriminator:
                discriminator_cond = prompt_embeds[[args.batch_size]] if args.method == 'pf-discriminator' else prompt_embeds[[0]]
                discriminator.set_text_embed(discriminator_cond)
            
            # Prepare latent variables
            set_seed(text_ind + args.seed * len(texts))
            num_channels_latents = pipe.unet.in_channels
            xt = pipe.prepare_latents(
                args.num_generation,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                args.device,
                None,
                None
            )
            
            # Resample stats
            prev_phi = torch.zeros(args.num_generation, device=args.device)
            total_restart_ind = 0
            
            # Denoising loop
            for step_ind, (sigma_cur, sigma_next) in tqdm(enumerate(zip(sigma_steps[:-1], sigma_steps[1:]))):
                
                if args.sampler == 'edm':
                    # Calculate weight and resample
                    if step_ind in args.resample_inds and 'pf' in args.method:
                        if object_occur_prob is not None:
                            step_object_occur_prob = object_occur_prob[total_restart_ind][coco_indices]
                            total_restart_ind += 1
                        else:
                            step_object_occur_prob = None
                                
                        sample_weights, prev_phi = calculate_weights(args, pipe, discriminator,
                            vpsde, detr, detector_normalize, xt, sigma_cur, prompt_embeds, coco_indices, step_object_occur_prob, prev_phi)
                        xt, prev_phi = resample(sample_weights, xt, args, prev_phi)
                
                    gamma_back = gamma if sigma_cur > t_min and sigma_cur < t_max else 0.0

                    # Increase noise temporarily.
                    if gamma_back > 0:
                        sigma_hat = sigma_cur + gamma_back * sigma_cur
                        xt = xt + (sigma_hat ** 2 - sigma_cur ** 2).sqrt() * args.S_noise * torch.randn_like(xt)
                        sigma_cur = sigma_hat
                
                xt_next = []
                num_samples = xt.shape[0]
                for sample_ind in range(0, num_samples, args.batch_size):
                    tmp_xt_next, _ = pipe.scheduler.step(
                        xt[sample_ind: sample_ind + args.batch_size],
                        sigma_cur,
                        sigma_next,
                        prompt_embeds,
                        second_order=args.sampler=='edm'
                    )
                    xt_next.append(tmp_xt_next)
                
                xt = torch.vstack(xt_next)
                
                # ================= restart ================== #
                if args.sampler == 'restart' and step_ind + 1 in restart_list.keys():
                    restart_idx = step_ind + 1
                    
                    for restart_iter in range(restart_list[restart_idx][1]):
                        if object_occur_prob is not None:
                            step_object_occur_prob = object_occur_prob[total_restart_ind][coco_indices]
                            total_restart_ind += 1
                        else:
                            step_object_occur_prob = None
                        new_t_steps = get_steps(min_t=sigma_steps[restart_idx], max_t=restart_list[restart_idx][-1],
                                                num_steps=restart_list[restart_idx][0], rho=args.rho)
                        
                        # ================== Resample and update weights ==================
                        if 'pf' in args.method:
                            # Calculate weights
                            sample_weights, prev_phi = calculate_weights(args, pipe, discriminator,
                                vpsde, detr, detector_normalize, xt, sigma_next, prompt_embeds, coco_indices, step_object_occur_prob, prev_phi)
                            xt, prev_phi = resample(sample_weights, xt, args, prev_phi)

                        # Increase noise temporarily.
                        xt = xt + torch.randn_like(xt) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * args.S_noise

                        for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1
                            xt_next = []
                            
                            # Increase noise temporarily.
                            gamma_back = gamma if t_cur > t_min and t_cur < t_max else 0.0
                            if gamma_back > 0:
                                t_hat = t_cur + gamma_back * t_cur
                                xt_hat = xt + (t_hat ** 2 - t_cur ** 2).sqrt() * args.S_noise * torch.randn_like(xt)
                                t_cur = t_hat
                            else:
                                xt_hat = xt

                            for sample_ind in range(0, num_samples, args.batch_size):
                                tmp_xt_next, _ = pipe.scheduler.step(
                                    xt_hat[sample_ind: sample_ind + args.batch_size],
                                    t_cur,
                                    t_next,
                                    prompt_embeds,
                                    second_order=True
                                )
                                xt_next.append(tmp_xt_next)
                            
                            xt = torch.vstack(xt_next)
            
            # save image
            for sample_ind in range(num_samples):
                final_img = pipe.decode_latents(xt[[sample_ind]])
                final_img = pipe.numpy_to_pil(final_img)[0]
                final_img.save(f'{args.figure_dir}/{text_ind}/{sample_ind}_final.png')


if __name__ == "__main__":
    main()
