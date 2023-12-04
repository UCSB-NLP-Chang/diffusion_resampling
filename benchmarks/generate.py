# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import click
import tqdm
import pickle
import re
import json
import numpy as np
import torch
import io
from torchvision.utils import make_grid, save_image
import classifier_lib


def diffusion_sampler(
    method, sampler, net, discriminator, vpsde, latents, class_labels=None, randn_like=torch.randn_like,
    multinomial=torch.multinomial, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, S_churn=0,
    S_min=0, S_max=float('inf'), S_noise=0, restart_info="", restart_gamma=0, num_particles=1,
    dg_weight_1st_order=0.0, time_min=0.01, time_max=1.0, resample_inds=[]
):

    def get_steps(min_t, max_t, num_steps, rho):
        step_indices = torch.arange(num_steps, dtype=torch.float, device=latents.device)
        t_steps = (max_t ** (1 / rho) + step_indices / (num_steps - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
        return t_steps
    
    def resample(xt, t, class_labels, prev_ll_ratio, img_resolution):
        """Calculate weights and resample based on given weights"""
        # Sampling weights
        ll_ratio, tau = classifier_lib.get_likelihood_ratio(discriminator, vpsde,
            xt.float(), t, class_labels, img_resolution) # N
        ll_ratio = ll_ratio.reshape(-1, num_particles)
        weights = ll_ratio / prev_ll_ratio
        # Resample
        weights = weights / weights.sum(dim=1, keepdim=True)
        inds = multinomial(weights, num_samples=num_particles)
        xt = xt.reshape(-1, num_particles, *xt.shape[1:])
        xt = torch.gather(xt, 1, inds[:, :, None, None, None].expand(-1, -1, 3, xt.shape[-1], xt.shape[-1]))
        xt = xt.reshape(-1, 3, xt.shape[-1], xt.shape[-1])
        if class_labels is not None:
            class_labels = class_labels.reshape(-1, num_particles, *class_labels.shape[1:])
            class_labels = torch.gather(class_labels, 1, inds[..., None].expand(-1, -1, class_labels.shape[-1]))
            class_labels = class_labels.reshape(-1, class_labels.shape[-1])
        
        prev_ll_ratio = torch.gather(ll_ratio, 1, inds)
        return xt, class_labels, prev_ll_ratio

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Get num_steps for restart sampler
    if sampler == 'restart':
        num_steps, restart_info = restart_info.split('; ')
        num_steps = int(num_steps)

    # Time step discretization.
    t_steps = get_steps(sigma_min, sigma_max, num_steps, rho)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    
    x_next = latents.to(torch.float64) * t_steps[0]

    if sampler == 'restart':
        # total_steps \t {[num_steps, number of restart iteration (K), t_min, t_max], ... }
        restart_list = json.loads(restart_info) if restart_info != '' else {}
        # cast t_min to the index of nearest value in t_steps
        restart_list = {int(torch.argmin(abs(t_steps - v[2]), dim=0)): v for k, v in restart_list.items()}
    elif sampler == 'edm':
        resample_S_churn = (np.sqrt(2) - 1) * num_steps
    
    if method == 'pf':
        # Resample stats
        prev_ll_ratio = torch.ones((latents.shape[0] // num_particles, num_particles), device=latents.device)
    elif method == 'dg':
        # DG parameters
        period_weight = 2

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N_main -1
        x_cur = x_next
        
        if sampler == 'edm' and method == 'pf' and i in resample_inds and num_particles > 1:
            # ================== Resampling for EDM ==================
            x_cur, class_labels, prev_ll_ratio = resample(x_cur, t_cur, class_labels, prev_ll_ratio, net.img_resolution)
            S_churn_ = resample_S_churn
        else:
            S_churn_ = S_churn
        
        # Increase noise temporarily.
        gamma = min(S_churn_ / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        assert torch.isnan(x_hat).sum() == 0, f"NaN detected at step {i}."
        
        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        ## DG correction
        if method == 'dg':
            discriminator_guidance, log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_hat, t_hat.view(-1), net.img_resolution, time_min, time_max, class_labels, log=True)
            # boosting
            if i % period_weight == 0:
                discriminator_guidance[log_ratio < 0.] *= 2.
            d_cur += dg_weight_1st_order * (discriminator_guidance / t_hat)
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # ================= restart ================== #
        if sampler == 'restart' and i + 1 in restart_list.keys():
            restart_idx = i + 1

            for restart_iter in range(restart_list[restart_idx][1]):
                new_t_steps = get_steps(min_t=t_steps[restart_idx], max_t=restart_list[restart_idx][3],
                                        num_steps=restart_list[restart_idx][0], rho=rho)
                new_total_step = len(new_t_steps)
                
                # ================== Resampling ==================
                if method == 'pf':
                    x_next, class_labels, prev_ll_ratio = resample(x_next, new_t_steps[-1], class_labels, prev_ll_ratio, net.img_resolution)
                
                # Add noise
                x_next = x_next + randn_like(x_next) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * S_noise

                for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1

                    x_cur = x_next
                    gamma = restart_gamma if S_min <= t_cur <= S_max else 0
                    t_hat = net.round_sigma(t_cur + gamma * t_cur)

                    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
                    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
                    d_cur = (x_hat - denoised) / (t_hat)
                    ## DG correction
                    if method == 'dg':
                        discriminator_guidance, log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_hat, t_hat.view(-1), net.img_resolution, time_min, time_max, class_labels, log=True)
                        # boosting
                        if j % period_weight == 0:
                            discriminator_guidance[log_ratio < 0.] *= 2.
                        d_cur += dg_weight_1st_order * (discriminator_guidance / t_hat)
                    x_next = x_hat + (t_next - t_hat) * d_cur

                    # Apply 2nd order correction.
                    if j < new_total_step - 2 or new_t_steps[-1] != 0:
                        denoised = net(x_next, t_next, class_labels).to(torch.float64)
                        d_prime = (x_next - denoised) / t_next
                        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # Evaluate final ll ratio
    tau = torch.ones_like(x_next[:, 0, 0, 0]) * 1e-4
    ll_ratio, _ = classifier_lib.get_likelihood_ratio(discriminator, vpsde, x_next.float(), tau, class_labels, net.img_resolution) # N

    return x_next, ll_ratio.cpu().detach().numpy()


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
    def multinomial(self, weights, num_samples):
        return torch.multinomial(weights, num_samples=num_samples, replacement=True, generator=self.generators[0])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'batch_size',     help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-999', show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--sampler',                 help='Diffusion sampler',      metavar='restart|edm',                    type=click.Choice(['edm', 'restart']), default='restart', show_default=True)
@click.option('--method',                  help='Generation method',      metavar='pf|dg|none',                     type=click.Choice(['pf', 'dg', 'none']), default='pf', show_default=True)

## Sampling configureation
@click.option('--num_particles',               help='Number of particles',           metavar='INT',                       type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--device',                  help='Device', metavar='STR',                                            type=str, default='cuda:0')
@click.option('--resample_inds',           help='Indices for resampling',      metavar='LIST',                      type=parse_int_list, default='', show_default=True)

# Restart parameters
@click.option('--restart_info', 'restart_info',               help='restart information', metavar='STR',            type =str, default='', show_default=True)
@click.option('--restart_gamma', 'restart_gamma',             help='restart gamma', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.05, show_default=True)

## DG configuration
@click.option('--dg_weight_1st_order',     help='Weight of DG for 1st prediction',       metavar='FLOAT',           type=click.FloatRange(min=0), default=1., show_default=True)

## Discriminator checkpoint
@click.option('--pretrained_classifier_ckpt',help='Path of ADM classifier(latent extractor)',  metavar='STR',       type=str, default='checkpoints/ADM_classifier/64x64_classifier.pt', show_default=True)
@click.option('--discriminator_ckpt',      help='Path of discriminator',  metavar='STR',                            type=str, default='checkpoints/discriminator/imagenet_cond/discriminator_50.pt', show_default=True)

## Discriminator architecture
@click.option('--cond',                    help='Is it conditional discriminator?', metavar='INT',                  type=click.IntRange(min=0, max=1), default=1, show_default=True)

def main(sampler, method, dg_weight_1st_order, cond, pretrained_classifier_ckpt, discriminator_ckpt, batch_size, seeds, num_particles, network_pkl, outdir, class_idx, device, resample_inds, **sampler_kwargs):
    
    ## Check arguments
    if method == 'dg':
        assert dg_weight_1st_order > 0, "Discriminator guidance weight should be positive"
    
    if sampler == 'edm':
        resample_inds = [i for i in resample_inds if i >= 0]
    
    ## Set seed
    num_batches = len(seeds) // batch_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)

    ## Load pretrained score network.
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    ## Load discriminator
    dropout = 0.1 if 'ffhq' in discriminator_ckpt else 0.0
    discriminator = classifier_lib.get_discriminator(pretrained_classifier_ckpt, discriminator_ckpt,
        net.label_dim and cond, net.img_resolution, device, enable_grad=method=='dg', dropout=dropout)
    
    print(f"Loaded discriminator from {discriminator_ckpt}")
    vpsde = classifier_lib.vpsde()

    ## Loop over batches.
    print(f'Generating {len(seeds)} images to "{outdir}"...')
    os.makedirs(outdir, exist_ok=True)
    for i in tqdm.tqdm(range(num_batches)):
        ## Check if already done
        if os.path.exists(os.path.join(outdir, f"samples_{all_batches[i][0].item()}.npz")):
            continue
        ## Pick latents and labels.
        rnd = StackedRandomGenerator(device, all_batches[i])
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
            class_labels = class_labels[:batch_size // num_particles]
            class_labels = torch.repeat_interleave(class_labels, num_particles, 0)
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        ## Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        images, ll_ratios = diffusion_sampler(method, sampler, net, discriminator, vpsde, latents, class_labels, randn_like=rnd.randn_like, num_particles=num_particles, dg_weight_1st_order=dg_weight_1st_order, resample_inds=resample_inds, multinomial=rnd.multinomial, **sampler_kwargs)

        ## Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        with open(os.path.join(outdir, f"samples_{all_batches[i][0].item()}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            if class_labels is None:
                np.savez_compressed(io_buffer, samples=images_np, ll_ratio=ll_ratios)
            else:
                np.savez_compressed(io_buffer, samples=images_np, ll_ratio=ll_ratios, label=class_labels.cpu().numpy())
            fout.write(io_buffer.getvalue())

        nrow = int(np.sqrt(images_np.shape[0]))
        image_grid = make_grid(torch.tensor(images_np).permute(0, 3, 1, 2) / 255., nrow, padding=2)
        with open(os.path.join(outdir, f"sample_{all_batches[i][0].item()}.png"), "wb") as fout:
            save_image(image_grid, fout)


#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
