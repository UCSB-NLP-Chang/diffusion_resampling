import math

import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

from discriminator_lib import get_likelihood_ratio


def calculate_object_prob(model, image, detector_normalize, coco_indices=None):
    # round to 0-255 integer
    image = image * 255
    image = image.round()
    # preprocess
    image = transforms.functional.resize(image, 800, antialias=True)
    image = image / 255
    image = detector_normalize(image)
    
    outputs = model(image)
    log_probs = nn.functional.log_softmax(outputs['pred_logits'], -1) # B, N, C
    all_logprobs = log_probs.max(dim=1)[0] # B, C
    exist_logprob = torch.zeros_like(all_logprobs) # for object-only discriminator
    obj_logprobs = all_logprobs[:, coco_indices] # B, N
    exist_logprob[:, coco_indices] = all_logprobs[:, coco_indices]
    
    return obj_logprobs, exist_logprob


def calculate_caption_prob_ratio(log_object_probs, args, object_occur_prob, obj_threshold=0.5):
    log_object_probs = torch.vstack(log_object_probs) # K, n_obj
    if object_occur_prob is None:
        undetected = log_object_probs < math.log(obj_threshold)
        hatx0_c1 = undetected.sum(dim=0) / args.num_generation
    else:
        hatx0_c1 = 1 - object_occur_prob
    q_c1_hatx0 = hatx0_c1 * (1 - args.c0_prior) / (args.c0_prior + hatx0_c1 * (1 - args.c0_prior))
    q_c1_hatx0 = q_c1_hatx0[None].expand_as(log_object_probs)
    # calculate expectation
    object_probs = log_object_probs.exp()
    q_c1_x = (1 - object_probs) * q_c1_hatx0 + object_probs
    log_object_prob_ratio = log_object_probs - q_c1_x.log()
    log_caption_prob_ratio = log_object_prob_ratio.sum(dim=1)
    
    return log_caption_prob_ratio


@torch.no_grad()
def calculate_weights(args, pipe, discriminator, vpsde, detector, normalizer,
        xt, sigma_cur, prompt_embeds, coco_indices, object_occur_prob, prev_phi):
    """Calculate resample weights"""
    log_object_probs, ll_ratios = [], []
    for i in range(0, args.num_generation, args.batch_size):
        if args.method == 'pf-hybrid' or args.use_obj_discriminator:
            # Calculate object probability
            if sigma_cur > 0:
                pred_x0, _ = pipe.scheduler.pred_x0(xt[i: i+args.batch_size], sigma_cur, prompt_embeds)
            else:
                pred_x0 = xt[i: i+args.batch_size]
            latents = 1 / pipe.vae.config.scaling_factor * pred_x0
            image = pipe.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            log_obj_prob, pad_obj_prob = calculate_object_prob(detector, image, normalizer, coco_indices)
        else:
            log_obj_prob = torch.zeros(min(args.num_generation, i+args.batch_size) - i, coco_indices.shape[0], device=xt.device)
        log_object_probs.append(log_obj_prob)
        
        # Discriminator
        if args.use_obj_discriminator:
            inputs = torch.cat([pad_obj_prob[:, :-1], sigma_cur.repeat(pad_obj_prob.shape[0]).reshape(-1, 1)], dim=1)
            logits = discriminator(inputs).view(-1)
            logits = F.sigmoid(logits)
            prediction = torch.clip(logits, 1e-6, 1. - 1e-6)
            l_ratio = prediction / (1. - prediction)
        else:
            l_ratio, _ = get_likelihood_ratio(discriminator, vpsde, xt[i: i+args.batch_size], sigma_cur, None, None)
        
        ll_ratios.append(l_ratio)
        
    # Calculate object mention ratio
    if args.method == 'pf-hybrid':
        log_caption_prob_ratio = calculate_caption_prob_ratio(log_object_probs, args, object_occur_prob)
    else:
        log_caption_prob_ratio = 0
    # Calculate ll_ratio
    ll_ratios = torch.cat(ll_ratios).log().squeeze()
    
    # Calculate weight
    sample_weights = log_caption_prob_ratio + ll_ratios
    cur_phi = sample_weights.clone()
    sample_weights -= prev_phi
    return sample_weights, cur_phi


def resample(sample_weights, xt, args, prev_phi):
    # rescale largest weight to 0
    sample_weights = sample_weights - sample_weights.max()
    sample_weights = sample_weights.exp()
    
    # Resample
    sample_weights = sample_weights / sample_weights.sum()
    inds = torch.multinomial(sample_weights, num_samples=args.num_generation, replacement=True)
    # Update
    xt = xt[inds]
    prev_phi = prev_phi[inds]
    
    return xt, prev_phi
