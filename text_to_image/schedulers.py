import torch


class SDEDMScheduler:
    def __init__(self, model, beta_min=0.00085, beta_max=0.012, device='cuda'):
        # Initialize parameters
        self.M = 1000
        self.model = model
        self.beta_min = beta_min * self.M
        self.beta_max = beta_max * self.M
        self.beta_d = self.beta_max**0.5 - self.beta_min**0.5
        self.init_noise_sigma = torch.tensor(self.precond_sigma(1)).to(device)
        self.sigma_min = self.precond_sigma(1e-3).to(device)
    
    def run_sd_wrapper(self, sigma, cur_x, prompt_embeds, guidance_scale):
        """Run stable diffusion under EDM framework"""
        # compute the coefficients
        sigma = sigma.reshape(-1)
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.precond_sigma_inv(sigma)
        
        repeat_num = 2 if guidance_scale > 0 else 1
        num_images = cur_x.shape[0]
        c_out = torch.cat([c_out] * num_images * repeat_num)[:, None, None, None]
        c_in = torch.cat([c_in] * num_images * repeat_num)[:, None, None, None]
        c_noise = torch.cat([c_noise] * num_images * repeat_num)
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([cur_x] * repeat_num)
        
        # make sure prompt_embeds are in correct shape
        if prompt_embeds.shape[0] > latent_model_input.shape[0]:
            prompt_embeds = torch.cat([prompt_embeds[:num_images],
                prompt_embeds[prompt_embeds.shape[0]//2: prompt_embeds.shape[0]//2 + num_images]], dim=0)
        else:
            prompt_embeds = prompt_embeds

        # predict the noise residual
        F_x = self.model(c_in * latent_model_input, c_noise, encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None).sample
        D_x = latent_model_input + c_out * F_x

        if guidance_scale > 0:
            # perform guidance
            noise_pred_uncond, noise_pred_text = D_x.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = D_x
        
        return noise_pred

    def step(
        self,
        sample: torch.FloatTensor,
        sigma_cur: float,
        sigma_next: float,
        prompt_embeds: torch.FloatTensor,
        guidance_scale: float = 7.5,
        second_order: bool = True,
        guidance_score: torch.FloatTensor = None
    ):
        """Predict the sample at sigma_next by reversing the SDE."""
        # Euler step.
        denoised = self.run_sd_wrapper(sigma_cur, sample, prompt_embeds, guidance_scale=guidance_scale)
        d_cur = (sample - denoised) / sigma_cur
        
        # Apply guidance
        if guidance_score is not None:
            d_cur += guidance_score
        x_next = sample + (sigma_next - sigma_cur) * d_cur

        # Apply 2nd order correction.
        if second_order and sigma_next >= self.sigma_min:
            denoised = self.run_sd_wrapper(sigma_next, x_next, prompt_embeds, guidance_scale=guidance_scale)
            d_prime = (x_next - denoised) / sigma_next
            x_next = sample + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next, denoised

    def pred_x0(
        self, sample: torch.FloatTensor, sigma: float, prompt_embeds: torch.FloatTensor,
            guidance_scale: float = 7.5,
    ):
        """
        Predict original x0
        """
        denoised = self.run_sd_wrapper(sigma, sample, prompt_embeds, guidance_scale=guidance_scale)

        return denoised, None
    
    def precond_sigma(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return ((self.beta_d**2 / 3 * t**3 + self.beta_min**0.5 * self.beta_d * t**2 + self.beta_min * t).exp() - 1).sqrt()
    
    def precond_sigma_inv(self, sigma):
        delta_1 = self.beta_min ** (3/2) + 3 * self.beta_d * (1 + sigma ** 2).log()
        return (-self.beta_min**0.5 + delta_1 ** (1/3)) / self.beta_d
