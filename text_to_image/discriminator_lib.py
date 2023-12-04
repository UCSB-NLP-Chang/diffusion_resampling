import torch
import numpy as np


def get_likelihood_ratio(discriminator, vpsde, input, sigma, class_labels, img_resolution):
    mean_vp_tau, tau = vpsde.transform_unnormalized_wve_to_normalized_vp(sigma) ## VP pretrained classifier
    input = mean_vp_tau.view(-1)[:,None,None,None] * input
    tau = 999 * tau
    tau = torch.ones(input.shape[0], device=tau.device) * tau
    logits = discriminator(input.float(), tau, class_labels)
    prediction = torch.clip(logits, 1e-6, 1. - 1e-6)
    ll_ratio = prediction / (1. - prediction)
    return ll_ratio, tau[0].item()


class vpsde():
    def __init__(self, scaled_linear=False):
        self.scaled_linear = scaled_linear
        if scaled_linear:
            self.beta_0 = 0.85
            self.beta_1 = 12.
            self.beta_d = self.beta_1**0.5 - self.beta_0**0.5
        else:
            self.beta_0 = 0.1
            self.beta_1 = 20.
            self.s = 0.008
            self.f_0 = np.cos(self.s / (1. + self.s) * np.pi / 2.) ** 2

    @property
    def T(self):
        return 1

    def compute_tau(self, std_wve_t):
        if self.scaled_linear:
            delta_1 = self.beta_0 ** (3/2) + 3 * self.beta_d * (1 + std_wve_t ** 2).log()
            tau = (-self.beta_0**0.5 + delta_1 ** (1/3)) / self.beta_d
        else:
            tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
            tau /= self.beta_1 - self.beta_0
        return tau

    def marginal_prob(self, t):
        if self.scaled_linear:
            log_mean_coeff = -0.5 * (1/3 * self.beta_d**2 * t**3 + self.beta_d * self.beta_0**0.5 * t**2 + self.beta_0 * t)
        else:
            log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def transform_unnormalized_wve_to_normalized_vp(self, t, std_out=False):
        tau = self.compute_tau(t)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau)
        if std_out:
            return mean_vp_tau, std_vp_tau, tau
        return mean_vp_tau, tau

    def compute_t_cos_from_t_lin(self, t_lin):
        sqrt_alpha_t_bar = torch.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = torch.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos

    def get_diffusion_time(self, batch_size, batch_device, t_min=1e-5, importance_sampling=True):
        if importance_sampling and not self.scaled_linear:
            Z = self.normalizing_constant(t_min)
            u = torch.rand(batch_size, device=batch_device)
            return (-self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) *
                    torch.log(1. + torch.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
        else:
            return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

    def antiderivative(self, t, stabilizing_constant=0.):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

    def normalizing_constant(self, t_min):
        return self.antiderivative(self.T) - self.antiderivative(t_min)

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0
