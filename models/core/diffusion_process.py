import torch
import torch.nn.functional as F
import math

def get_noise_schedule(schedule_type, num_timesteps, beta_start=0.0001, beta_end=0.02, s=0.008, device='cpu'):
    if schedule_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64, device=device)
    elif schedule_type == 'cosine':
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64, device=device)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999) # this will help prevent a ton of numerical issues
    elif schedule_type == 'sqrt_linear':
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float64, device=device) ** 2
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule_type}")
    return betas.float() 

def extract_tensor_values(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.long()) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class DiffusionProcess:
    def __init__(self, config, device='cpu'): 
        self.num_timesteps = config.diffusion_timesteps
        self.device = device

        self.betas = get_noise_schedule(
            schedule_type=config.noise_schedule_type,
            num_timesteps=self.num_timesteps,
            beta_start=getattr(config, 'beta_start', 0.0001), 
            beta_end=getattr(config, 'beta_end', 0.02),
            s=getattr(config, 'cosine_schedule_s', 0.008), 
            device=self.device
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.maximum(self.posterior_variance, torch.tensor(1e-20, device=self.device))
        )
        
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract_tensor_values(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_tensor_values(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        noised_sample = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noised_sample
