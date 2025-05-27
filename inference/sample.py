import torch
import torch.nn.functional as F
import os
import sys
import argparse
import logging
from tqdm import tqdm 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models import ModelConfig, DiffusionTransformerModel, DiffusionProcess, extract_tensor_values
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, device):
    logger.info(f"loading checkpoint from: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        logger.error(f"checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        logger.error("config not found in checkpoint. please ensure config is saved during training.")
        sys.exit(1)
    
    config_dict = checkpoint['config']
    config = ModelConfig() 
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"config key '{key}' from checkpoint not found in current modelconfig definition.")

    tokenizer_name = getattr(config, 'tokenizer_name', 'gpt2') 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config.vocab_size = tokenizer.vocab_size

    model = DiffusionTransformerModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion_helper = DiffusionProcess(config, device=device)
    
    logger.info("model, tokenizer, and diffusion helper loaded successfully.")
    return model, tokenizer, config, diffusion_helper

@torch.no_grad()
def p_sample(model, x_t, t_tensor, diffusion_helper: DiffusionProcess):
    """Original DDPM sampling step"""
    predicted_noise, _ = model(
        noised_token_ids_or_embeddings=x_t,
        timesteps=t_tensor,
        input_is_embeddings=True
    )

    sqrt_recip_alphas_cumprod_t = extract_tensor_values(1.0 / diffusion_helper.sqrt_alphas_cumprod, t_tensor, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_tensor_values(diffusion_helper.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)
    
    x_0_hat = sqrt_recip_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise

    if t_tensor.min() == 0:
        return x_0_hat

    posterior_mean_coef1_t = extract_tensor_values(diffusion_helper.posterior_mean_coef1, t_tensor, x_t.shape)
    posterior_mean_coef2_t = extract_tensor_values(diffusion_helper.posterior_mean_coef2, t_tensor, x_t.shape)
    
    posterior_mean = posterior_mean_coef1_t * x_0_hat + posterior_mean_coef2_t * x_t

    posterior_log_variance_t = extract_tensor_values(diffusion_helper.posterior_log_variance_clipped, t_tensor, x_t.shape)
    noise_z = torch.randn_like(x_t) 
    
    x_t_minus_1 = posterior_mean + (0.5 * posterior_log_variance_t).exp() * noise_z
    return x_t_minus_1

@torch.no_grad()
def ddim_sample(model, x_t, t_tensor, t_prev_tensor, diffusion_helper, eta=0.0):
    """
    eta=0.0: deterministic (DDIM), eta=1.0: stochastic (DDPM)
    """
    predicted_noise, _ = model(
        noised_token_ids_or_embeddings=x_t,
        timesteps=t_tensor,
        input_is_embeddings=True
    )
    
    # Extract values for current and previous timesteps
    alpha_t = extract_tensor_values(diffusion_helper.alphas_cumprod, t_tensor, x_t.shape)
    
    # Handle the case where t_prev might be negative (final step)
    if t_prev_tensor.min() < 0:
        alpha_prev = torch.ones_like(alpha_t)
    else:
        alpha_prev = extract_tensor_values(diffusion_helper.alphas_cumprod, t_prev_tensor, x_t.shape)
    
    # Predict x_0 from current noisy sample
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
    
    # DDIM sampling formula
    sqrt_alpha_prev = torch.sqrt(alpha_prev)
    sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
    
    # Direction pointing towards x_t
    dir_xt = sqrt_one_minus_alpha_prev * predicted_noise
    
    # Add noise if eta > 0 (stochastic sampling)
    if eta > 0 and t_prev_tensor.min() >= 0:  # Don't add noise on final step
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        noise = torch.randn_like(x_t)
        x_prev = sqrt_alpha_prev * x_0_pred + dir_xt + sigma * noise
    else:
        # Deterministic sampling (pure DDIM)
        x_prev = sqrt_alpha_prev * x_0_pred + dir_xt
    
    return x_prev

@torch.no_grad()
def p_sample_loop(model, shape, diffusion_helper: DiffusionProcess, device, args):
    """Original DDPM sampling loop (full steps)"""
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)

    for t_val in tqdm(reversed(range(0, diffusion_helper.num_timesteps)), desc="DDPM sampling", total=diffusion_helper.num_timesteps):
        t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t_tensor, diffusion_helper)
    
    return x_t

@torch.no_grad()
def ddim_sample_loop(model, shape, diffusion_helper, device, num_steps=50, eta=0.0):
    """
    DDIM sampling loop with configurable number of steps
    """
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)
    
    # Create timestep schedule (skip steps for acceleration)
    total_timesteps = diffusion_helper.num_timesteps
    
    if num_steps >= total_timesteps:
        logger.warning(f"Requested {num_steps} steps >= total timesteps {total_timesteps}. Using full DDPM sampling.")
        return p_sample_loop(model, shape, diffusion_helper, device, args=None)
    
    # Create evenly spaced timesteps
    step_size = total_timesteps // num_steps
    timesteps = list(range(total_timesteps - 1, -1, -step_size))
    
    # Ensure we end at timestep 0
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    timesteps = torch.tensor(timesteps, device=device)
    
    logger.info(f"DDIM sampling with {len(timesteps)-1} steps, eta={eta}")
    
    for i in tqdm(range(len(timesteps) - 1), desc=f"DDIM sampling ({len(timesteps)-1} steps)"):
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        
        t_curr_tensor = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
        t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
        
        x_t = ddim_sample(model, x_t, t_curr_tensor, t_prev_tensor, diffusion_helper, eta)
    
    return x_t

def generate_text_from_embeddings(generated_embeddings, model, tokenizer):
    if not hasattr(model, 'token_embedding'):
        logger.error("model does not have 'token_embedding' attribute for converting embeddings to ids.")
        return ["error: token_embedding layer not found in model."] * generated_embeddings.shape[0]

    embedding_matrix = model.token_embedding.weight.data.detach()
    logits_for_ids = torch.matmul(generated_embeddings, embedding_matrix.t())
    generated_ids = torch.argmax(logits_for_ids, dim=-1)

    decoded_texts = []
    for i in range(generated_ids.shape[0]):
        text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        decoded_texts.append(text)
    return decoded_texts

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"using device: {device}")

    model, tokenizer, config, diffusion_helper = load_checkpoint(args.checkpoint_path, device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)

    output_shape = (args.num_samples, config.block_size, config.n_embd)

    logger.info(f"starting generation of {args.num_samples} samples...")
    logger.info(f"sequence length (block_size): {config.block_size}, embedding dim: {config.n_embd}")
    logger.info(f"total diffusion timesteps in model config: {config.diffusion_timesteps}")
    
    if args.sampling_steps is not None and args.sampling_steps < config.diffusion_timesteps:
        logger.info(f"using DDIM sampling with {args.sampling_steps} steps (eta={args.eta})")
        generated_embeddings = ddim_sample_loop(
            model, output_shape, diffusion_helper, device, 
            num_steps=args.sampling_steps, eta=args.eta
        )
    else:
        if args.sampling_steps is not None:
            logger.info(f"requested {args.sampling_steps} steps >= {config.diffusion_timesteps}, using full DDPM sampling")
        else:
            logger.info("using full DDPM sampling (all timesteps)")
        generated_embeddings = p_sample_loop(model, output_shape, diffusion_helper, device, args)
    
    logger.info("converting generated embeddings to text...")
    decoded_texts = generate_text_from_embeddings(generated_embeddings, model, tokenizer)

    for i, text in enumerate(decoded_texts):
        logger.info(f"\n--- generated sample {i+1} ---")
        print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate text using a trained diffusion model.")
    parser.add_argument("checkpoint_path", type=str, help="path to the model checkpoint (.pt file).")
    parser.add_argument("--num_samples", type=int, default=1, help="number of text samples to generate.")
    parser.add_argument("--sampling_steps", type=int, default=None, help="number of DDIM sampling steps (default: use all timesteps with DDPM).")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter: 0.0=deterministic, 1.0=stochastic like DDPM.")
    parser.add_argument("--cpu", action="store_true", help="force use cpu.")
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling.")
    
    args = parser.parse_args()
    main(args)