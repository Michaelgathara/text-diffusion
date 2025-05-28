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
    """eta=0.0: deterministic (DDIM), eta=1.0: stochastic (DDPM)"""
    predicted_noise, _ = model(
        noised_token_ids_or_embeddings=x_t,
        timesteps=t_tensor,
        input_is_embeddings=True
    )
    
    alpha_t = extract_tensor_values(diffusion_helper.alphas_cumprod, t_tensor, x_t.shape)
    
    if t_prev_tensor.min() < 0:
        alpha_prev = torch.ones_like(alpha_t)
    else:
        alpha_prev = extract_tensor_values(diffusion_helper.alphas_cumprod, t_prev_tensor, x_t.shape)
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
    
    sqrt_alpha_prev = torch.sqrt(alpha_prev)
    sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
    
    dir_xt = sqrt_one_minus_alpha_prev * predicted_noise
    
    if eta > 0 and t_prev_tensor.min() >= 0:
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        noise = torch.randn_like(x_t)
        x_prev = sqrt_alpha_prev * x_0_pred + dir_xt + sigma * noise
    else:
        x_prev = sqrt_alpha_prev * x_0_pred + dir_xt
    
    return x_prev

@torch.no_grad()
def ddim_sample_with_prompt(model, x_t, t_tensor, t_prev_tensor, diffusion_helper, prompt_mask, prompt_embeddings, eta=0.0):
    """DDIM sampling step with prompt conditioning"""
    predicted_noise, _ = model(
        noised_token_ids_or_embeddings=x_t,
        timesteps=t_tensor,
        input_is_embeddings=True
    )
    
    alpha_t = extract_tensor_values(diffusion_helper.alphas_cumprod, t_tensor, x_t.shape)
    
    if t_prev_tensor.min() < 0:
        alpha_prev = torch.ones_like(alpha_t)
    else:
        alpha_prev = extract_tensor_values(diffusion_helper.alphas_cumprod, t_prev_tensor, x_t.shape)
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
    
    sqrt_alpha_prev = torch.sqrt(alpha_prev)
    sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
    
    dir_xt = sqrt_one_minus_alpha_prev * predicted_noise
    
    if eta > 0 and t_prev_tensor.min() >= 0:
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        noise = torch.randn_like(x_t)
        x_prev = sqrt_alpha_prev * x_0_pred + dir_xt + sigma * noise
    else:
        x_prev = sqrt_alpha_prev * x_0_pred + dir_xt
    
    # Keep prompt positions fixed
    x_prev = torch.where(prompt_mask.unsqueeze(-1), prompt_embeddings, x_prev)
    
    return x_prev

@torch.no_grad()
def ddim_sample_loop_with_prompt(model, tokenizer, prompt_text, max_new_tokens, diffusion_helper, device, num_steps=50, eta=0.0):
    """DDIM sampling loop with prompt conditioning"""
    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    prompt_length = prompt_tokens.shape[1]
    
    if prompt_length >= model.config.block_size:
        raise ValueError(f"Prompt too long: {prompt_length} >= {model.config.block_size}")
    
    # Calculate total sequence length
    total_length = min(prompt_length + max_new_tokens, model.config.block_size)
    batch_size = 1
    
    # Get prompt embeddings (these stay fixed)
    prompt_embeddings = model.token_embedding(prompt_tokens)
    
    # Create mask for prompt vs generated tokens
    prompt_mask = torch.zeros(batch_size, total_length, dtype=torch.bool, device=device)
    prompt_mask[:, :prompt_length] = True
    
    # Initialize with noise for generated part, prompt embeddings for prompt part
    x_t = torch.randn(batch_size, total_length, model.config.n_embd, device=device)
    x_t[:, :prompt_length] = prompt_embeddings
    
    # Pad prompt embeddings to full sequence length for masking
    full_prompt_embeddings = torch.zeros(batch_size, total_length, model.config.n_embd, device=device)
    full_prompt_embeddings[:, :prompt_length] = prompt_embeddings
    
    total_timesteps = diffusion_helper.num_timesteps
    
    if num_steps >= total_timesteps:
        num_steps = total_timesteps
    
    # Create timestep schedule
    step_size = total_timesteps // num_steps
    timesteps = list(range(total_timesteps - 1, -1, -step_size))
    
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    timesteps = torch.tensor(timesteps, device=device)
    
    logger.info(f"DDIM sampling with prompt: '{prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}'")
    logger.info(f"Prompt length: {prompt_length}, generating {total_length - prompt_length} new tokens")
    
    for i in tqdm(range(len(timesteps) - 1), desc=f"DDIM sampling ({len(timesteps)-1} steps)"):
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        
        t_curr_tensor = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
        t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
        
        x_t = ddim_sample_with_prompt(
            model, x_t, t_curr_tensor, t_prev_tensor, diffusion_helper, 
            prompt_mask, full_prompt_embeddings, eta
        )
    
    return x_t

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
    """DDIM sampling loop with configurable number of steps"""
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)
    
    total_timesteps = diffusion_helper.num_timesteps
    
    if num_steps >= total_timesteps:
        logger.warning(f"Requested {num_steps} steps >= total timesteps {total_timesteps}. Using full DDPM sampling.")
        return p_sample_loop(model, shape, diffusion_helper, device, args=None)
    
    step_size = total_timesteps // num_steps
    timesteps = list(range(total_timesteps - 1, -1, -step_size))
    
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

# def generate_text_from_embeddings(generated_embeddings, model, tokenizer):
#     if not hasattr(model, 'token_embedding'):
#         logger.error("model does not have 'token_embedding' attribute for converting embeddings to ids.")
#         return ["error: token_embedding layer not found in model."] * generated_embeddings.shape[0]

#     embedding_matrix = model.token_embedding.weight.data.detach()
#     logits_for_ids = torch.matmul(generated_embeddings, embedding_matrix.t())
#     generated_ids = torch.argmax(logits_for_ids, dim=-1)

#     decoded_texts = []
#     for i in range(generated_ids.shape[0]):
#         text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
#         decoded_texts.append(text)
#     return decoded_texts

def generate_text_from_embeddings(generated_embeddings, model, tokenizer, temperature=0.7, top_k=40, top_p=0.9):
    if not hasattr(model, 'token_embedding'):
        logger.error("Model does not have 'token_embedding' attribute.")
        return ["error: token_embedding layer not found."] * generated_embeddings.shape[0]

    embedding_matrix = model.token_embedding.weight.data.detach()
    logits_for_ids = torch.matmul(generated_embeddings, embedding_matrix.t())
    
    decoded_texts = []
    for batch_idx in range(logits_for_ids.shape[0]):
        batch_tokens = []
        
        for pos_idx in range(logits_for_ids.shape[1]):
            logits = logits_for_ids[batch_idx, pos_idx]
            
            if temperature > 0:
                logits = logits / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                logits = filtered_logits
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            if temperature == 0:
                next_token = torch.argmax(logits).item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            batch_tokens.append(next_token)
        
        text = tokenizer.decode(batch_tokens, skip_special_tokens=True)
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

    if args.prompt:
        logger.info(f"generating with prompt: '{args.prompt}'")
        
        max_new_tokens = args.max_new_tokens or (config.block_size // 2)
        sampling_steps = args.sampling_steps or 50
        
        generated_embeddings = ddim_sample_loop_with_prompt(
            model, tokenizer, args.prompt, max_new_tokens, 
            diffusion_helper, device, sampling_steps, args.eta
        )
        
        # Convert to text
        embedding_matrix = model.token_embedding.weight.data.detach()
        logits_for_ids = torch.matmul(generated_embeddings, embedding_matrix.t())
        generated_ids = torch.argmax(logits_for_ids, dim=-1)
        
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"\n--- Generated Text ---")
        print(full_text)
        
    else:
        output_shape = (args.num_samples, config.block_size, config.n_embd)

        logger.info(f"starting unconditional generation of {args.num_samples} samples...")
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
    
    # Unconditional generation options
    parser.add_argument("--num_samples", type=int, default=1, help="number of text samples to generate (unconditional mode).")
    
    # Conditional generation options  
    parser.add_argument("--prompt", type=str, default=None, help="text prompt for conditional generation.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="max new tokens to generate after prompt (default: block_size//2).")
    
    # Sampling options
    parser.add_argument("--sampling_steps", type=int, default=None, help="number of DDIM sampling steps (default: 50 for prompt mode, all timesteps for unconditional).")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter: 0.0=deterministic, 1.0=stochastic like DDPM.")
    
    # Other options
    parser.add_argument("--cpu", action="store_true", help="force use cpu.")
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling.")
    
    args = parser.parse_args()
    main(args)