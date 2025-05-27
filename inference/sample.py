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
        # as a fallback, try to load default config, but this might lead to mismatches
        # config = ModelConfig()
        # logger.warning("using default modelconfig due to missing config in checkpoint.")
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
    config.vocab_size = tokenizer.vocab_size # ensure this matches during model init

    model = DiffusionTransformerModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # set to evaluation mode

    diffusion_helper = DiffusionProcess(config, device=device)
    
    logger.info("model, tokenizer, and diffusion helper loaded successfully.")
    return model, tokenizer, config, diffusion_helper

@torch.no_grad()
def p_sample(model, x_t, t_tensor, diffusion_helper: DiffusionProcess):
    predicted_noise, _ = model(
        noised_token_ids_or_embeddings=x_t,
        timesteps=t_tensor,
        input_is_embeddings=True
    )

    sqrt_recip_alphas_cumprod_t = extract_tensor_values(1.0 / diffusion_helper.sqrt_alphas_cumprod, t_tensor, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_tensor_values(diffusion_helper.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)
    
    x_0_hat = sqrt_recip_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise

    if t_tensor.min() == 0: # if current timestep is 0, we are done
        return x_0_hat # return the predicted clean sample

    posterior_mean_coef1_t = extract_tensor_values(diffusion_helper.posterior_mean_coef1, t_tensor, x_t.shape)
    posterior_mean_coef2_t = extract_tensor_values(diffusion_helper.posterior_mean_coef2, t_tensor, x_t.shape)
    
    posterior_mean = posterior_mean_coef1_t * x_0_hat + posterior_mean_coef2_t * x_t

    posterior_log_variance_t = extract_tensor_values(diffusion_helper.posterior_log_variance_clipped, t_tensor, x_t.shape)
    noise_z = torch.randn_like(x_t) 
    
    x_t_minus_1 = posterior_mean + (0.5 * posterior_log_variance_t).exp() * noise_z
    return x_t_minus_1

@torch.no_grad()
def p_sample_loop(model, shape, diffusion_helper: DiffusionProcess, device, args):
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)

    for t_val in tqdm(reversed(range(0, diffusion_helper.num_timesteps)), desc="sampling loop", total=diffusion_helper.num_timesteps):
        t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t_tensor, diffusion_helper)
    
    return x_t

def generate_text_from_embeddings(generated_embeddings, model, tokenizer):
    if not hasattr(model, 'token_embedding'):
        logger.error("model does not have 'token_embedding' attribute for converting embeddings to ids.")
        return ["error: token_embedding layer not found in model."] * generated_embeddings.shape[0]

    embedding_matrix = model.token_embedding.weight.data.detach() # (vocab_size, embed_dim)
    
    # calculate logits by taking dot product: (batch, seq_len, embed_dim) @ (embed_dim, vocab_size)
    # result: (batch, seq_len, vocab_size)
    # using einsum for clarity or matmul:
    # logits_for_ids = torch.einsum('bse,ve->bsv', generated_embeddings, embedding_matrix)
    logits_for_ids = torch.matmul(generated_embeddings, embedding_matrix.t())
    
    generated_ids = torch.argmax(logits_for_ids, dim=-1) # (batch, seq_len)

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
    # if args.sampling_steps is not None and args.sampling_steps < config.diffusion_timesteps:
    #     logger.warning(f"note: full ddpm sampling uses all configured timesteps ({config.diffusion_timesteps}).")
    #     logger.warning("ddim or other accelerated samplers are needed for fewer steps (not implemented here).")

    generated_embeddings = p_sample_loop(model, output_shape, diffusion_helper, device, args)
    
    logger.info("converting generated embeddings to text...")
    decoded_texts = generate_text_from_embeddings(generated_embeddings, model, tokenizer)

    for i, text in enumerate(decoded_texts):
        logger.info(f"\n--- generated sample {i+1} ---")
        print(text) # print to stdout for easy viewing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate text using a trained diffusion model.")
    parser.add_argument("checkpoint_path", type=str, help="path to the model checkpoint (.pt file).")
    parser.add_argument("--num_samples", type=int, default=1, help="number of text samples to generate.")
    # parser.add_argument("--sampling_steps", type=int, default=50, help="number of ddim steps (not fully implemented for ddim here, uses ddpm full steps).")
    parser.add_argument("--cpu", action="store_true", help="force use cpu.")
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling.")
    
    args = parser.parse_args()
    # the sampling_steps argument is a bit misleading for basic ddpm as it uses all steps.
    # for ddim or other schedulers, this would be more relevant.
    # for now, it's a placeholder if you extend to ddim.
    main(args)