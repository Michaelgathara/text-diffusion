import torch
import torch.nn.functional as F
import os
import sys
import logging
import argparse
import numpy as np 
import gc
import bitsandbytes as bnb

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project Root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)

from models import ModelConfig, DiffusionTransformerModel, DiffusionProcess
from transformers import AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def log_memory_usage(device, step_name=""):
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        logger.info(f"GPU Memory {step_name}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")

def get_dataloaders(config, tokenizer, args):
    logger.info("loading fineweb-edu dataset...")
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_subset = args.dataset_subset

    try:
        full_stream_raw = load_dataset(dataset_name, name=dataset_subset, streaming=True)['train']
        
        val_stream_raw = full_stream_raw.take(args.num_validation_samples)
        train_stream_raw = full_stream_raw.skip(args.num_validation_samples)
        
        logger.info(f"loaded '{dataset_name}' subset '{dataset_subset}'.")
        logger.info(f"using {args.num_validation_samples} samples for validation.")

    except Exception as e:
        logger.error(f"failed to load dataset: {e}")
        sys.exit(1)

    def tokenize_function(examples):
        tokenized_batch = tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=config.block_size,
            return_attention_mask=False,
        )
        return {"input_ids": tokenized_batch['input_ids']}

    logger.info(f"tokenizing and processing dataset (block_size: {config.block_size})...")
    
    train_dataset = train_stream_raw.map(
        tokenize_function,
        batched=True,
        batch_size=args.map_batch_size 
    ).shuffle(
        buffer_size=args.shuffle_buffer_size,
        seed=config.seed
    ).with_format("torch")

    val_dataset = val_stream_raw.map(
        tokenize_function,
        batched=True,
        batch_size=args.map_batch_size
    ).with_format("torch")
    
    return train_dataset, val_dataset

@torch.no_grad() 
def evaluate_model(model, val_dataset_iterator, config, diffusion_helper, device, args):
    model.eval() 
    total_val_loss = 0.0
    actual_eval_iters = 0

    logger.info(f"starting evaluation for {config.eval_iters} iterations...")
    
    eval_batch_size = max(1, config.batch_size // 2)
    
    for i in range(config.eval_iters):
        try:
            val_batch_input_ids_list = []
            for _ in range(eval_batch_size):
                example = next(val_dataset_iterator)
                val_batch_input_ids_list.append(example['input_ids'])
            
            val_batch_token_ids = torch.stack(val_batch_input_ids_list).to(device, non_blocking=True)

            x_start_embeddings = model.token_embedding(val_batch_token_ids)
            current_micro_batch_size = val_batch_token_ids.shape[0]
            t = torch.randint(0, config.diffusion_timesteps, (current_micro_batch_size,), device=device).long()
            noise_eps = torch.randn_like(x_start_embeddings)
            x_t_noised_embeddings = diffusion_helper.q_sample(
                x_start=x_start_embeddings, t=t, noise=noise_eps
            )

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=(device.type == 'cuda' and args.use_amp)):
                predicted_noise, loss = model(
                    noised_token_ids_or_embeddings=x_t_noised_embeddings,
                    timesteps=t,
                    targets_noise=noise_eps,
                    input_is_embeddings=True
                )
            
            if loss is not None:
                total_val_loss += loss.item()
                actual_eval_iters += 1
            
            del val_batch_token_ids, x_start_embeddings, noise_eps, x_t_noised_embeddings, predicted_noise
            
            if i % 10 == 0:
                clear_memory()

        except StopIteration:
            logger.warning(f"validation data iterator exhausted after {i} eval iterations.")
            break
    
    clear_memory()
    model.train()
    
    if actual_eval_iters == 0:
        logger.warning("no validation batches were processed. returning inf loss.")
        return float('inf')
    return total_val_loss / actual_eval_iters

def setup_optimizer(model, config, args):
    if args.use_8bit_optimizer:
        try:
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
                block_wise=True,        # Better stability
                percentile_clipping=100, # Conservative clipping
                min_8bit_size=4096,     # Keep small tensors in 32-bit
            )
            logger.info("Using 8-bit AdamW optimizer")
            return optimizer
        except:
            logger.warning("Optimizer failed to init")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    logger.info("Using standard 32-bit AdamW optimizer")
    return optimizer

def main(args):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    config = ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"using device: {device}")

    clear_memory()

    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config.vocab_size = tokenizer.vocab_size
    
    if args.block_size is not None:
        config.block_size = args.block_size
    logger.info(f"using block_size (sequence length): {config.block_size}")

    model = DiffusionTransformerModel(config).to(device)
    log_memory_usage(device, "after model loading")
    
    diffusion_helper = DiffusionProcess(config, device=device)

    # Setup optimizer with 8-bit support
    optimizer = setup_optimizer(model, config, args)
    log_memory_usage(device, "after optimizer setup")

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and args.use_amp))

    train_iterable_dataset, val_iterable_dataset = get_dataloaders(config, tokenizer, args)
    train_dataset_iterator = iter(train_iterable_dataset)
    
    logger.info("starting training loop...")
    model.train()
    start_iter = 0
    best_val_loss = float('inf')

    # Checkpoint loading logic
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            logger.info(f"resuming from checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and args.use_amp:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_iter = checkpoint.get('iter_num', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"resumed from iteration {start_iter}. best_val_loss: {best_val_loss:.4f}")
        else:
            logger.warning(f"checkpoint specified but not found: {args.resume_checkpoint}. starting from scratch.")

    for iter_num in range(start_iter, config.max_iters):
        optimizer.zero_grad(set_to_none=True)
        
        batch_input_ids_list = []
        for _ in range(config.batch_size * config.accumulation_steps):
            try:
                example = next(train_dataset_iterator)
                batch_input_ids_list.append(example['input_ids'])
            except StopIteration:
                logger.info("training data iterator exhausted. re-initializing.")
                train_dataset_iterator = iter(train_iterable_dataset)
                example = next(train_dataset_iterator)
                batch_input_ids_list.append(example['input_ids'])
        
        effective_batch_input_ids = torch.stack(batch_input_ids_list).to(device, non_blocking=True)

        current_iter_loss_sum = 0.0
        for micro_step in range(config.accumulation_steps):
            start_idx = micro_step * config.batch_size
            end_idx = (micro_step + 1) * config.batch_size
            batch_token_ids = effective_batch_input_ids[start_idx:end_idx]

            x_start_embeddings = model.token_embedding(batch_token_ids)
            current_micro_batch_size = batch_token_ids.shape[0]
            t = torch.randint(0, config.diffusion_timesteps, (current_micro_batch_size,), device=device).long()
            noise_eps = torch.randn_like(x_start_embeddings)
            x_t_noised_embeddings = diffusion_helper.q_sample(
                x_start=x_start_embeddings, t=t, noise=noise_eps
            )

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=(device.type == 'cuda' and args.use_amp)):
                predicted_noise, loss = model(
                    noised_token_ids_or_embeddings=x_t_noised_embeddings,
                    timesteps=t,
                    targets_noise=noise_eps,
                    input_is_embeddings=True
                )
            
            if loss is None: 
                continue
            
            current_iter_loss_sum += loss.item()
            loss = loss / config.accumulation_steps
            scaler.scale(loss).backward()
            
            # Clear intermediate tensors to save memory
            del batch_token_ids, x_start_embeddings, noise_eps, x_t_noised_embeddings, predicted_noise
        
        # Clear the large batch tensor
        del effective_batch_input_ids
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if (iter_num + 1) % args.log_interval == 0:
            avg_loss_for_iter = current_iter_loss_sum / config.accumulation_steps
            if device.type == 'cuda':
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                logger.info(f"iter {iter_num+1}/{config.max_iters} | loss: {avg_loss_for_iter:.4f} | GPU mem: {gpu_mem:.2f}GB")
            else:
                logger.info(f"iter {iter_num+1}/{config.max_iters} | loss: {avg_loss_for_iter:.4f}")

        if (iter_num + 1) % config.eval_interval == 0:
            clear_memory()
            val_dataset_iterator = iter(val_iterable_dataset)
            val_loss = evaluate_model(model, val_dataset_iterator, config, diffusion_helper, device, args)
            logger.info(f"iter {iter_num+1} | val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"new best val_loss: {best_val_loss:.4f}. saving best model...")
                best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                torch.save({
                    'iter_num': iter_num + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config': vars(config),
                    'best_val_loss': best_val_loss,
                }, best_model_path)
                logger.info(f"best model saved to {best_model_path}")

        # Periodic memory cleanup
        if (iter_num + 1) % 100 == 0:
            clear_memory()

        if (iter_num + 1) % args.save_interval == 0:
            logger.info(f"saving checkpoint at iteration {iter_num+1}...")
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_iter_{iter_num+1}.pt")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save({
                'iter_num': iter_num + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': vars(config),
                'best_val_loss': best_val_loss, 
            }, checkpoint_path)
            logger.info(f"checkpoint saved to {checkpoint_path}")

    logger.info("training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a text diffusion model with fineweb-edu.")
    parser.add_argument("--dataset_subset", type=str, default="sample-10BT", help="fineweb-edu subset (e.g., 'sample-10BT', 'default')")
    parser.add_argument("--num_validation_samples", type=int, default=1000, help="number of samples from the beginning of the stream to use for validation.")
    parser.add_argument("--map_batch_size", type=int, default=1000, help="batch size for the .map() tokenization function.") 
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="name or path of the hugging face tokenizer.")
    parser.add_argument("--block_size", type=int, default=None, help="sequence length. overrides modelconfig if set.")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000, help="buffer size for shuffling dataset.")
    
    parser.add_argument("--cpu", action="store_true", help="force use cpu.")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision.")
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="use 8-bit AdamW optimizer to save memory.")
    parser.add_argument("--log_interval", type=int, default=100, help="logging interval.")
    parser.add_argument("--save_interval", type=int, default=5000, help="periodic checkpoint saving interval.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="path to checkpoint to resume from.")
    
    args = parser.parse_args()
    main(args)