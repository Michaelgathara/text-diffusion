import torch
import torch.nn.functional as F
import os
import sys
import logging
import argparse
import numpy as np 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project Root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)

from models import ModelConfig, DiffusionTransformerModel, DiffusionProcess
from transformers import AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dataloaders(config, tokenizer, args): # renamed to get_dataloaders
    logger.info("loading fineweb-edu dataset...")
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_subset = args.dataset_subset

    try:
        # an alternative is to use a different split if available, or a different small dataset
        full_stream_raw = load_dataset(dataset_name, name=dataset_subset, streaming=True)['train']
        
        # important: .take() on a streaming dataset creates a new iterable dataset
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
        batch_size=args.map_batch_size # can use same map batch size
    ).with_format("torch")
    
    return train_dataset, val_dataset

@torch.no_grad() 
def evaluate_model(model, val_dataset_iterator, config, diffusion_helper, device, args):
    model.eval() 
    total_val_loss = 0.0
    actual_eval_iters = 0

    logger.info(f"starting evaluation for {config.eval_iters} iterations...")
    for i in range(config.eval_iters):
        try:
            # manual batching for validation
            val_batch_input_ids_list = []
            for _ in range(config.batch_size): # use training batch_size for eval batch consistency
                example = next(val_dataset_iterator)
                val_batch_input_ids_list.append(example['input_ids'])
            
            val_batch_token_ids = torch.stack(val_batch_input_ids_list).to(device)

            # same noising process as in training
            x_start_embeddings = model.token_embedding(val_batch_token_ids)
            current_micro_batch_size = val_batch_token_ids.shape[0] # can be less than config.batch_size if dataset ends
            t = torch.randint(0, config.diffusion_timesteps, (current_micro_batch_size,), device=device).long()
            noise_eps = torch.randn_like(x_start_embeddings)
            x_t_noised_embeddings = diffusion_helper.q_sample(
                x_start=x_start_embeddings, t=t, noise=noise_eps
            )

            # forward pass (no amp needed for eval usually, but can be used if desired)
            predicted_noise, loss = model(
                noised_token_ids_or_embeddings=x_t_noised_embeddings,
                timesteps=t,
                targets_noise=noise_eps,
                input_is_embeddings=True
            )
            if loss is not None:
                total_val_loss += loss.item()
                actual_eval_iters +=1
            else:
                logger.warning("evaluation loss is none for one batch.")

        except StopIteration:
            logger.warning(f"validation data iterator exhausted after {i} eval iterations.")
            break # stop if validation data runs out
    
    model.train() # set model back to training mode
    if actual_eval_iters == 0:
        logger.warning("no validation batches were processed. returning inf loss.")
        return float('inf')
    return total_val_loss / actual_eval_iters


def main(args):
    config = ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"using device: {device}")

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
    diffusion_helper = DiffusionProcess(config, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.use_amp))

    train_iterable_dataset, val_iterable_dataset = get_dataloaders(config, tokenizer, args)
    train_dataset_iterator = iter(train_iterable_dataset)
    # we'll create a new val iterator each time we evaluate to ensure it starts from the beginning
    
    logger.info("starting training loop...")
    model.train()
    start_iter = 0
    best_val_loss = float('inf')

    # checkpoint loading logic
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            logger.info(f"resuming from checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and args.use_amp: # only load scaler if amp is used
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_iter = checkpoint.get('iter_num', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            # re-initialize dataset iterator to roughly where it might have been (not perfect for streams)
            # for _ in range(start_iter * config.batch_size * config.accumulation_steps):
            # try: next(train_dataset_iterator) # this is tricky with streams, often just restart
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
        
        effective_batch_input_ids = torch.stack(batch_input_ids_list).to(device)

        current_iter_loss_sum = 0.0 # to sum losses from micro-batches for logging
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

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.use_amp)):
                predicted_noise, loss = model(
                    noised_token_ids_or_embeddings=x_t_noised_embeddings,
                    timesteps=t,
                    targets_noise=noise_eps,
                    input_is_embeddings=True
                )
            
            if loss is None: continue
            
            current_iter_loss_sum += loss.item() # sum un-normalized loss for logging
            loss = loss / config.accumulation_steps
            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if (iter_num + 1) % args.log_interval == 0:
            avg_loss_for_iter = current_iter_loss_sum / config.accumulation_steps # average loss for the effective batch
            logger.info(f"iter {iter_num+1}/{config.max_iters} | loss: {avg_loss_for_iter:.4f}")

        if (iter_num + 1) % config.eval_interval == 0:
            val_dataset_iterator = iter(val_iterable_dataset) # get fresh iterator for val set
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
    parser.add_argument("--log_interval", type=int, default=100, help="logging interval.")
    parser.add_argument("--save_interval", type=int, default=5000, help="periodic checkpoint saving interval.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="path to checkpoint to resume from.")
    
    args = parser.parse_args()
    main(args)