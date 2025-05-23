# In models/config.py
class ModelConfig:
    def __init__(self):
        # --- Model Architecture Parameters ---
        self.vocab_size = 50257  # Example: GPT-2 vocab size, adjust as needed
        self.n_embd = 768       # Embedding dimension (e.g., 512, 768)
        self.n_head = 12        # Number of attention heads (e.g., 8, 12)
        self.n_layer = 12       # Number of transformer layers (e.g., 6, 12)
        self.block_size = 256   # Max sequence length / context window (e.g., 128, 256, 512)
        self.dropout = 0.1      # Dropout rate

        self.diffusion_timesteps = 1000
        self.noise_schedule_type = 'cosine' # ('linear', 'cosine', 'sqrt_linear', etc.)

        self.batch_size = 32
        self.accumulation_steps = 8
        
        self.max_iters = 100_000
        self.eval_interval = 1000
        self.eval_iters = 100
        self.warmup_iters = 1500
        
        self.learning_rate = 1e-4
        self.weight_decay = 0.1
        
        self.beta1 = 0.9
        self.beta2 = 0.99 
        
        self.checkpoint_dir = 'diffusion_checkpoints'
        self.log_dir = 'diffusion_logs'
        self.seed = 1337 