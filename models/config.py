# In models/config.py
class ModelConfig:
    def __init__(self):
        self.vocab_size = 50257  # Example: GPT-2 vocab size, adjust as needed
        self.n_embd = 2560       # Embedding dimension (increased from 2048)
        self.n_head = 32        # Number of attention heads (kept same)
        self.n_layer = 40      # Number of transformer layers (increased from 32)
        self.block_size = 1024   # Max sequence length / context window (e.g., 128, 256, 512)
        self.dropout = 0.1      # Dropout rate
        
        # mem stuff
        self.use_flash_attention = True
        self.use_gradient_checkpointing = True 

        self.diffusion_timesteps = 1000
        self.noise_schedule_type = 'cosine' # ('linear', 'cosine', 'sqrt_linear', etc.)
        self.cosine_schedule_s = 0.008
        
        self.beta_start = 0.0001
        self.beta_end = 0.02

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