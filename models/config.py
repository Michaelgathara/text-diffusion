class ModelConfig:
    def __init__(self):
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
        
        # According to papers these should be a good starting point
        # TODO: Dig into this more
        self.beta1 = 0.9
        self.beta2 = 0.99 
        
        self.checkpoint_dir = 'diffusion_checkpoints'
        self.log_dir = 'diffusion_logs'