import os, sys
import torch
from torchinfo import summary

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project Root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)
from models import ModelConfig, DiffusionTransformerModel

def main():
    config = ModelConfig()
    model = DiffusionTransformerModel(config)
    
    batch_size = config.batch_size
    seq_length = config.block_size
    
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    dummy_timesteps = torch.randint(0, config.diffusion_timesteps, (batch_size,))
    dummy_noise = torch.randn(batch_size, seq_length, config.n_embd)
    
    summary(
        model=model,
        input_data=[dummy_input, dummy_timesteps, dummy_noise],  # Pass all required arguments
        dtypes=[torch.long, torch.long, torch.float32],  # Input types for each argument
        col_names=["input_size", "output_size", "num_params", "trainable", "mult_adds"],
        depth=3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1e9:.2f} GB (in FP32)")

if __name__ == "__main__":
    main() 