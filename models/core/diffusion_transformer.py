import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    """
    integer timesteps into vector embeddings using sinusoidal embeddings
    followed by an MLP
    """
    def __init__(self, embed_dim, max_period=10000):
        super().__init__()
        self.embed_dim = embed_dim
        
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer('freqs', freqs)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), 
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, timesteps):
        args = timesteps[:, None].float() * self.freqs[None, :] 
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2 != 0: 
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding) 

class SelfAttention(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # causal mask not strictly needed for diffusion like in autoregressive models,
        # but can be kept if desired or for future flexibility.
        # for diffusion, all tokens usually attend to all others.
        # self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape # batch, time (seq_len), Channels (embed_dim)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.size(-1)**-0.5 # (B, T, T)
        # if using causal mask:
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([SelfAttention(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd) # n_embd = n_head * head_size
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.Mish(), # Or GELU/SiLU. Mish is often good.
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

        # for AdaLN (Adaptive Layer Normalization) or FiLM (Feature-wise Linear Modulation)
        # this is where you'd project timestep_embedding to get scale/shift parameters
        # for simplicity, we'll start without it, but it's a common improvement.
        # example: self.ada_ln_scale_shift = nn.Linear(config.n_embd_timestep, 2 * config.n_embd)

    def forward(self, x, timestep_embedding=None):
        # x: (B, T, C)
        # timestep_embedding: (B, C_ts) - if used for AdaLN/FiLM

        # if using AdaLN:
        # scale_shift = self.ada_ln_scale_shift(timestep_embedding).unsqueeze(1) # (B, 1, 2*C)
        # scale, shift = scale_shift.chunk(2, dim=-1) # (B, 1, C), (B, 1, C)
        # x_norm = self.ln1(x) * (1 + scale) + shift
        # x = x + self.attn(x_norm)
        
        # standard Pre-Norm structure:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DiffusionTransformerModel(nn.Module):
    def __init__(self, config): # config is an instance of ModelConfig
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        self.timestep_embedding = TimestepEmbedding(config.n_embd) # timestep embedding dim matches n_embd

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        self.final_ln = nn.LayerNorm(config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.n_embd)

        self.apply(self._init_weights)

        print(f"DiffusionTransformerModel initialized with {sum(p.numel() for p in self.parameters()):,} parameters.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, noised_token_ids_or_embeddings, timesteps, targets_noise=None, input_is_embeddings=False):
        # noised_token_ids_or_embeddings: (batch_size, seq_len) if token_ids, or (B, T, C) if embeddings
        # timesteps: (batch_size,) - integer timesteps
        # targets_noise: (batch_size, seq_len, embed_dim) - the actual noise added (if model predicts noise)
        # input_is_embeddings: bool, True if noised_token_ids_or_embeddings are already embeddings

        B, T = noised_token_ids_or_embeddings.shape[:2]

        if input_is_embeddings:
            x = noised_token_ids_or_embeddings # input is already (B, T, C_emb)
        else:
            token_embed = self.token_embedding(noised_token_ids_or_embeddings) # (B, T, C_emb)
            x = token_embed

        pos_embed = self.position_embedding(torch.arange(T, device=x.device)) # (T, C_emb)
        time_embed = self.timestep_embedding(timesteps) # (B, C_emb_ts)

        x = x + pos_embed.unsqueeze(0) # (B, T, C_emb)
        
        x = x + time_embed.unsqueeze(1)

        for block in self.blocks:
            # if blocks are adapted for AdaLN, pass time_embed here:
            # x = block(x, time_embed)
            x = block(x) 

        x = self.final_ln(x)
        predicted_output = self.output_projection(x) # (B, T, C_emb), predicted noise

        loss = None
        if targets_noise is not None:
            loss = F.mse_loss(predicted_output, targets_noise)
        
        return predicted_output, loss