import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

# GPT-2 (124M) model from scratch, taken from Karpathy's NanoGPT.


@dataclass
class Config:
    block_size: int = 1024  # maximum sequence length
    vocab_size: int = 50257  # number of tokens in our vocabulary
    n_layer: int = 12
    n_head: int = 12
    n_embeddings: int = 768  # embedding dimensions


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embeddings % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embeddings, 3 * config.n_embeddings)
        self.c_proj = nn.Linear(config.n_embeddings, config.n_embeddings)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embeddings

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length and embedding dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # normal attention computation
        # attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # apply the mask
        # attention = attention.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attention = F.softmax(attention, dim=-1)
        # y = attention @ v

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron.
    Linear layer -> GELU activation -> Linear layer.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embeddings, 4 * config.n_embeddings)
        self.gelu = nn.GELU(approximate="tanh")  # the approximate version of GELU
        self.c_proj = nn.Linear(4 * config.n_embeddings, config.n_embeddings)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attentionb block of in GPT-2 model.
    Composed of layer normalization, a self-attention layer, and another layer normalization.
    Finally, a MLP.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embeddings)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embeddings)
        self.mlp = MLP(config)

    def forward(self, x) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embeddings),
                wpe=nn.Embedding(config.block_size, config.n_embeddings),
                h=nn.ModuleList(
                    [AttentionBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embeddings),
            )
        )
        self.lm_head = nn.Linear(config.n_embeddings, config.vocab_size, bias=False)

        # weight sharing between token embedding and output layer (saves a lot of params)
        self.transformer.wte.weight = self.lm_head.weight

        # initialize the weights like in the original GPT-2
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx) -> torch.Tensor:
        # token idx (batch, sequence length)
        B, T = idx.size()
        assert T <= self.config.block_size, "Problem with T: larger than block size"

        pos = torch.arange(0, T, dtype=torch.int, device=idx.device)
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(pos)

        x: torch.Tensor = position_embeddings + token_embeddings

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]

        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["n_embeddings"] = config_args["n_embd"]
        del config_args["n_embd"]

        config = Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
