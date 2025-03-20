from model import GPT, Config
import torch
from torch.nn import functional as F
import tiktoken
import time
import math


class ShakeSpeareLoader:
    # TODO: as a custom dataset instead?
    def __init__(self, B, T):
        # data preparation
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_pos = 0

    def next_batch(self):
        buf = self.tokens[self.current_pos : self.current_pos + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_pos += self.B * self.T
        if self.current_pos + (self.B * self.T + 1) > len(self.tokens):
            self.current_pos = 0

        return x, y


MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 50


def get_lr(step: int) -> float:
    # warmup and after warmup regions
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    elif step > WARMUP_STEPS:
        return MIN_LR

    # in between: decrease the LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def setup_optimizer(model: GPT) -> torch.optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    no_decay_params = [p for p in params if p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    return optimizer


def inference(
    n_repeat: int = 5, max_length: int = 30, pretrained: bool = False, seed: int = 42
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if pretrained:
        model = GPT.from_pretrained("gpt2")
        print("Loaded HF Pretrained GPT-2. Forward pass...")
    else:
        model = GPT(Config())
        print("Loaded random GPT-2. Forward pass...")

    model.eval()
    model.to(device)

    # the gpt-2 tokenizer (using OpenAI library)
    # compression ratio: 3 to 1 (1000 char -> 300 tokens)
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode("Hello, I'm a language model,"), dtype=torch.int)
    tokens = tokens.unsqueeze(0).repeat(n_repeat, 1)

    x = tokens.to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            topk_probs, top_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)

            xcol = torch.gather(top_indices, -1, ix)

            x = torch.cat((x, xcol), dim=1)

    for i in range(n_repeat):
        tokens: list[int] = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(decoded)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # For final version: B = 16, T = 1024
    train_loader = ShakeSpeareLoader(B=4, T=512)

    # torch.set_float32_matmul_precision("high") #TF32 setting
    # load the model
    # override the vocab size to the next power of 2
    model = GPT(Config(vocab_size=50304))
    model.to(device)

    print("Compiling GPT-2...")
    model = torch.compile(model)

    optimizer = setup_optimizer(model)

    steps = 100
    for step in range(steps):
        start_time = time.time()
        # Forward Pass
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # use autocast to use bfloat16 for fwd
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = model(x)

        # Backward Pass
        optimizer.zero_grad()
        # (N=B*T, Vocab size (# classes)) vs target (N,)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), y.flatten())
        loss.backward()
        # avoid shock from high gradients through clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()
        backward_time = time.time()

        train_time = backward_time - start_time
        print(
            f"step: {step:04d} | loss: {loss.item():.6f} | norm: {norm:.4f} | time: {1000 * train_time:.4f}ms"
        )


if __name__ == "__main__":
    main()
