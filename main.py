from model import GPT, Config
from dataset import TinyStories
from train_utils import get_lr, setup_optimizer
import torch
from torch.nn import functional as F
import tiktoken
import time


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
    # GPT-3 uses a batch size of 488: we do so with gradient accumulation
    total_batch_size = 52488
    B = 64
    T = 1024
    accumulate_steps = total_batch_size // (B * T)
    train_loader = TinyStories(B=B, T=T, split="train")
    test_loader = TinyStories(B=B, T=T, split="val")

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

        loss_acc = 0.0
        for micro_batch in range(accumulate_steps):
            # Forward Pass
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # use autocast to use bfloat16 for fwd
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                torch.cuda.synchronize()
                forward_time = time.time()

                # Backward Pass
                optimizer.zero_grad()
                # (N=B*T, Vocab size (# classes)) vs target (N,)
                loss = F.cross_entropy(out.view(-1, out.size(-1)), y.flatten())

            loss = loss / accumulate_steps
            loss_acc += loss.detach()
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
        print(f"Forward pass time:\t{1000 * (forward_time - start_time):.4f}ms")
        print(f"Backward pass time:\t{1000 * (backward_time - forward_time):.4f}ms")


if __name__ == "__main__":
    main()
