from model import GPT, Config
import torch
from torch.nn import functional as F
import tiktoken


def inference(
    n_repeat: int = 5, max_length: int = 30, pretrained: bool = False, seed: int = 42
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(n_repeat, 1)

    x = tokens.to(device)

    torch.manual_seed(seed)
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
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(decoded)


def forward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # data preparation
    with open("input.txt", "r") as f:
        text = f.read(512)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)

    B, T = 4, 32
    buf = torch.tensor(tokens[: B * T + 1])

    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    print("Input shape: ", x.shape)
    print("Target shape: ", y.shape)

    # load the model
    model = GPT(Config())
    model.to(device)

    # forward pass
    out = model(x)
    print("Logits shape: ", out.shape)


if __name__ == "__main__":
    forward()
