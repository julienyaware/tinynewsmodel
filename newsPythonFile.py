import argparse
import torch
import torch.nn.functional as F
from torch import nn

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    characters = sorted(list(set(text)))
    vocab_size = len(characters)

    char_to_idx = {ch: i for i, ch in enumerate(characters)}
    idx_to_char = {i: ch for i, ch in enumerate(characters)}
    encode = lambda s: [char_to_idx[c] for c in s]
    decode = lambda l: ''.join([idx_to_char[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, encode, decode, vocab_size

# Get batch function
def get_batch(split, batch_size, context_size, train_data, val_data):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - context_size, (batch_size,))
    x = torch.stack([data_split[i:i + context_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + context_size + 1] for i in ix])
    return x.to(device), y.to(device)

# Transformer Model Components
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, context_size=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, context_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

def train(model, steps, batch_size, context_size, train_data, val_data, report_frequency=500):
    optimizer = torch.optim.AdamW(model.parameters())
    model.train()
    for step in range(steps):
        xb, yb = get_batch('train', batch_size, context_size, train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % report_frequency == 0 or step == steps - 1:
            print(f"Step {step}, loss: {loss.item():.4f}")

# Text generation
def generate_with_temperature(model, start_idx, context_size, number_of_tokens, temperature=1.0, top_k=10):
    model.eval()
    idx = start_idx

    for _ in range(number_of_tokens):
        idx_cond = idx[:, -context_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(top_probs, 1)
        next_token = top_idx.gather(-1, next_token)
        idx = torch.cat([idx, next_token], dim=1)

    return idx

def interactive_generation(model, context_size, encode, decode, temperature=1.0, top_k=10):
    model.eval()
    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        start_idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        generated_output = generate_with_temperature(model, start_idx, context_size, number_of_tokens=500, temperature=temperature, top_k=top_k)
        generated_text = decode(generated_output[0].tolist())
        print(f"Generated text: {generated_text}")

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a Transformer language model.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input text file.")
    parser.add_argument('--model_path', type=str, help="Path to save/load the trained model.")
    parser.add_argument('--eval', action='store_true', help="Run the model in evaluation mode.")
    parser.add_argument('--train', action='store_true', help="Run the model in training mode.")
    parser.add_argument('--epoch', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--context_size', type=int, default=128, help="Context size for the model.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for text generation.")
    parser.add_argument('--top_k', type=int, default=10, help="Top-k sampling for text generation.")

    args = parser.parse_args()

    # Load dataset
    train_data, val_data, encode, decode, vocab_size = load_data(args.input)

    model = TransformerLanguageModel(vocab_size, n_embd=128, context_size=args.context_size).to(device)

    if args.train:
        if args.model_path is None:
            print("Error: --model_path must be specified to save the trained model.")
            return
        train(model, steps=args.epoch, batch_size=args.batch_size, context_size=args.context_size, train_data=train_data, val_data=val_data)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    if args.eval:
        if args.model_path is None:
            print("Error: --model_path must be specified to load the model for evaluation.")
            return

        try:
            model.load_state_dict(torch.load(args.model_path))
            model.eval()
            print(f"Model loaded from {args.model_path}")

            prompt = input("Enter a prompt: ")
            start_idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            generated_output = generate_with_temperature(model, start_idx, args.context_size, number_of_tokens=500, temperature=args.temperature, top_k=args.top_k)
            generated_text = decode(generated_output[0].tolist())
            print(f"Generated text: {generated_text}")

        except Exception as e:
            print(f"Error loading the model: {e}")

if __name__ == '__main__':
    main()
