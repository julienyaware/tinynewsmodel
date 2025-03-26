import torch
import torch.nn.functional as F
from torch import nn
import argparse

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def get_batch(split, batch_size, context_size, train_data, val_data):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - context_size, (batch_size,))
    x = torch.stack([data_split[i:i + context_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + context_size + 1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size, n_embd, context_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, context_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, context_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.ReLU(),
            nn.Linear(2 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, context_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

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
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device) % self.position_embedding_table.num_embeddings)
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

# CLI and Arguments
def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate a Transformer Language Model")
    parser.add_argument('--input', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--train', type=str, required=True, help='Path to save or load the model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--context-size', type=int, default=128, help='Context size')
    parser.add_argument('--n-embd', type=int, default=384, help='Embedding dimension size')
    parser.add_argument('--n-head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n-layer', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--report', type=int, default=500, help='Report frequency')
    parser.add_argument('--evaluate', action='store_true', help='Flag to indicate evaluation')

    args = parser.parse_args()

    # Load your dataset
    with open(args.input, "r") as f:
        text = f.read()

    # Build character-level encoding
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

    model = TransformerLanguageModel(
        vocab_size=vocab_size, 
        n_embd=args.n_embd, 
        context_size=args.context_size, 
        n_head=args.n_head, 
        n_layer=args.n_layer, 
        dropout=args.dropout
    ).to(device)


    if args.evaluate:
        print(f"Evaluating model from {args.train}")
        model.load_state_dict(torch.load(args.train))

    
        evaluate(model, args.context_size, train_data, val_data, char_to_idx, idx_to_char)
    else:

        train(model, steps=args.epochs, batch_size=args.batch_size, context_size=args.context_size, train_data=train_data, val_data=val_data, report_frequency=args.report)
        
        torch.save(model.state_dict(), args.train)
        print(f"Model saved to {args.train}")

def evaluate(model, context_size, train_data, val_data, char_to_idx, idx_to_char):
    model.eval()
    while True:
        prompt = input("Enter a prompt for the model (or 'exit' to stop): ")
        if prompt.lower() == 'exit':
            break

        prompt_encoded = torch.tensor([char_to_idx.get(c, 0) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)

      
        with torch.no_grad():
            for _ in range(700): 
               
                if prompt_encoded.shape[1] > context_size:
                    input_seq = prompt_encoded[:, -context_size:]
                else:
                    input_seq = prompt_encoded
                logits, _ = model(input_seq)
                prob = F.softmax(logits[:, -1, :], dim=-1)
                pred_idx = torch.multinomial(prob, 1)
                prompt_encoded = torch.cat((prompt_encoded, pred_idx), dim=1)  # Add predicted character to the prompt
                print(idx_to_char[pred_idx.item()], end='', flush=True)
            print()

if __name__ == "__main__":
    main()