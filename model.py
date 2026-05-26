import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class Head(nn.Module):

    def __init__(self, head_size, n_embed, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wieght = q @ k.transpose(-2, -1) * C**-0.5
        wieght == wieght.masked_fill(self.tril[:T, :T] == 0, float('-inf') )
        wieght = F.softmax(wieght, dim=-1)
        
        v = self.value(x)
        
        return wieght @ v


class BigramModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, device):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.self_attention_head = Head(n_embed, n_embed, block_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        self.eval_iters = 200
        self.device= device
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embed = self.token_embedding_table(idx)
        positional_embed = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_embed + positional_embed
        x = self.self_attention_head(x)
        logits = self.lm_head(token_embed)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.block_size:]

            logits, loss = self(idx_c)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx