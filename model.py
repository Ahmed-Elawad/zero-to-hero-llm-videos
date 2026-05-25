import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, device):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        self.eval_iters = 200
        self.device= device

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx)
        positional_embed = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_embed + positional_embed
        logits = self.lm_head(token_embed)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # tril = torch.tril(torch.ones(T, T))
            # weight = torch.zeros((T, T))
            # weight = weight.masked_fill(tril==0, float('-inf'))
            # weight = F.softmax(weight, dim=-1)
            # xbow3 = weight @ x
            # torch.allclose()

            loss = F.cross_entropy(logits, targets) 
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx