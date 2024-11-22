import torch 
from torch import nn
from torch.nn import functional as F

from llm_models.blocks.basic_block import Block

class GPT(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int, context: int, n_layers: int, n_heads: int, dropout: int, bias: bool, device: str):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.positions = nn.Embedding(context, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, n_heads, context, dropout, bias) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_size)
        self.final_linear = nn.Linear(embed_size, vocab_size, bias=bool)
        self.context = context
        self.device = device
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input, targets=None):
        loss = None
        BS, SL = input.shape
        emb = self.embeddings(input)
        pos = self.positions(torch.arange(SL, device=self.device))
        x = emb + pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.final_linear(x)

        if targets is not None:
            BS, SL, VS = logits.shape
            logits = logits.view(BS*SL, VS)
            targets = targets.view(BS*SL)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, input, max=500):
        for _ in range(max):
            input = input[:, -self.context:]
            logits, _ = self(input)
            logits = logits[:, -1, :] # pick last probability
            probs = F.softmax(logits, dim=-1) # dim indicates last dimension
            next = torch.multinomial(probs, num_samples=1)
            input = torch.cat((input, next), dim=-1)
        return input

def main():    
    model = GPT(1, 1, 1, 1, 1, False, 'mps')    