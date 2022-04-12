import torch 
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self, embed_size:int, heads:int):
        super().__init__()
        self.embed_size = embed_size
        self.heads      = heads
        self.head_dim   = embed_size // heads 

        assert (self.head_dim * heads == embed_size), 'Embed size needs to be div by heads'

        # Linear Layers of Values, Keys, and Queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Liner(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embeddings into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)

        # https://pytorch.org/docs/stable/generated/torch.einsum.html
        # Q * K^T (batched)
        energy = torch.einsum('nqhd,nkhd->nhqk', queries, keys)


