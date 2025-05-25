import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class AttHead(nn.Module):
    def __init__(self, att_size, head_count, drop_out):
        super().__init__()
        head_size = int(att_size / head_count)
        
        self.head_size = head_size
        self.head_count = head_count
        self.att_size = att_size
        self.drop_out = drop_out

        self.qkv = nn.Linear(att_size, 3 * att_size)

        self.c_proj = nn.Linear(att_size, att_size)
        self.resid_dropout = nn.Dropout(drop_out)

    def forward(self, logits):
        qkv = self.qkv(logits)
        q, k, v = qkv.split(self.att_size, dim=-1)
        B, T, C = k.shape

        k = k.view(B, T, self.head_count, C // self.head_count).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.head_count, C // self.head_count).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.head_count, C // self.head_count).transpose(1, 2) # (B, nh, T, hs)

        # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.drop_out if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y


class AttModule(nn.Module):
    def __init__(self, att_size, head_count, drop_out, layer_count):
        super().__init__()
        assert att_size % head_count == 0

        self.att_head = AttHead(att_size, head_count, drop_out)

        self.linear_1 = nn.Linear(att_size, 4*att_size)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(4*att_size, att_size)
        self.linear_2.is_residual = True
        self.linear_2.num_layers = layer_count
        self.dropout = nn.Dropout(drop_out)

        self.norm1 = nn.LayerNorm(att_size)
        self.norm2 = nn.LayerNorm(att_size)

    def forward(self, logits):
        # attentioned = torch.cat([head(logits) for head in self.att_heads], dim=-1)
        attentioned = self.att_head(self.norm1(logits))
        logits = attentioned + logits

        tmp = self.linear_1(self.norm2(logits))
        tmp = self.gelu(tmp)
        tmp = self.linear_2(tmp)
        tmp = self.dropout(tmp)

        logits = tmp + logits
        return logits

        
class AttentionModel(nn.Module):
    def __init__(self, vocab_size, att_size, head_count, layer_count, context_size, drop_out):
        super().__init__()
        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, att_size)
        self.position_embedding = nn.Embedding(context_size, att_size) # IDK WHY WE NEED

        self.att_layers = nn.ModuleList([AttModule(att_size, head_count, drop_out, layer_count) for _ in range(layer_count)])

        self.last_norm = nn.LayerNorm(att_size)

        self.to_vocab = nn.Linear(att_size, vocab_size)

        self.to_vocab.weight = self.token_embedding_table.weight
        # self.to(DEVICE)

    def forward(self, idx, targets=None):
        # idx = idx.to(DEVICE)
        # if targets is not None:
            # targets = targets.to(DEVICE)
        
        logits = self.token_embedding_table(idx) # (B,T,C)
        B, T, C = logits.shape

        positions = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)

        logits = logits + pos_emb

        for att_layer in self.att_layers:
            logits = att_layer(logits)
  
        logits = self.last_norm(logits)
        logits = self.to_vocab(logits)

        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature = 0.5):
        # idx = idx.to(DEVICE)
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -self.context_size :])
            logits = logits[:, -1, :]
            probs = F.softmax(logits/temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    # Residual scaling trick for final projection layers
    if isinstance(module, nn.Linear) and hasattr(module, 'is_residual'):
        with torch.no_grad():
            nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * module.num_layers))