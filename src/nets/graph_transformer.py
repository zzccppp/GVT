import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_proj = nn.Linear(dim, dim)
        self.shift_proj = nn.Linear(dim, dim)

    def forward(self, x, t_emb):
        # x: [B, L, D]
        # t_emb: [B, D]
        scale = self.scale_proj(t_emb).unsqueeze(1)  # [B, 1, D]
        shift = self.shift_proj(t_emb).unsqueeze(1)  # [B, 1, D]
        x = self.norm(x)
        return x * (1 + scale) + shift


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# Helper for Sinusoidal Positional Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Ensure seq_len dimension exists for broadcasting
        if x.ndim == 1:  # Assume input is just seq_len
            positions = torch.arange(x.item(), device=device).float().unsqueeze(1)
        elif x.ndim == 2:  # Assume input is (batch, seq_len) or similar, take seq_len
            positions = (
                torch.arange(x.shape[1], device=device)
                .float()
                .unsqueeze(0)
                .unsqueeze(-1)
            )  # (1, L, 1)
        elif x.ndim == 3:  # Assume input is (batch, seq_len, dim)
            positions = (
                torch.arange(x.shape[1], device=device)
                .float()
                .unsqueeze(0)
                .unsqueeze(-1)
            )  # (1, L, 1)
        else:
            raise ValueError("Unsupported input dimension for SinusoidalPosEmb")

        emb = positions * emb.unsqueeze(
            0
        )  # Broadcasting: (1/B, L, 1) * (1, 1, H/2) -> (1/B, L, H/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))  # Pad last dimension
        return emb


# Reimplement MultiheadAttention to allow for attention bias
class BiasedMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_dropout = nn.Dropout(dropout)

        # Learnable scale for the graph bias
        # Initialize to zero so the initial behavior is standard attention
        self.graph_bias_scale = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(self, x, key_padding_mask=None):
        B, L, D = x.shape

        # 1. Calculate Graph Bias (using input tokens x as proxy for VQ code similarity)
        # Normalize for stability before dot product? Or use normalized x? Let's use x directly here.
        # Consider normalizing x across dim D if needed: x_norm = F.normalize(x, dim=-1)
        # Using x directly: [B, L, D] @ [B, D, L] -> [B, L, L]
        token_similarity = torch.bmm(x, x.transpose(1, 2)) / math.sqrt(
            D
        )  # Normalize by sqrt(D)
        # Expand for heads: [B, L, L] -> [B, 1, L, L] -> [B, H, L, L]
        graph_attn_bias = token_similarity.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # 2. Standard MHA Projection
        qkv = self.qkv(x).chunk(3, dim=-1)  # (q, k, v), each [B, L, D]
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.num_heads), qkv
        )

        # 3. Attention Calculation
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # 4. Add Graph Bias
        attn_scores = attn_scores + self.graph_bias_scale * graph_attn_bias

        # 5. Apply Padding Mask (before softmax)
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True where padded
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(
                mask, -torch.finfo(attn_scores.dtype).max
            )

        # 6. Softmax and Dropout
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 7. Apply attention to Values
        out = torch.einsum("bhij,bhjd->bhid", attn_probs, v)
        out = rearrange(out, "b h l d -> b l (h d)")

        # 8. Output Projection and Dropout
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        # self.attn = nn.MultiheadAttention(
        #     dim, heads, dropout=dropout, batch_first=True, bias=True
        # )
        self.attn = BiasedMultiHeadAttention(dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim, bias=True),
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, t_emb, padding_mask):
        params = self.adaLN_modulation(t_emb)  # [B, 6*D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(
            6, dim=1
        )

        mask = (~padding_mask.unsqueeze(-1)).float()

        # Self-attention
        x_norm1 = self.norm1(x)
        x_mod = modulate(x_norm1, shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        # attn_out, _ = self.attn(
        #     query=x_mod,
        #     key=x_mod,
        #     value=x_mod,
        #     key_padding_mask=padding_mask,
        #     need_weights=False,
        # )
        attn_out = self.attn(x_mod, key_padding_mask=padding_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x * mask

        # FFN
        x_norm2 = self.norm2(x)
        x_mod = modulate(x_norm2, shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        ffn_out = self.ff(x_mod)
        x = x + gate_mlp.unsqueeze(1) * ffn_out
        x = x * mask

        return x


class DiffusionGraphTransformer(nn.Module):
    def __init__(self, num_classes, codebook, dim=128, depth=12, heads=8, dropout=0.1):
        super().__init__()

        self.token_emb = nn.Embedding(
            codebook.shape[0] + 1, codebook.shape[1], padding_idx=codebook.shape[0]
        )
        # copy from codebook to token_emb
        self.token_emb.weight.data[:-1] = codebook
        self.token_emb.weight.data[-1] = torch.zeros(codebook.shape[1])
        # self.token_emb.weight.requires_grad_(False)

        self.token_emb_proj = nn.Linear(codebook.shape[1], dim, bias=False)

        self.pos_emb = SinusoidalPosEmb(dim)
        self.time_emb = TimestepEmbedder(hidden_size=dim, frequency_embedding_size=256)

        self.layers = nn.ModuleList(
            [GraphTransformerLayer(dim, heads, dropout) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)

        self.output_layer = nn.Linear(dim, num_classes, bias=True)
        # self.output_layer = nn.Sequential(
        #     nn.Linear(dim, dim, bias=False),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim, num_classes, bias=False),
        # )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                pass # Embeddings are initialized by default

    def forward(self, x_t, t, padding_mask=None):
        B, L = x_t.shape
        x = self.token_emb(x_t)  # [B, L, D]
        x = self.token_emb_proj(x)  # [B, L, D]

        pos_enc = self.pos_emb(x)  # [B, L, D]
        x = x + pos_enc

        if padding_mask is None:
            padding_mask = x_t == self.token_emb.num_embeddings - 1

        # t = t.unsqueeze(1)  # [B, 1]
        t_emb = self.time_emb(t)  # [B, D]

        for layer in self.layers:
            x = layer(x, t_emb, padding_mask)

        x = self.norm_out(x)
        logits = self.output_layer(x)  # [B, L, num_classes]

        # logits[..., -1] = -1e9
        return logits

    @torch.no_grad()
    def generate_padding_mask(self, x_t, pad_idx):
        return x_t == pad_idx
