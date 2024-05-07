import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the keys and values
    vocab_size: int = -1 # This will be set during tokenizer
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim%2==0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula theta_i = 
    # Shape : (Head_dim/2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape : (Head_dim/2)
    theta = 1.0 / (theta** (theta_numerator / head_dim)).to(device)
    m = torch.arrange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device:str):

    # (B, seq_len, Head_dim) -> (B, seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[-1],-1,2))
    # (Seq_len, Head_dim/2) -> (1, Seq_len, 1, Head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x.out.type_as(x).to(device)



def  repeat_kv(x: torch.Tensor, n_repeats: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_repeats == 1:
        return x
    return (
        # (B, seq_len, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # (B, seq_ken, n_kv_heads, n_repeats, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, n_repeats, head_dim)
        # (B, seq_len, n_kv_heads * n_repeats, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_repeats, head_dim)
    )



class FeedForward(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()


        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)

        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()


        # Indicate the number of keys and values heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicate the number of heads for the queries
        self.n_q_heads = args.n_heads
        # Indicate how many times the keys and vakyes should be repeated
        self.n_reap = self.n_q_heads // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim,args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim,self.n_kv_heads*self.head_dim ,bias = False)
        self.wv = nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias = False)
        self.wo = nn.Linear(args.n_heads*self.head_dim,args.dim, bias = False)


        self.cache_keys = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_values = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))



    @staticmethod
    def attention(query, key, value, head_dim: int):

        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(head_dim)
        attention_scores = F.softmax(attention_scores, dim=-1).type_as(query)
        output =  attention_scores @ value

        return output
    
    def forwward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freq_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape # (Bias,1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        query = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        key = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        value = self.wv(x)

         # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        query = query.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
         # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        query = query.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        query = apply_rotary_embeddings(query, start_pos, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        key = apply_rotary_embeddings(key, start_pos, device=x.device)

        # entry is replaced in cache
        # (B, seq_len,h_kv,head_dim)
        self.cache_keys[:batch_size, start_pos: start_pos:seq_len] = key
        self.cache_values[:batch_size, start_pos: start_pos+seq_len] = value

        # retrieve the cache so far
        keys = self.cache_keys[:batch_size,0:start_pos+seq_len]

        values = self.cache_values[:batch_size,0:start_pos+seq_len]

        # Q -> share the same key and value for all the positions, we just need to repeat the key and value head 
        keys = repeat_kv(keys, self.n_reap)
        values = repeat_kv(values, self.n_reap)

        query  = query.transpose(1,2)
        value = value.transpose(1,2)
        key = key.transpose(1,2)

        output = MultiHeadAttentionBlock.attention(query, key, value, self.head_dim)

        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output)


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = MultiHeadAttentionBlock(args)
        self.feed_forward = FeedForward(args)


        # Normalization before attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x:torch.tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )

        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim,eps = args.norm_eps)
        self.output = nn.Linear(args.dim,self.vocab_size,bias=False)


        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_position:int):

        # (B, seq_len)
        batch_size, seq_len = tokens.size()
        assert seq_len == 1, "Only one token at a time"

        # Converting to embeddings
        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrive the Pairs (m, theta) correposing to the position [start_position, start_position+seq_len] because this is rotary embedding
        freqs_complex = self.freqs_complex(start_position, start_position+seq_len)


        # Apply all the encoder layers
        for layer in self.layers:
            h = layer(h,start_position ,freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output