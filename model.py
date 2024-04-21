import torch
import torch.nn as nn

# Hiperparâmetros
vocab_size = 65024  # Tamanho do vocabulário
max_len = 8192  # Janela de contexto máxima (8k tokens)
d_model = 14848  # Dimensão do modelo
num_heads = 64  # Número de cabeças de atenção
num_layers = 80  # Número de camadas
dropout = 0.1  # Taxa de dropout
top_k = 50  # Top-k para amostragem
top_p = 0.95  # Top-p (núcleo) para amostragem
freq_penalty = 1.2  # Penalidade para tokens frequentes
max_tokens = 4000 # Número máximo de tokens a gerar
mem_size = 8192  # Tamanho da memória (em tokens)

# Camada de embeddings
embedding = nn.Embedding(vocab_size, d_model)

# Encoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_output)
        return out2

encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])

# Decoder
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mem):
        attn_output1, _ = self.attn1(x, x, x, need_weights=False)
        out1 = self.norm1(x + attn_output1)
        attn_output2, attn_weights = self.attn2(out1, encoder_output, encoder_output, need_weights=True)
        out2 = self.norm2(out1 + attn_output2)
        ffn_output = self.ffn(out2)
        out3 = self.norm3(out2 + ffn_output)
        mem = torch.cat([mem, out3], dim=1)
        return out3, mem, attn_weights

decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])

# Modelo Transformer
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder_layers
        self.decoder = decoder_layers
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, target_ids=None, mem=None):
        input_embeds = self.embedding(input_ids)
        if target_ids is not None:
            target_embeds = self.embedding(target_ids)

        encoder_output = input_embeds
        for layer in self.encoder:
            encoder_output = layer(encoder_output)

        if target_ids is None:
            mem = torch.zeros(1, 0, d_model, device=input_ids.device)
            output = []
            prev_output = torch.LongTensor([[0]]).to(input_ids.device)
            for _ in range(max_tokens):
                prev_embeds = self.embedding(prev_output).squeeze(1)
                decoder_output, mem, _ = self.decoder[0](prev_embeds, encoder_output, mem)
                for layer in self.decoder[1:]:
                    decoder_output, mem, _ = layer(decoder_output, encoder_output, mem)
                logits = self.output(decoder_output[:, -1])
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, filter_value=-float('Inf'))
                next_token = sample_from_filtered_logits(filtered_logits, freq_penalty)
                output.append(next_token.item())
                prev_output = next_token.unsqueeze(0)
            return output
        else:
            decoder_output = target_embeds
            for layer in self.decoder:
                decoder_output, mem, _ = layer(decoder_output, encoder_output, mem)

            output = self.output(decoder_output)
            return output

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        topk_logits, topk_indices = torch.topk(logits, top_k)
        weight_logits = topk_logits.clone()
        weight_logits[weight_logits < topk_logits[..., -1, None]] = filter_value
        logits = weight_logits
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

def sample_from_filtered_logits(logits, freq_penalty=1.0):
    logits = logits / freq_penalty
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
