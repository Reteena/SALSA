from transformers import WavLMModel
import torch, torch.nn as nn, torch.nn.functional as F
import peft
import math

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size=768, heads=12, dropout=0.1):
        """
        Attention-based pooling: summarizes (B, T, D) -> (B, D)
        by learning per-head query vectors.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_dim = hidden_size // heads

        # learnable queries, one per head
        self.q = nn.Parameter(torch.randn(heads, self.head_dim))

        # projections
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # init
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.q, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        """
        x: (B, T, D) from WavLMModel
        mask: (B, T) optional, True=keep, False=pad
        """
        B, T, D = x.shape
        H, d = self.heads, self.head_dim

        k = self.k_proj(x).view(B, T, H, d).permute(0, 2, 1, 3)  # (B,H,T,d)
        v = self.v_proj(x).view(B, T, H, d).permute(0, 2, 1, 3)  # (B,H,T,d)

        q = self.q.unsqueeze(0).expand(B, -1, -1).unsqueeze(2)   # (B,H,1,d)

        # scores: (B,H,1,T)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)

        if mask is not None:
            mask_ = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            scores = scores.masked_fill(~mask_, float("-1e9"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        pooled = torch.matmul(attn, v).squeeze(2)  # (B,H,d)
        pooled = pooled.reshape(B, H * d)          # (B,D)
        return self.out_proj(pooled)               # (B,D)


class AcousticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.wavlm = wavlm_model

        # freze wavlm
        for p in wavlm_model.parameters():
            p.requires_grad = False

        # LoRA
        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
            task_type=peft.TaskType.FEATURE_EXTRACTION,
            layers_to_transform=list(range(len(wavlm_model.encoder.layers) - 4,
                                           len(wavlm_model.encoder.layers))),
            layers_pattern="layers"
        )
        self.wavlm = peft.get_peft_model(wavlm_model, lora_config)
        self.attention_pooling = AttentionPooling(self.wavlm.config.hidden_size)

    def forward(self, x, attention_mask=None):
        # WavLMModel returns dict
        outputs = self.wavlm(x, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (B,T,D)
        pooled = self.attention_pooling(hidden_states, mask=attention_mask)
        return pooled


class AcousticProbe(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.linear(x)
