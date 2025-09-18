"""
Multimodal Fusion with Gated Cross-Attention for SALSA
Implements the fusion module that combines acoustic and lexical features.
Handles variable-length utterance sequences and aggregates to recording-level predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention layer that attends lexical summary over acoustic tokens
    or vice-versa for multimodal fusion.
    """
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, query, key_value, attention_mask=None):
        """
        Args:
            query: (B, hidden_size) - either acoustic or lexical features
            key_value: (B, hidden_size) - the other modality features
            attention_mask: Optional mask for attention
        Returns:
            Fused features with gating
        """
        B = query.size(0)
        
        # Expand dimensions for attention computation
        query_expanded = query.unsqueeze(1)  # (B, 1, hidden_size)
        key_value_expanded = key_value.unsqueeze(1)  # (B, 1, hidden_size)
        
        # Compute Q, K, V
        q = self.q_proj(query_expanded)  # (B, 1, hidden_size)
        k = self.k_proj(key_value_expanded)  # (B, 1, hidden_size)
        v = self.v_proj(key_value_expanded)  # (B, 1, hidden_size)
        
        # Reshape for multi-head attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.hidden_size)
        attn_output = attn_output.squeeze(1)  # (B, hidden_size)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        # Gating mechanism
        gate_input = torch.cat([query, attn_output], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # Apply gate and residual connection
        fused = gate_weights * attn_output + (1 - gate_weights) * query
        fused = self.layer_norm(fused + query)
        
        return fused


class UtteranceAggregator(nn.Module):
    """
    Aggregates variable-length utterance sequences to recording-level representation.
    Supports different aggregation strategies: attention pooling, LSTM, etc.
    """
    def __init__(self, input_dim=768, aggregation_type='attention', dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.aggregation_type = aggregation_type
        
        if aggregation_type == 'attention':
            # Learnable attention pooling
            self.attention_weights = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Linear(input_dim // 2, 1)
            )
        elif aggregation_type == 'lstm':
            # LSTM-based aggregation
            self.lstm = nn.LSTM(input_dim, input_dim // 2, batch_first=True, dropout=dropout)
            self.output_proj = nn.Linear(input_dim // 2, input_dim)
        elif aggregation_type == 'transformer':
            # Transformer encoder for temporal modeling
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=input_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.pooling = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Linear(input_dim // 2, 1)
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sequence, attention_mask=None):
        """
        Args:
            sequence: (batch_size, max_seq_len, input_dim)
            attention_mask: (batch_size, max_seq_len) bool mask (True = valid)
            
        Returns:
            aggregated: (batch_size, input_dim)
        """
        batch_size, max_seq_len, input_dim = sequence.shape
        
        if self.aggregation_type == 'mean':
            # Simple mean pooling with mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
                sequence_masked = sequence * mask
                lengths = attention_mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
                aggregated = sequence_masked.sum(dim=1) / lengths.clamp(min=1)
            else:
                aggregated = sequence.mean(dim=1)
                
        elif self.aggregation_type == 'max':
            # Max pooling with mask
            if attention_mask is not None:
                sequence_masked = sequence.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
                aggregated = sequence_masked.max(dim=1)[0]
            else:
                aggregated = sequence.max(dim=1)[0]
                
        elif self.aggregation_type == 'attention':
            # Attention-based pooling
            attn_weights = self.attention_weights(sequence)  # (batch_size, seq_len, 1)
            
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)
            aggregated = (sequence * attn_weights).sum(dim=1)  # (batch_size, input_dim)
            
        elif self.aggregation_type == 'lstm':
            # LSTM aggregation
            if attention_mask is not None:
                # Pack padded sequence for efficiency
                lengths = attention_mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    sequence, lengths, batch_first=True, enforce_sorted=False
                )
                lstm_out, (hidden, _) = self.lstm(packed)
                aggregated = self.output_proj(hidden[-1])  # Use last hidden state
            else:
                lstm_out, (hidden, _) = self.lstm(sequence)
                aggregated = self.output_proj(hidden[-1])
                
        elif self.aggregation_type == 'transformer':
            # Transformer aggregation
            # Create attention mask for transformer (True = valid, False = padding)
            if attention_mask is not None:
                # Transformer expects (seq_len, seq_len) attention mask
                src_key_padding_mask = ~attention_mask  # False = valid, True = padding
            else:
                src_key_padding_mask = None
                
            transformer_out = self.transformer(
                sequence, 
                src_key_padding_mask=src_key_padding_mask
            )
            
            # Apply attention pooling to transformer output
            attn_weights = self.pooling(transformer_out)  # (batch_size, seq_len, 1)
            
            if attention_mask is not None:
                attn_weights = F.softmax(attn_weights, dim=1)
            aggregated = (transformer_out * attn_weights).sum(dim=1)
            
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
            
        return self.dropout(aggregated)


class MultimodalFusion(nn.Module):
    """
    Complete multimodal fusion module for SALSA.
    Handles variable-length utterance sequences and produces recording-level predictions.
    """
    def __init__(self, 
                 acoustic_dim=768, 
                 lexical_dim=400,  # 384 embeddings + 16 handcrafted features
                 fusion_dim=512, 
                 num_classes=2,
                 lora_rank=8,
                 use_cross_attention=True,
                 aggregation_type='attention',
                 num_fusion_layers=2,
                 dropout=0.1):
        super().__init__()
        
        self.acoustic_dim = acoustic_dim
        self.lexical_dim = lexical_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.use_cross_attention = use_cross_attention
        self.aggregation_type = aggregation_type
        
        # Acoustic branch (optional - for when we have audio)
        if acoustic_dim > 0:
            self.acoustic_branch = self._build_acoustic_branch(lora_rank)
            self.acoustic_proj = nn.Linear(acoustic_dim, fusion_dim)
        else:
            self.acoustic_branch = None
            self.acoustic_proj = None
            
        # Lexical feature processing
        self.lexical_proj = nn.Linear(lexical_dim, fusion_dim)
        
        # Utterance-level fusion (for each utterance in a recording)
        if use_cross_attention and acoustic_dim > 0:
            self.cross_attention_layers = nn.ModuleList([
                GatedCrossAttention(fusion_dim, num_heads=8, dropout=dropout)
                for _ in range(num_fusion_layers)
            ])
        else:
            self.cross_attention_layers = None
            
        # Utterance-level feature transformation
        self.utterance_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2 if acoustic_dim > 0 else fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Recording-level aggregation
        self.recording_aggregator = UtteranceAggregator(
            input_dim=fusion_dim,
            aggregation_type=aggregation_type,
            dropout=dropout
        )
        
        # Final classifier for recording-level prediction
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _build_acoustic_branch(self, lora_rank=8):
        """Build acoustic branch with LoRA adapters."""
        from acoustic.acoustic import AcousticBranch
        # Note: In practice, this would be integrated with actual audio processing
        # For now, we'll assume acoustic features are pre-computed
        return None
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                lexical_features: torch.Tensor,
                acoustic_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multimodal fusion.
        
        Args:
            lexical_features: (batch_size, max_seq_len, lexical_dim) utterance features
            acoustic_features: (batch_size, max_seq_len, acoustic_dim) acoustic features (optional)
            attention_mask: (batch_size, max_seq_len) bool mask (True = valid utterance)
            
        Returns:
            logits: (batch_size, num_classes) recording-level predictions
        """
        batch_size, max_seq_len, _ = lexical_features.shape
        
        # Process lexical features
        lexical_proj = self.lexical_proj(lexical_features)  # (batch_size, max_seq_len, fusion_dim)
        
        # Process acoustic features if available
        if acoustic_features is not None and self.acoustic_proj is not None:
            acoustic_proj = self.acoustic_proj(acoustic_features)  # (batch_size, max_seq_len, fusion_dim)
            
            # Apply cross-attention fusion at utterance level
            if self.cross_attention_layers is not None:
                # Flatten sequences for cross-attention
                lexical_flat = lexical_proj.view(-1, lexical_proj.size(-1))  # (batch_size * seq_len, fusion_dim)
                acoustic_flat = acoustic_proj.view(-1, acoustic_proj.size(-1))
                
                # Apply cross-attention layers
                fused_lexical = lexical_flat
                fused_acoustic = acoustic_flat
                
                for layer in self.cross_attention_layers:
                    # Lexical attends to acoustic
                    temp_lexical = layer(fused_lexical.unsqueeze(0), fused_acoustic.unsqueeze(0))
                    temp_lexical = temp_lexical.squeeze(0)
                    
                    # Acoustic attends to lexical  
                    temp_acoustic = layer(fused_acoustic.unsqueeze(0), fused_lexical.unsqueeze(0))
                    temp_acoustic = temp_acoustic.squeeze(0)
                    
                    fused_lexical = temp_lexical
                    fused_acoustic = temp_acoustic
                
                # Reshape back to sequences
                fused_lexical = fused_lexical.view(batch_size, max_seq_len, self.fusion_dim)
                fused_acoustic = fused_acoustic.view(batch_size, max_seq_len, self.fusion_dim)
                
                # Concatenate modalities for utterance-level fusion
                utterance_features = torch.cat([fused_lexical, fused_acoustic], dim=-1)
            else:
                # Simple concatenation without cross-attention
                utterance_features = torch.cat([lexical_proj, acoustic_proj], dim=-1)
        else:
            # Lexical-only mode
            utterance_features = lexical_proj
            
        # Apply utterance-level fusion
        utterance_fused = self.utterance_fusion(utterance_features)  # (batch_size, max_seq_len, fusion_dim)
        
        # Aggregate utterances to recording-level representation
        recording_features = self.recording_aggregator(
            utterance_fused, 
            attention_mask=attention_mask
        )  # (batch_size, fusion_dim)
        
        # Final classification
        logits = self.classifier(recording_features)  # (batch_size, num_classes)
        
        return logits
    
    def get_utterance_features(self,
                              lexical_features: torch.Tensor,
                              acoustic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get utterance-level features before recording-level aggregation.
        Useful for analysis and debugging.
        
        Returns:
            utterance_features: (batch_size, max_seq_len, fusion_dim)
        """
        lexical_proj = self.lexical_proj(lexical_features)
        
        if acoustic_features is not None and self.acoustic_proj is not None:
            acoustic_proj = self.acoustic_proj(acoustic_features)
            utterance_features = torch.cat([lexical_proj, acoustic_proj], dim=-1)
        else:
            utterance_features = lexical_proj
            
        return self.utterance_fusion(utterance_features)


# Model factory functions
def create_multimodal_salsa(acoustic_dim=768, lexical_dim=400, **kwargs):
    """Create full multimodal SALSA model."""
    return MultimodalFusion(
        acoustic_dim=acoustic_dim,
        lexical_dim=lexical_dim,
        use_cross_attention=True,
        **kwargs
    )

def create_lexical_only_salsa(lexical_dim=400, **kwargs):
    """Create lexical-only SALSA model."""
    return MultimodalFusion(
        acoustic_dim=0,  # No acoustic features
        lexical_dim=lexical_dim,
        use_cross_attention=False,
        **kwargs
    )

def create_acoustic_only_salsa(acoustic_dim=768, **kwargs):
    """Create acoustic-only SALSA model."""
    return MultimodalFusion(
        acoustic_dim=acoustic_dim,
        lexical_dim=0,  # No lexical features
        use_cross_attention=False,
        **kwargs
    )