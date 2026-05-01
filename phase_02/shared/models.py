import torch
import torch.nn as nn

class DualHeadLayerRouter(nn.Module):
    def __init__(self, 
                 num_layers: int = 5, 
                 embed_dim: int = 128, 
                 hidden_dim: int = 256, 
                 dropout: float = 0.1,
                 use_layernorm: bool = True):
        """
        Phase 02 Dual-Head LayerRouter.
        Implements Multi-Task Learning for ColBERT Margin and Mistral Soft Label Margin.
        """
        super().__init__()
        
        # Backbone input configuration based on Phase 1 Golden Recipe (abl2_inter_norm)
        # Features: [Q; D; |Q-D|; Q*D]
        feature_dim = num_layers * embed_dim
        input_dim = feature_dim * 4
        
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norm = nn.LayerNorm(input_dim)
            
        # ---------------------------------------------------------
        # Shared Backbone (Extracts task-agnostic semantic features)
        # ---------------------------------------------------------
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---------------------------------------------------------
        # Task-Specific Heads
        # ---------------------------------------------------------
        # Head C: Predicts ColBERT's introspective margin (Self-Correction)
        self.head_c = nn.Linear(hidden_dim, 1) 
        
        # Head M: Predicts Mistral 7B's semantic margin (Knowledge Distillation)
        self.head_m = nn.Linear(hidden_dim, 1) 

    def forward(self, q_reps, d_reps):
        """
        Args:
            q_reps: (batch, num_layers, embed_dim)
            d_reps: (batch, num_layers, embed_dim)
        Returns:
            pred_c: ColBERT margin prediction
            pred_m: Mistral margin prediction
        """
        # 1. Flatten layer representations
        batch_size = q_reps.size(0)
        q_flat = q_reps.reshape(batch_size, -1)
        d_flat = d_reps.reshape(batch_size, -1)
        
        # 2. Fusion Strategy (Interaction + Difference)
        diff = torch.abs(q_flat - d_flat)
        prod = q_flat * d_flat
        x = torch.cat([q_flat, d_flat, diff, prod], dim=-1)
        
        # 3. Normalization
        if self.use_layernorm:
            x = self.layer_norm(x)
            
        # 4. Shared Representation Extraction
        shared_repr = self.shared_backbone(x)
        
        # 5. Multi-Head Predictions
        pred_c = self.head_c(shared_repr).squeeze(-1)
        pred_m = self.head_m(shared_repr).squeeze(-1)
        
        return pred_c, pred_m
