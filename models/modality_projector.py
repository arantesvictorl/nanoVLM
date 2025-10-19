# Modality Projection from Vision to Language
import torch
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim, bias=False),
            nn.LayerNorm(self.output_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor
        
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)
        
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x


class VisualRegisters(nn.Module):
    """
    Visual Compact Token Registers para sumarizar tokens visuais.
    
    Implementa registros aprendíveis que comprimem informação visual através
    de atenção com tokens visuais, seguindo o conceito de "Visual Compact Token Registers".
    
    Args:
        cfg: Configuracao contendo:
            - victor_num_registers (int): Numero de registros visuais
            - lm_hidden_dim (int): Dimensao dos embeddings do modelo de linguagem
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_registers = cfg.victor_num_registers
        self.hidden_dim = cfg.lm_hidden_dim
        
        # Registros aprendiveis inicializados aleatoriamente
        self.registers = nn.Parameter(torch.randn(1, self.num_registers, self.hidden_dim) * 0.02)
        
        # Mecanismo de atenção para comprimir tokens visuais
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,  # Ajustar conforme necessário
            dropout=0.1,
            batch_first=True
        )
        
        # Normalização e projeção
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        # Gate aprendível para controlar influência
        self._gate = nn.Parameter(torch.tensor(-1.5))
    
    def forward(self, visual_tokens, num_images):
        """
        Comprime tokens visuais em registros compactos usando atenção.
        
        Args:
            visual_tokens: Tokens visuais [num_images, num_visual_tokens, hidden_dim]
            num_images (int): Número de imagens
            
        Returns:
            torch.Tensor: Registros comprimidos [num_images, num_registers, hidden_dim]
        """
        batch_size = visual_tokens.size(0)
        
        # Expandir registros para o batch
        registers = self.registers.expand(batch_size, -1, -1)  # [B, num_registers, hidden_dim]
        
        # Aplicar atenção: registros "atendem" aos tokens visuais
        # Query: registros, Key/Value: tokens visuais
        attended_registers, _ = self.attention(
            query=registers,  # [B, num_registers, hidden_dim]
            key=visual_tokens,  # [B, num_visual_tokens, hidden_dim]
            value=visual_tokens  # [B, num_visual_tokens, hidden_dim]
        )
        
        # Residual connection + layer norm
        registers = self.norm1(registers + attended_registers)
        
        # FFN
        ffn_out = self.ffn(registers)
        registers = self.norm2(registers + ffn_out)
        
        # Gate para controlar influência
        gate = torch.sigmoid(self._gate)
        registers = gate * registers
        
        return registers
