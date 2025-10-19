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
    Visual Compact Token Registers (VICTOR) para sumarizar tokens visuais.
    
    Esta classe implementa registros aprendiveis que sao usados para comprimir
    a informacao visual em um conjunto menor de tokens, reduzindo o custo
    computacional do modelo de linguagem.
    
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
        # Gate aprendível para controlar a influência dos registros no início do treino
        # Iniciar negativo para gate baixo (~0.1-0.2 após sigmoid)
        self._gate = nn.Parameter(torch.tensor(-1.5))
        # Normalização para manter escala estável ao injetar no LLM
        self.norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, num_images):
        """
        Retorna os registros visuais replicados para o número de imagens.
        
        Args:
            num_images (int): Número de imagens para replicar os registros
            
        Returns:
            torch.Tensor: Registros com shape [num_images, num_registers, hidden_dim]
        """
        gate = torch.sigmoid(self._gate)
        regs = gate * self.norm(self.registers)
        return regs.expand(num_images, -1, -1)
