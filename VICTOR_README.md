# Implementação do VICTOR no nanoVLM

Esta branch adiciona suporte ao **VICTOR (Visual Compact Token Registers)** ao nanoVLM, baseado no paper [Efficient Vision-Language Models by Summarizing Visual Tokens into Compact Registers](https://arxiv.org/html/2410.14072v1).

## O que é VICTOR?

VICTOR é um método que reduz o número de tokens visuais através de registros compactos aprendíveis, melhorando significativamente a eficiência computacional de modelos de visão-linguagem sem perda significativa de desempenho.

### Principais Características

- **Registros Visuais Aprendíveis**: Adiciona um pequeno conjunto de tokens aprendíveis após os tokens visuais
- **Sumarização nas Primeiras Camadas**: Usa as primeiras k camadas do LLM para sumarizar informação visual nos registros
- **Descarte de Tokens**: Após a camada k, todos os tokens visuais originais são descartados, mantendo apenas os registros compactos
- **Overhead Mínimo**: Adiciona apenas ~0.03% de parâmetros ao modelo total

### Benefícios Esperados

Conforme o paper original:
- ✅ **43% de redução** no tempo de treinamento
- ✅ **3.3x aumento** no throughput de inferência  
- ✅ **<4% de queda** na acurácia com apenas 8 registros (~1% dos tokens originais)
- ✅ Compatível com implementações eficientes de atenção (FlashAttention)

## Alterações Implementadas

### Arquivos Modificados

1. **`models/config.py`**
   - Adiciona `use_victor`: bool - Flag para ativar/desativar VICTOR
   - Adiciona `victor_num_registers`: int - Número de registros visuais (padrão: 8)
   - Adiciona `victor_drop_layer`: int - Camada onde tokens são descartados (padrão: 3)

2. **`models/modality_projector.py`**
   - Nova classe `VisualRegisters`: Implementa registros visuais aprendíveis
   - Parâmetros inicializados aleatoriamente com std=0.02

3. **`models/vision_language_model.py`**
   - Método `_add_visual_registers()`: Adiciona registros após tokens visuais
   - Forward pass modificado para integrar registros
   - Suporte em treinamento e inferência (generate)

4. **`models/language_model.py`**
   - Método `_drop_visual_tokens()`: Remove tokens visuais originais após camada k
   - Forward pass aceita `victor_info` para controlar o descarte
   - Atualiza attention mask e embeddings posicionais correspondentes

### Arquivos Novos

5. **`test_victor.py`**
   - Script completo de testes com 3 validações
   - Testa inicialização e forward pass
   - Compara VICTOR ativado vs desativado
   - Verifica contagem de parâmetros

## Como Usar

### Configuração Básica

```python
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel

# Ativar VICTOR
cfg = VLMConfig()
cfg.use_victor = True
cfg.victor_num_registers = 8  # Ajustar conforme necessário
cfg.victor_drop_layer = 3     # Camada onde os tokens são descartados

# Inicializar modelo
model = VisionLanguageModel(cfg)
```

### Treinamento

```python
# O VICTOR funciona automaticamente durante o treinamento
# Nenhuma alteração necessária no código de treinamento

# Exemplo com train.py
python train.py --compile False
```

### Inferência

```python
# VICTOR também funciona automaticamente na inferência
outputs = model.generate(
    input_ids=input_ids,
    images=images,
    max_new_tokens=100
)
```

### Desativar VICTOR

```python
cfg = VLMConfig()
cfg.use_victor = False  # Volta ao comportamento padrão
```

## Testes

Execute o script de teste para validar a implementação:

```bash
# Ativar ambiente virtual
source .venv/bin/activate

# Executar testes
python test_victor.py
```

O script executa 3 testes:
1. **Teste básico**: Inicialização e forward pass com VICTOR
2. **Comparação**: VICTOR ativado vs desativado
3. **Parâmetros**: Verifica overhead de parâmetros

## Configurações Recomendadas

### Para Máxima Eficiência

```python
cfg.victor_num_registers = 8   # Registros mínimos
cfg.victor_drop_layer = 2      # Descarte mais cedo
```

**Trade-off**: Máxima velocidade, possível queda maior na acurácia (~4%)

### Para Melhor Qualidade

```python
cfg.victor_num_registers = 64  # Mais registros
cfg.victor_drop_layer = 6      # Descarte mais tarde
```

**Trade-off**: Menor queda de acurácia, menor ganho de velocidade

### Balanceado (Recomendado)

```python
cfg.victor_num_registers = 8   # Conforme o paper
cfg.victor_drop_layer = 3      # Conforme o paper
```

**Trade-off**: Bom equilíbrio entre velocidade e qualidade

## Ablations e Experimentos

### Variando Número de Registros

```python
for num_regs in [4, 8, 16, 32, 64, 128]:
    cfg.victor_num_registers = num_regs
    # Treinar e avaliar
```

### Variando Camada de Descarte

```python
for drop_layer in [1, 2, 3, 4, 5, 6]:
    cfg.victor_drop_layer = drop_layer
    # Treinar e avaliar
```

## Resultados Esperados

Com base no paper original (usando LLaVA-NeXT como baseline):

| Configuração | Registros | Throughput | Acurácia |
|-------------|-----------|------------|----------|
| Baseline | 2880 tokens | 1.0x | 100% |
| VICTOR-256 | 256 tokens | 2.0x | ~98% |
| VICTOR-64 | 64 tokens | 2.8x | ~97% |
| VICTOR-8 | 8 tokens | 3.3x | ~96% |

## Limitações

- **Requer retreinamento**: Não funciona bem com ajuste fino (inference-time adjustment tem queda de performance)
- **Depende da camada de descarte**: A escolha da camada k é crítica
- **Trade-off qualidade/velocidade**: Menos registros = mais rápido mas menor acurácia

## Trabalhos Futuros

- [ ] Loss auxiliar para melhor controle da sumarização
- [ ] Ajuste dinâmico do número de registros em inferência
- [ ] Experimentos com diferentes inicializações dos registros
- [ ] Integração com outras técnicas de compressão de tokens

## Referências

- Paper original: [Efficient Vision-Language Models by Summarizing Visual Tokens into Compact Registers](https://arxiv.org/html/2410.14072v1)
- Wen, Y., Cao, Q., Fu, Q., Mehta, S., & Najibi, M. (2024)
- nanoVLM: [github.com/huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)

## Autor da Implementação

Implementado por: @arantesvictorl
Data: Outubro 2025
Baseado no repositório original do nanoVLM da Hugging Face

---

**Nota**: Esta é uma implementação independente baseada na descrição do paper. Para reproduzir os resultados exatos do paper, ajustes nos hiperparâmetros podem ser necessários.

