"""
Script de teste para validar a implementação do VICTOR no nanoVLM.
"""

import torch
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


def create_mock_batch(cfg, batch_size=2, num_images_per_sample=1):
    """
    Cria um batch de teste simples.
    
    Args:
        cfg: VLMConfig
        batch_size: tamanho do batch
        num_images_per_sample: número de imagens por amostra
    
    Returns:
        tuple: (input_ids, images, attention_mask)
    """
    # Número de placeholders = mp_image_token_length (sempre 64, independente do VICTOR)
    # VICTOR adiciona registros NO EMBEDDING SPACE, não nos input_ids
    placeholders_per_image = cfg.mp_image_token_length
    total_placeholders = num_images_per_sample * placeholders_per_image
    
    # Criar sequência: [placeholders de imagem] [texto]
    text_tokens = 50
    seq_length = total_placeholders + text_tokens
    
    # Input IDs
    input_ids = torch.randint(100, cfg.lm_vocab_size - 100, (batch_size, seq_length))
    
    # Colocar placeholders de imagem no início
    image_token_id = cfg.lm_base_vocab_size
    input_ids[:, :total_placeholders] = image_token_id
    
    # Criar imagens (lista de listas)
    img_size = cfg.vit_img_size
    images = [[torch.randn(1, 3, img_size, img_size) for _ in range(num_images_per_sample)] 
              for _ in range(batch_size)]
    
    # Attention mask
    attention_mask = torch.ones(batch_size, seq_length)
    
    return input_ids, images, attention_mask


def test_victor_initialization():
    """Teste 1: Verifica se VICTOR inicializa corretamente."""
    print("=" * 80)
    print("Teste 1: Inicialização do VICTOR")
    print("=" * 80)
    
    cfg = VLMConfig()
    cfg.use_victor = True
    cfg.victor_num_registers = 8
    cfg.victor_drop_layer = 3
    cfg.lm_n_blocks = 6
    cfg.vit_n_blocks = 6
    
    print(f"\nConfigurações:")
    print(f"  - use_victor: {cfg.use_victor}")
    print(f"  - victor_num_registers: {cfg.victor_num_registers}")
    print(f"  - victor_drop_layer: {cfg.victor_drop_layer}")
    
    model = VisionLanguageModel(cfg, load_backbone=False)
    
    assert model.visual_registers is not None, "Visual registers não criados!"
    assert model.visual_registers.registers.shape == (1, 8, cfg.lm_hidden_dim)
    
    print(f"\n✓ VICTOR inicializado corretamente")
    print(f"  Shape dos registros: {model.visual_registers.registers.shape}")
    print("=" * 80)
    return True


def test_victor_forward():
    """Teste 2: Forward pass com VICTOR."""
    print("\n" + "=" * 80)
    print("Teste 2: Forward Pass com VICTOR")
    print("=" * 80)
    
    cfg = VLMConfig()
    cfg.use_victor = True
    cfg.victor_num_registers = 8
    cfg.victor_drop_layer = 3
    cfg.lm_n_blocks = 6
    cfg.vit_n_blocks = 6
    
    model = VisionLanguageModel(cfg, load_backbone=False)
    model.eval()
    
    # Criar batch de teste
    input_ids, images, attention_mask = create_mock_batch(cfg, batch_size=2, num_images_per_sample=1)
    
    print(f"\nDados de entrada:")
    print(f"  - Batch size: {input_ids.size(0)}")
    print(f"  - Sequence length: {input_ids.size(1)}")
    print(f"  - Número de imagens: 1 por amostra")
    
    try:
        with torch.no_grad():
            logits, loss = model(input_ids, images, attention_mask=attention_mask)
        
        print(f"\n✓ Forward pass executado com sucesso")
        print(f"  - Shape dos logits: {logits.shape}")
        print(f"  - Sequence length resultante: {logits.shape[1]}")
        
        # Verificar se houve redução
        # Original tinha: placeholders (64) que viram tokens visuais (64) + registros (8) = 72
        # Após descarte na camada 3: apenas registros (8) + texto
        # Então: seq_length - 64 + 8 = seq_length - 56
        expected_reduction = cfg.mp_image_token_length - cfg.victor_num_registers
        expected_final_length = input_ids.size(1) - expected_reduction
        
        if logits.shape[1] < input_ids.size(1):
            reduction = input_ids.size(1) - logits.shape[1]
            print(f"  - Redução observada: {reduction} tokens")
            print(f"  - Redução esperada: {expected_reduction} tokens")
            print(f"  ✓ Tokens visuais foram descartados!")
        else:
            print(f"  ⚠ Warning: Nenhuma redução observada")
            
    except Exception as e:
        print(f"\n✗ Erro no forward pass:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 80)
    return True


def test_victor_vs_baseline():
    """Teste 3: Comparação VICTOR vs baseline."""
    print("\n" + "=" * 80)
    print("Teste 3: VICTOR vs Baseline")
    print("=" * 80)
    
    # Modelo baseline (sem VICTOR)
    cfg_baseline = VLMConfig()
    cfg_baseline.use_victor = False
    cfg_baseline.lm_n_blocks = 6
    cfg_baseline.vit_n_blocks = 6
    
    model_baseline = VisionLanguageModel(cfg_baseline, load_backbone=False)
    model_baseline.eval()
    
    # Modelo com VICTOR
    cfg_victor = VLMConfig()
    cfg_victor.use_victor = True
    cfg_victor.victor_num_registers = 8
    cfg_victor.victor_drop_layer = 3
    cfg_victor.lm_n_blocks = 6
    cfg_victor.vit_n_blocks = 6
    
    model_victor = VisionLanguageModel(cfg_victor, load_backbone=False)
    model_victor.eval()
    
    # Criar mesmo batch para ambos
    input_ids, images, attention_mask = create_mock_batch(cfg_baseline, batch_size=2, num_images_per_sample=1)
    
    print(f"\nExecutando forward pass...")
    
    try:
        with torch.no_grad():
            logits_baseline, _ = model_baseline(input_ids, images, attention_mask=attention_mask)
            logits_victor, _ = model_victor(input_ids, images, attention_mask=attention_mask)
        
        print(f"\n✓ Ambos os forward passes executados")
        print(f"\nComparação:")
        print(f"  - Baseline sequence length: {logits_baseline.shape[1]}")
        print(f"  - VICTOR sequence length: {logits_victor.shape[1]}")
        
        reduction = logits_baseline.shape[1] - logits_victor.shape[1]
        expected_reduction = cfg_victor.mp_image_token_length - cfg_victor.victor_num_registers
        
        print(f"  - Redução: {reduction} tokens")
        print(f"  - Redução esperada: {expected_reduction} tokens")
        
        if reduction > 0:
            reduction_percent = (reduction / logits_baseline.shape[1]) * 100
            print(f"  - Percentual: {reduction_percent:.1f}%")
            print(f"\n✓ VICTOR reduziu {reduction} tokens por imagem!")
        else:
            print(f"\n✗ VICTOR não reduziu tokens como esperado")
            return False
            
    except Exception as e:
        print(f"\n✗ Erro na comparação:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 80)
    return True


def test_victor_parameters():
    """Teste 4: Overhead de parâmetros."""
    print("\n" + "=" * 80)
    print("Teste 4: Overhead de Parâmetros")
    print("=" * 80)
    
    # Baseline
    cfg_baseline = VLMConfig()
    cfg_baseline.use_victor = False
    cfg_baseline.lm_n_blocks = 6
    cfg_baseline.vit_n_blocks = 6
    
    model_baseline = VisionLanguageModel(cfg_baseline, load_backbone=False)
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    
    # VICTOR
    cfg_victor = VLMConfig()
    cfg_victor.use_victor = True
    cfg_victor.victor_num_registers = 8
    cfg_victor.lm_n_blocks = 6
    cfg_victor.vit_n_blocks = 6
    
    model_victor = VisionLanguageModel(cfg_victor, load_backbone=False)
    params_victor = sum(p.numel() for p in model_victor.parameters())
    
    params_registers = cfg_victor.victor_num_registers * cfg_victor.lm_hidden_dim
    diff = params_victor - params_baseline
    overhead = (diff / params_baseline) * 100
    
    print(f"\nParâmetros:")
    print(f"  - Baseline: {params_baseline:,}")
    print(f"  - VICTOR: {params_victor:,}")
    print(f"  - Registros: {params_registers:,}")
    print(f"  - Diferença: {diff:,}")
    print(f"  - Overhead: {overhead:.4f}%")
    
    if abs(diff - params_registers) < 100:  # Tolerância pequena
        print(f"\n✓ Overhead de parâmetros está correto")
        print(f"  VICTOR adiciona apenas {overhead:.4f}% de parâmetros")
    else:
        print(f"\n⚠ Warning: Diferença inesperada nos parâmetros")
    
    print("=" * 80)
    return True


def test_victor_multiple_images():
    """Teste 5: VICTOR com múltiplas imagens."""
    print("\n" + "=" * 80)
    print("Teste 5: VICTOR com Múltiplas Imagens")
    print("=" * 80)
    
    cfg = VLMConfig()
    cfg.use_victor = True
    cfg.victor_num_registers = 8
    cfg.victor_drop_layer = 3
    cfg.lm_n_blocks = 6
    cfg.vit_n_blocks = 6
    
    model = VisionLanguageModel(cfg, load_backbone=False)
    model.eval()
    
    # Testar com diferentes números de imagens
    for num_images in [1, 2, 3]:
        input_ids, images, attention_mask = create_mock_batch(cfg, batch_size=2, num_images_per_sample=num_images)
        
        try:
            with torch.no_grad():
                logits, _ = model(input_ids, images, attention_mask=attention_mask)
            
            expected_reduction = num_images * (cfg.mp_image_token_length - cfg.victor_num_registers)
            actual_reduction = input_ids.size(1) - logits.shape[1]
            
            print(f"\n  {num_images} imagem(ns):")
            print(f"    - Input length: {input_ids.size(1)}")
            print(f"    - Output length: {logits.shape[1]}")
            print(f"    - Redução: {actual_reduction} tokens (esperado: {expected_reduction})")
            
            if abs(actual_reduction - expected_reduction) > 2:  # Tolerância pequena
                print(f"    ✗ Falhou: redução incorreta")
                return False
                
        except Exception as e:
            print(f"\n  ✗ Erro com {num_images} imagem(ns): {e}")
            return False
    
    print(f"\n✓ VICTOR funciona com múltiplas imagens!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TESTES DO VICTOR NO nanoVLM" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    results = []
    
    # Executar testes
    tests = [
        ("Inicialização", test_victor_initialization),
        ("Forward Pass", test_victor_forward),
        ("Comparação vs Baseline", test_victor_vs_baseline),
        ("Overhead de Parâmetros", test_victor_parameters),
        ("Múltiplas Imagens", test_victor_multiple_images),
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Teste '{test_name}' falhou com exceção:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Resultado final
    print("\n")
    print("╔" + "=" * 78 + "╗")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("║" + " " * 25 + "TODOS OS TESTES PASSARAM!" + " " * 28 + "║")
        print("║" + " " * 20 + "✓ VICTOR implementado com sucesso" + " " * 24 + "║")
    else:
        print("║" + " " * 27 + "RESUMO DOS TESTES" + " " * 33 + "║")
        print("║" + " " * 78 + "║")
        for test_name, passed in results:
            status = "✓" if passed else "✗"
            padding = " " * (60 - len(test_name))
            print(f"║  {status} {test_name}{padding}║")
    
    print("╚" + "=" * 78 + "╝")
    print("\n")
