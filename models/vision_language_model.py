import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector, VisualRegisters
from models.config import VLMConfig

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        
        # VICTOR: Visual Compact Token Registers
        if cfg.use_victor:
            self.visual_registers = VisualRegisters(cfg)
            print(f"VICTOR ativado: {cfg.victor_num_registers} registros, descarte na camada {cfg.victor_drop_layer}")
        else:
            self.visual_registers = None
            
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

    def _adjust_targets_for_victor(self, targets, victor_info, input_ids, logits_shape):
        """
        Ajusta os targets para corresponder à redução de tokens pelo VICTOR.
        Remove as posições dos tokens visuais originais, mantendo os registros.
        Usa o shape dos logits como referência para garantir sincronização perfeita.
        """
        num_original_visual_tokens = victor_info['num_original_visual_tokens']
        num_registers = victor_info['num_registers']
        
        B, T = targets.size()
        # Pegar o comprimento exato dos logits após drop + padding
        target_seq_len = logits_shape[1]
        
        image_token_mask = (input_ids == self.tokenizer.image_token_id)
        
        new_targets_list = []
        
        for b in range(B):
            num_placeholders = image_token_mask[b].sum().item()
            num_images_in_sample = num_placeholders // num_original_visual_tokens if num_placeholders > 0 else 0
            
            if num_images_in_sample == 0:
                # Sem imagens - apenas copiar e fazer padding para target_seq_len
                targets_b = targets[b]
                if targets_b.size(0) < target_seq_len:
                    pad_len = target_seq_len - targets_b.size(0)
                    targets_b = torch.cat([targets_b, torch.full((pad_len,), -100, dtype=targets.dtype, device=targets.device)], dim=0)
                elif targets_b.size(0) > target_seq_len:
                    targets_b = targets_b[:target_seq_len]
                new_targets_list.append(targets_b)
                continue
            
            # Comprimento após inserir registros (antes do drop)
            seq_len_with_registers = T + num_images_in_sample * num_registers
            
            # Expandir targets para incluir posições de registros
            expanded_targets = torch.full((seq_len_with_registers,), -100, 
                                         dtype=targets.dtype, device=targets.device)
            
            # Copiar targets originais para posições corretas
            target_idx = 0
            expanded_idx = 0
            for img_idx in range(num_images_in_sample):
                # Tokens visuais
                visual_len = num_original_visual_tokens
                if target_idx + visual_len <= T and expanded_idx + visual_len <= seq_len_with_registers:
                    expanded_targets[expanded_idx:expanded_idx+visual_len] = targets[b, target_idx:target_idx+visual_len]
                expanded_idx += visual_len
                target_idx += visual_len
                
                # Registros (sempre -100, não computam loss)
                expanded_idx += num_registers
            
            # Restante da sequência
            if target_idx < T and expanded_idx < seq_len_with_registers:
                remaining = min(T - target_idx, seq_len_with_registers - expanded_idx)
                expanded_targets[expanded_idx:expanded_idx+remaining] = targets[b, target_idx:target_idx+remaining]
            
            # Criar máscara para remover tokens visuais (igual ao decoder)
            keep_mask = torch.ones(seq_len_with_registers, dtype=torch.bool, device=targets.device)
            current_pos = 0
            for _ in range(num_images_in_sample):
                visual_start = current_pos
                visual_end = current_pos + num_original_visual_tokens
                if visual_end > seq_len_with_registers:
                    break
                keep_mask[visual_start:visual_end] = False
                current_pos += num_original_visual_tokens + num_registers
            
            # Aplicar máscara
            targets_after_drop = expanded_targets[keep_mask]
            
            # Fazer padding/truncate para target_seq_len (igual ao decoder)
            if targets_after_drop.size(0) < target_seq_len:
                pad_len = target_seq_len - targets_after_drop.size(0)
                targets_after_drop = torch.cat([targets_after_drop, torch.full((pad_len,), -100, dtype=targets.dtype, device=targets.device)], dim=0)
            elif targets_after_drop.size(0) > target_seq_len:
                targets_after_drop = targets_after_drop[:target_seq_len]
            
            new_targets_list.append(targets_after_drop)
        
        # Stack final
        padded_targets = torch.stack(new_targets_list, dim=0)
        
        return padded_targets
    
    def _insert_visual_registers(self, token_embd, attention_mask, input_ids, num_visual_tokens):
        """
        Insere registros visuais após cada bloco de tokens visuais na sequência.
        
        Args:
            token_embd: Tensor de embeddings [B, T, D]
            attention_mask: Máscara de atenção [B, T]
            input_ids: IDs de input originais [B, T_original]
            num_visual_tokens: Número de tokens visuais por imagem
            
        Returns:
            Tuple de (token_embd_com_registros, attention_mask_atualizada)
        """
        B, T, D = token_embd.size()
        
        # Identificar onde estavam os placeholders de imagem nos input_ids originais
        image_token_mask = (input_ids == self.tokenizer.image_token_id)  # [B, T_original]
        
        # Para cada exemplo no batch, inserir registros
        new_embd_list = []
        new_mask_list = []
        
        for b in range(B):
            # Contar quantas imagens temos
            # Total de placeholders / placeholders_por_imagem = número de imagens
            num_placeholders = image_token_mask[b].sum().item()
            num_images = num_placeholders // num_visual_tokens if num_placeholders > 0 else 0
            
            if num_images == 0:
                # Sem imagens neste exemplo
                new_embd_list.append(token_embd[b])
                if attention_mask is not None:
                    new_mask_list.append(attention_mask[b])
                continue
            
            # Obter registros para este exemplo
            registers = self.visual_registers(num_images)  # [num_images, num_registers, D]
            
            # Inserir registros após cada bloco de tokens visuais
            parts = []
            mask_parts = []
            current_pos = 0
            
            for img_idx in range(num_images):
                # Adicionar tokens visuais
                visual_start = current_pos
                visual_end = current_pos + num_visual_tokens
                parts.append(token_embd[b, visual_start:visual_end])
                
                # Adicionar registros
                parts.append(registers[img_idx])
                
                # Atualizar attention mask
                if attention_mask is not None:
                    mask_parts.append(attention_mask[b, visual_start:visual_end])
                    mask_parts.append(torch.ones(self.cfg.victor_num_registers, device=attention_mask.device))
                
                current_pos = visual_end
            
            # Adicionar resto da sequência (texto)
            if current_pos < T:
                parts.append(token_embd[b, current_pos:])
                if attention_mask is not None:
                    mask_parts.append(attention_mask[b, current_pos:])
            
            # Concatenar
            new_embd_list.append(torch.cat(parts, dim=0))
            if attention_mask is not None:
                new_mask_list.append(torch.cat(mask_parts, dim=0))
        
        # Fazer padding para mesmo tamanho antes de empilhar
        max_len = max(emb.size(0) for emb in new_embd_list)
        
        # Armazenar comprimentos individuais antes do padding para ajuste de targets
        seq_lengths = [emb.size(0) for emb in new_embd_list]
        
        padded_embd_list = []
        padded_mask_list = []
        
        for b in range(B):
            emb = new_embd_list[b]
            current_len = emb.size(0)
            
            if current_len < max_len:
                # Pad embeddings com zeros
                pad_len = max_len - current_len
                padding = torch.zeros(pad_len, D, device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, padding], dim=0)
            
            padded_embd_list.append(emb)
            
            if attention_mask is not None:
                mask = new_mask_list[b]
                if mask.size(0) < max_len:
                    # Pad mask com zeros (tokens ignorados)
                    pad_len = max_len - mask.size(0)
                    mask_padding = torch.zeros(pad_len, device=mask.device, dtype=mask.dtype)
                    mask = torch.cat([mask, mask_padding], dim=0)
                padded_mask_list.append(mask)
        
        # Stack de volta
        token_embd_new = torch.stack(padded_embd_list, dim=0)
        
        if attention_mask is not None:
            attention_mask_new = torch.stack(padded_mask_list, dim=0)
        else:
            attention_mask_new = None
        
        return token_embd_new, attention_mask_new, seq_lengths
    
    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = (input_ids == self.tokenizer.image_token_id)
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype) # torch flattens before assigning

        return updated_token_embd

    def _process_images(self, images, device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]

            if not images:  # Handle cases with no images
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images # Already a tensor

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids) # [B, T_sequence, D_lm]

        # VICTOR: informações sobre tokens visuais
        victor_info = None
        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)  # [num_images, mp_image_token_length, D_lm]
            
            # Substituir placeholders com embeddings visuais primeiro
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
            
            # VICTOR: adicionar registros APÓS substituir placeholders
            if self.cfg.use_victor:
                num_original_visual_tokens = image_embd.size(1)
                
                # Número total de imagens = total de embeddings visuais processados
                # Dividido por batch para obter imagens por amostra (assumindo uniforme)
                total_images = images_tensor.size(0)
                num_images_per_sample = total_images // input_ids.size(0)
                
                # Inserir registros após cada bloco de tokens visuais
                token_embd, attention_mask, seq_lengths = self._insert_visual_registers(
                    token_embd, attention_mask, input_ids, num_original_visual_tokens
                )
                
                victor_info = {
                    'drop_layer': self.cfg.victor_drop_layer,
                    'num_original_visual_tokens': num_original_visual_tokens,
                    'num_registers': self.cfg.victor_num_registers,
                    'num_images': num_images_per_sample,
                    'seq_lengths_before_drop': seq_lengths,  # Comprimentos antes de remover tokens visuais
                }

        logits, _ = self.decoder(token_embd, attention_mask=attention_mask, victor_info=victor_info)

        loss = None
        if targets is not None:
            # VICTOR: ajustar targets para corresponder à redução de tokens
            # Usa o shape dos logits como referência para garantir sincronização perfeita
            if self.cfg.use_victor and victor_info is not None:
                targets = self._adjust_targets_for_victor(targets, victor_info, input_ids, logits.shape)
            
            logits = self.decoder.head(logits) # Apply LM head
            
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids) # [B, T_prompt_text, D_lm]

        # VICTOR: informações sobre tokens visuais
        victor_info = None
        if images_tensor is not None:
            # 1. Process image if present
            image_embd = self.vision_encoder(images_tensor) # [B, T_img_feat, D_model]
            image_embd = self.MP(image_embd)      # [B, mp_image_token_length, D_lm]
            
            # 2. Combine image and text embeddings
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
            
            # VICTOR: adicionar registros APÓS substituir placeholders
            if self.cfg.use_victor:
                num_original_visual_tokens = image_embd.size(1)
                
                # Número total de imagens = total de embeddings visuais processados
                # Dividido por batch para obter imagens por amostra (assumindo uniforme)
                total_images = images_tensor.size(0)
                num_images_per_sample = total_images // input_ids.size(0)
                
                token_embd, attention_mask, seq_lengths = self._insert_visual_registers(
                    token_embd, attention_mask, input_ids, num_original_visual_tokens
                )
                
                victor_info = {
                    'drop_layer': self.cfg.victor_drop_layer,
                    'num_original_visual_tokens': num_original_visual_tokens,
                    'num_registers': self.cfg.victor_num_registers,
                    'num_images': num_images_per_sample,
                    'seq_lengths_before_drop': seq_lengths,
                }

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0) # Or token_embd.size(0)
        
        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            token_embd,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0,
            victor_info=victor_info
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] 
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(next_token_id) # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            # With KV cache: only process the new token
            # VICTOR: não precisa passar victor_info na fase de decodificação (tokens visuais já foram descartados)
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
                victor_info=None  # Tokens visuais já foram processados no prefill
            )
      
            last_token_output = decode_step_output[:, -1, :] 
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id) # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(seq_len, device=device) # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
