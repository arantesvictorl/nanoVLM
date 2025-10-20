import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
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
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

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
        token_embd = self.decoder.token_embedding(input_ids)
        B = token_embd.size(0)
        
        if images_tensor is not None and self.cfg.use_victor:
            image_embd = self.vision_encoder(images_tensor)
            v_proj, registers = self.MP(image_embd)
            N_v = v_proj.size(1)
            R = registers.size(0)
            
            registers_expanded = registers.unsqueeze(0).expand(B, -1, -1)
            token_embd_with_img = self._replace_img_tokens_with_embd(input_ids, token_embd, v_proj)
            
            img_mask = (input_ids == self.tokenizer.image_token_id)
            first_img_pos = img_mask.float().argmax(dim=1)
            
            result_embd = []
            for b in range(B):
                pos = first_img_pos[b].item()
                seq = torch.cat([
                    token_embd_with_img[b, :pos],
                    v_proj[b],
                    registers_expanded[b],
                    token_embd_with_img[b, pos+N_v:]
                ], dim=0)
                result_embd.append(seq)
            
            x0 = torch.stack(result_embd, dim=0)
            T_full = x0.size(1)
            N_t = token_embd.size(1) - N_v
            
            if attention_mask is not None:
                attn_mask_full = torch.ones((B, T_full), device=x0.device, dtype=attention_mask.dtype)
                for b in range(B):
                    pos = first_img_pos[b].item()
                    orig_mask = attention_mask[b]
                    attn_mask_full[b, :pos] = orig_mask[:pos]
                    attn_mask_full[b, pos:pos+N_v+R] = 1
                    attn_mask_full[b, pos+N_v+R:] = orig_mask[pos+N_v:]
            else:
                attn_mask_full = None
            
            position_ids_full = torch.arange(T_full, device=x0.device).unsqueeze(0).expand(B, -1)
            cos_full, sin_full = self.decoder.rotary_embd(position_ids_full)
            
            k = self.cfg.k_fuse_layers
            h, kv_cache = self.decoder.forward_blocks(x0, cos_full, sin_full, attn_mask_full, None, 0, k)
            
            h_cut = []
            for b in range(B):
                pos = first_img_pos[b].item()
                h_cut.append(torch.cat([
                    h[b, :pos],
                    h[b, pos+N_v:pos+N_v+R],
                    h[b, pos+N_v+R:]
                ], dim=0))
            h = torch.stack(h_cut, dim=0)
            
            T_short = h.size(1)
            if attention_mask is not None:
                attn_mask_short = torch.ones((B, T_short), device=h.device, dtype=attention_mask.dtype)
                for b in range(B):
                    pos = first_img_pos[b].item()
                    orig_mask = attention_mask[b]
                    attn_mask_short[b, :pos] = orig_mask[:pos]
                    attn_mask_short[b, pos:pos+R] = 1
                    attn_mask_short[b, pos+R:] = orig_mask[pos+N_v:]
            else:
                attn_mask_short = None
            
            position_ids_short = []
            for b in range(B):
                pos = first_img_pos[b].item()
                pos_ids = torch.cat([
                    torch.arange(pos, device=h.device),
                    torch.arange(pos+N_v, pos+N_v+R, device=h.device),
                    torch.arange(pos+N_v+R, position_ids_full.size(1), device=h.device)
                ])
                position_ids_short.append(pos_ids)
            position_ids_short = torch.stack(position_ids_short, dim=0)
            cos_short, sin_short = self.decoder.rotary_embd(position_ids_short)
            
            kv_cache = [None] * len(self.decoder.blocks)
            h, _ = self.decoder.forward_blocks(h, cos_short, sin_short, attn_mask_short, kv_cache, k, None)
            logits = self.decoder.norm(h)
            
            if targets is not None:
                targets_cut = []
                for b in range(B):
                    pos = first_img_pos[b].item()
                    ignore_tokens = torch.full((R,), -100, device=targets.device, dtype=targets.dtype)
                    targets_cut.append(torch.cat([
                        targets[b, :pos],
                        ignore_tokens,
                        targets[b, pos+N_v:]
                    ], dim=0))
                targets = torch.stack(targets_cut, dim=0)
            
        elif images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
            logits, _ = self.decoder(token_embd, attention_mask=attention_mask)
        else:
            logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)
        batch_size = input_ids.size(0)

        if images_tensor is not None and self.cfg.use_victor:
            image_embd = self.vision_encoder(images_tensor)
            _, registers = self.MP(image_embd)
            
            registers_expanded = registers.unsqueeze(0).expand(batch_size, -1, -1)
            
            img_mask = (input_ids == self.tokenizer.image_token_id)
            first_img_pos = img_mask.float().argmax(dim=1)
            
            result_embd = []
            for b in range(batch_size):
                pos = first_img_pos[b].item()
                N_v = (input_ids[b] == self.tokenizer.image_token_id).sum().item()
                seq = torch.cat([
                    token_embd[b, :pos],
                    registers_expanded[b],
                    token_embd[b, pos+N_v:]
                ], dim=0)
                result_embd.append(seq)
            
            token_embd = torch.stack(result_embd, dim=0)
            
            if attention_mask is not None:
                R = registers.size(0)
                attn_mask_new = torch.ones((batch_size, token_embd.size(1)), device=token_embd.device, dtype=attention_mask.dtype)
                for b in range(batch_size):
                    pos = first_img_pos[b].item()
                    N_v = (input_ids[b] == self.tokenizer.image_token_id).sum().item()
                    orig_mask = attention_mask[b]
                    attn_mask_new[b, :pos] = orig_mask[:pos]
                    attn_mask_new[b, pos:pos+R] = 1
                    attn_mask_new[b, pos+R:] = orig_mask[pos+N_v:]
                attention_mask = attn_mask_new
        elif images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        current_total_seq_len = token_embd.size(1)
        
        prefill_output, kv_cache_list = self.decoder(
            token_embd,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0
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
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
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
