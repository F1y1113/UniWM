import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from .wrapped_visualizer import AnoleforConditionalGeneration
from .custom_chameleon import ChameleonAttention, ChameleonDecoderLayer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

import logging

# Import repeat_kv function
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CustomAnoleAttention(ChameleonAttention):
    """
    Custom attention layer that supports memory bank functionality.
    Based on CustomQwen2VLSdpaAttention from CMMCoT project.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        
        # memory bank (intra-step)
        self.stored_keys = None
        self.stored_values = None
        self.use_memory_bank = False
        self.memory_bank_initialized = False
        
        # memory bank (cross-step)
        self.global_stored_keys = []  # List of tensors from different steps
        self.global_stored_values = []  # List of tensors from different steps
        self.global_step_timestamps = []  # Track when each memory was stored
        self.use_global_memory_bank = False
        
        # Special token IDs for triggering memory bank
        self.special_token_start = 8197
        self.special_token_end = 8196
        
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def reset_memory_bank(self):
        """Reset intra-step memory bank state (but keep cross-step memory bank)"""
        self.stored_keys = None
        self.stored_values = None
        self.memory_bank_initialized = False
    
    def reset_global_memory_bank(self):
        """Reset cross-step memory bank state (only at trajectory start)"""
        self.global_stored_keys = []
        self.global_stored_values = []
        self.global_step_timestamps = []
        self.use_global_memory_bank = False
        
    def store_to_global_memory_bank(self, current_step: int):
        """Store current intra-step K,V to global cross-step memory bank"""
        if self.stored_keys is not None and self.stored_values is not None:
            # Clone and store the current intra-step memory

            self.global_stored_keys.append(self.stored_keys.clone())
            self.global_stored_values.append(self.stored_values.clone())
            self.global_step_timestamps.append(current_step)
            self.use_global_memory_bank = True

    
    def compute_similarity_and_select_topk(self, current_keys: torch.Tensor, k: int = 3):
        """Compute cosine similarity and select top-k most similar historical entries"""
        if not self.global_stored_keys or current_keys is None:
            return [], []
        
        # Flatten current keys for similarity computation
        # Shape: [batch_size, num_heads, seq_len, head_dim] -> [batch_size * num_heads * seq_len, head_dim]
        current_flat = current_keys.flatten(start_dim=1)  # [batch, features]

        similarities = []
        for hist_keys in self.global_stored_keys:

            hist_flat = hist_keys.flatten(start_dim=1)  # [batch, features]

            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(current_flat, hist_flat, dim=1)
            similarities.append(cos_sim.mean().item())  # Average across batch
        
        # Select top-k most similar entries
        num_entries = len(similarities)
        k = min(k, num_entries)  # Handle case where we have fewer than k entries
        
        if k == 0:
            return [], []
        
        # Get indices of top-k similarities
        top_indices = torch.topk(torch.tensor(similarities), k=k).indices.tolist()
        
        return top_indices, [similarities[i] for i in top_indices]
    
    def compute_temporal_decay_weights(self, current_step: int, selected_indices: list, gamma: float = 0.3):
        """Compute exponential temporal decay weights for selected historical entries"""
        if not selected_indices or not self.global_step_timestamps:
            return []
        
        # Compute recency gaps and exponential weights
        exp_weights = []
        for idx in selected_indices:
            if idx < len(self.global_step_timestamps):
                delta_t = current_step - self.global_step_timestamps[idx]
                exp_weight = torch.exp(torch.tensor(-gamma * delta_t)).item()
                exp_weights.append(exp_weight)
            else:
                exp_weights.append(0.0)
        
        # Normalize weights
        total_weight = sum(exp_weights)
        if total_weight > 0:
            exp_weights = [w / total_weight * 0.1 for w in exp_weights]
        
        return exp_weights
        
    def extract_special_token_positions(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Extract positions of special token pairs (8197, 8196) from input_ids.
        Returns list of (start_pos, end_pos) tuples for each pair.
        """
        pairs = []
        
        # Handle None or invalid input_ids
        if input_ids is None:
            return pairs
        
        # Convert to list safely
        try:
            if isinstance(input_ids, torch.Tensor):
                # Handle tensor conversion more carefully
                if input_ids.numel() == 0:
                    return pairs
                input_ids_list = input_ids.squeeze().tolist() if input_ids.dim() > 1 else input_ids.tolist()
                # Ensure we have a list, not a single int
                if isinstance(input_ids_list, int):
                    input_ids_list = [input_ids_list]
            elif isinstance(input_ids, (list, tuple)):
                input_ids_list = list(input_ids)
            elif isinstance(input_ids, int):
                input_ids_list = [input_ids]
            else:
                print(f"Layer {getattr(self, 'layer_idx', 'unknown')}: Warning - input_ids type {type(input_ids)} not supported")
                return pairs
                
            # Final safety check
            if not isinstance(input_ids_list, (list, tuple)):
                print(f"Layer {getattr(self, 'layer_idx', 'unknown')}: Error - input_ids_list is still not a list: {type(input_ids_list)}")
                return pairs
                
        except Exception as e:
            print(f"Layer {getattr(self, 'layer_idx', 'unknown')}: Error converting input_ids to list: {e}")
            return pairs
        
        i = 0
        while i < len(input_ids_list):
            if input_ids_list[i] == self.special_token_start:
                # Find corresponding end token
                for j in range(i + 1, len(input_ids_list)):
                    if input_ids_list[j] == self.special_token_end:
                        pairs.append((i, j))
                        i = j
                        break
                else:
                    break
            i += 1
        
        return pairs
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        use_memory_bank: bool = False,
        is_memory_bank_init: bool = False,
        current_step: Optional[int] = None,
        current_substep: Optional[str] = None,
        use_global_memory_bank: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # Call parent forward to get standard attention computation
        attn_output, attn_weights, present_key_value = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        
        if not use_memory_bank:
            return attn_output, attn_weights, present_key_value

        bsz, q_len, _ = hidden_states.size()
        
        # Compute Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings if needed
        if hasattr(self, 'rotary_emb') and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            # Apply rotary position embeddings
            def apply_rotary_pos_emb(q, k, cos, sin):
                # This is a simplified version - you may need to adjust based on the actual implementation
                q_embed = (q * cos) + (self._rotate_half(q) * sin)
                k_embed = (k * cos) + (self._rotate_half(k) * sin)
                return q_embed, k_embed
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Ensure data type consistency before any operations
        target_dtype = self.q_proj.weight.dtype
        if query_states.dtype != target_dtype:
            query_states = query_states.to(target_dtype)
        if key_states.dtype != target_dtype:
            key_states = key_states.to(target_dtype)
        if value_states.dtype != target_dtype:
            value_states = value_states.to(target_dtype)
        
        # Handle memory bank initialization phase
        if is_memory_bank_init and input_ids is not None:
            # print(f"Layer {self.layer_idx}: *** MEMORY BANK INITIALIZATION PHASE ***")
            # Extract special token pairs first
            token_pairs = self.extract_special_token_positions(input_ids)
            
            start_pos, end_pos = None, None
            
            if len(token_pairs) >= 3:  # Use special tokens if available
                start_pos, end_pos = token_pairs[2]  # Third pair (0-indexed)
                print(f"Layer {self.layer_idx}: Using special token positions {start_pos}:{end_pos+1}")
            
            # Extract K, V for the determined token range
            # print(f"Layer {self.layer_idx}: Checking token positions - start_pos: {start_pos}, end_pos: {end_pos}, q_len: {q_len}")
            if start_pos is not None and end_pos is not None and start_pos < q_len and end_pos < q_len and start_pos < end_pos:
                # Store keys and values for the selected tokens
                self.stored_keys = key_states[:, :, start_pos:end_pos+1, :].clone().to(target_dtype)
                self.stored_values = value_states[:, :, start_pos:end_pos+1, :].clone().to(target_dtype)
                self.memory_bank_initialized = True
                
                print(f"Layer {self.layer_idx}: ✅ Memory bank initialized with K,V from positions {start_pos}:{end_pos+1}")
  
            else:
                self.memory_bank_initialized = False
                print(f"Layer {self.layer_idx}: memory_bank_initialized set to: {self.memory_bank_initialized}")
        
        # Perform standard attention computation
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} if hasattr(self, 'rotary_emb') else {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        # Handle memory bank usage phase - modify query computation for specific positions
        if self.memory_bank_initialized and self.stored_keys is not None and input_ids is not None and not is_memory_bank_init:
            token_pairs = self.extract_special_token_positions(input_ids)
            # print(f"Layer {self.layer_idx}: Found {len(token_pairs)} token pairs in current input")
            if len(token_pairs) >= 3:  # Use the third pair
                start_pos, end_pos = token_pairs[2]
                print(f"Layer {self.layer_idx}: Using token positions {start_pos}:{end_pos+1} for memory bank enhancement")
                
                if start_pos < q_len and end_pos < q_len and start_pos < end_pos:
                    # Extract query states for the image tokens
                    image_query_states = query_states[:, :, start_pos:end_pos+1, :]
                    print(f"Layer {self.layer_idx}: Extracted query states shape: {image_query_states.shape}, dtype: {image_query_states.dtype}")
                    
                    # Prepare current intra-step K,V
                    current_stored_keys = self.stored_keys
                    current_stored_values = self.stored_values

                    # Merge with global memory bank if available and enabled
                    if use_global_memory_bank and self.use_global_memory_bank and current_step is not None:
                        # print(f"Layer {self.layer_idx}: Merging with global memory bank (step {current_step})")
                        # print(f"Layer {self.layer_idx}: Global memory bank has {len(self.global_stored_keys)} historical steps")
                        
                        # Step 1: Similarity gating - select top-k most similar entries
                        selected_indices, similarities = self.compute_similarity_and_select_topk(current_stored_keys, k=3)
                        # print(f"Layer {self.layer_idx}: Selected {len(selected_indices)} most similar entries: {selected_indices}")
                        # print(f"Layer {self.layer_idx}: Similarity scores: {[f'{s:.4f}' for s in similarities]}")
                        
                        if selected_indices:
                            # Step 2: Temporal decay - compute exponential decay weights for selected entries
                            weights = self.compute_temporal_decay_weights(current_step, selected_indices, gamma=0.3)
                            print(f"Layer {self.layer_idx}: Temporal decay weights: {[f'{w:.4f}' for w in weights]}")
                            
                            # Prepare list of selected K,V tensors (historical + current)
                            all_keys = []
                            all_values = []
                            
                            # Add weighted selected historical memories
                            for i, (idx, weight) in enumerate(zip(selected_indices, weights)):
                                if idx < len(self.global_stored_keys):
                                    hist_k = self.global_stored_keys[idx]
                                    hist_v = self.global_stored_values[idx]
                                    # Apply temporal weight
                                    weighted_k = hist_k * weight
                                    weighted_v = hist_v * weight
                                    all_keys.append(weighted_k)
                                    all_values.append(weighted_v)
                                    step_timestamp = self.global_step_timestamps[idx] if idx < len(self.global_step_timestamps) else 'unknown'
                                    print(f"Layer {self.layer_idx}: Added selected historical step {step_timestamp} (idx={idx}) with weight {weight:.4f}, similarity {similarities[i]:.4f}")
                            
                            # Add current memory with highest weight (1.0)
                            all_keys.append(current_stored_keys * 0.1)
                            all_values.append(current_stored_values * 0.1)
                            print(f"Layer {self.layer_idx}: Added current step {current_step} with weight 1.0")
                            
                            # Concatenate selected memories along sequence dimension
                            merged_keys = torch.cat(all_keys, dim=2)  # Concat along seq_len dimension
                            merged_values = torch.cat(all_values, dim=2)
                            
                            print(f"Layer {self.layer_idx}: Merged keys shape: {merged_keys.shape}")
                            print(f"Layer {self.layer_idx}: Merged values shape: {merged_values.shape}")
                            
                            current_stored_keys = merged_keys
                            current_stored_values = merged_values
                        else:
                            print(f"Layer {self.layer_idx}: No similar historical entries found, using only current memory")
                    else:
                        print(f"Layer {self.layer_idx}: Using only current intra-step memory bank")
                    
                    # Repeat stored K,V if needed for multi-head attention
                    if self.num_key_value_heads != self.num_heads:
                        stored_keys = current_stored_keys.repeat_interleave(
                            self.num_heads // self.num_key_value_heads, dim=1
                        )
                        stored_values = current_stored_values.repeat_interleave(
                            self.num_heads // self.num_key_value_heads, dim=1
                        )
                        print(f"Layer {self.layer_idx}: Repeated stored K,V for multi-head attention")
                    else:
                        stored_keys = current_stored_keys
                        stored_values = current_stored_values
                    
                    print(f"Layer {self.layer_idx}: Final stored_keys shape: {stored_keys.shape}, dtype: {stored_keys.dtype}")
                    print(f"Layer {self.layer_idx}: Final stored_values shape: {stored_values.shape}, dtype: {stored_values.dtype}")
                    
                    # Compute new attention for image tokens using stored K,V
                    new_image_attn = F.scaled_dot_product_attention(
                        image_query_states,
                        stored_keys,
                        stored_values,
                        attn_mask=None,
                        dropout_p=0.0 if not self.training else 0.1
                    )
                    
                    print(f"Layer {self.layer_idx}: Computed new attention shape: {new_image_attn.shape}, dtype: {new_image_attn.dtype}")
                    
                    # Calculate attention statistics for verification
                    attn_mean = new_image_attn.mean().item()
                    attn_std = new_image_attn.std().item()
                    attn_max = new_image_attn.max().item()
                    attn_min = new_image_attn.min().item()
                                        
                    # Replace the query states with the new attention results
                    query_states[:, :, start_pos:end_pos+1, :] = new_image_attn
                    
            # else:
            #     print(f"Layer {self.layer_idx}: ❌ Insufficient token pairs for memory bank usage")
        
        # Compute final attention output
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        # Reshape and project attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, present_key_value


class CustomAnoleDecoderLayer(ChameleonDecoderLayer):
    """
    Custom decoder layer that uses CustomAnoleAttention for specific layers.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        
        # Replace self_attn with custom attention for specific layers
        self.use_memory_bank_layers = {7, 13 ,19, 25, 31}
        if layer_idx in self.use_memory_bank_layers:
            self.self_attn = CustomAnoleAttention(config, layer_idx)
    
    def _get_root_model(self):
        """Get the root MemoryBankAnoleForConditionalGeneration instance."""
        # First check if we have a weak reference
        if hasattr(self, '_root_model_ref') and self._root_model_ref is not None:
            root_model = self._root_model_ref()
            if root_model is not None:
                return root_model
        
        # Alternative: use a global registry approach
        # Search for the root model in the current module tree without recursion
        import sys
        current_module = sys.modules.get(self.__class__.__module__)
        if current_module and hasattr(current_module, '_current_memory_bank_model'):
            return current_module._current_memory_bank_model
        
        # Fallback: return a dummy object that provides default values
        class DummyRootModel:
            def __init__(self):
                self._temp_use_memory_bank = False
                self._temp_is_memory_bank_init = False
                self._temp_input_ids = None
                self._temp_current_step = None
                self._temp_current_substep = None
                self._temp_use_global_memory_bank = False
                self._generation_use_memory_bank = False
                self._generation_is_memory_bank_init = False
                self._generation_current_step = None
                self._generation_current_substep = None
                self._generation_use_global_memory_bank = False
        
        print(f"Warning: Could not find root model for layer {getattr(self, 'layer_idx', 'unknown')}, using dummy")
        return DummyRootModel()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        use_memory_bank: bool = False,
        is_memory_bank_init: bool = False,
        current_step: Optional[int] = None,
        current_substep: Optional[str] = None,
        use_global_memory_bank: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        if isinstance(self.self_attn, CustomAnoleAttention):
            # Get memory bank parameters from the root model instance
            root_model = self._get_root_model()
            # Prioritize generation-level parameters over temporary parameters
            use_memory_bank_param = getattr(root_model, '_generation_use_memory_bank', 
                                           getattr(root_model, '_temp_use_memory_bank', use_memory_bank))
            is_memory_bank_init_param = getattr(root_model, '_generation_is_memory_bank_init', 
                                               getattr(root_model, '_temp_is_memory_bank_init', is_memory_bank_init))
            input_ids_param = getattr(root_model, '_temp_input_ids', input_ids)
            current_step_param = getattr(root_model, '_generation_current_step', 
                                        getattr(root_model, '_temp_current_step', current_step))
            current_substep_param = getattr(root_model, '_generation_current_substep', 
                                           getattr(root_model, '_temp_current_substep', current_substep))
            use_global_memory_bank_param = getattr(root_model, '_generation_use_global_memory_bank', 
                                                   getattr(root_model, '_temp_use_global_memory_bank', use_global_memory_bank))
            
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                input_ids=input_ids_param,
                use_memory_bank=use_memory_bank_param,
                is_memory_bank_init=is_memory_bank_init_param,
                current_step=current_step_param,
                current_substep=current_substep_param,
                use_global_memory_bank=use_global_memory_bank_param,
                **kwargs
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )
        
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class MemoryBankAnoleForConditionalGeneration(AnoleforConditionalGeneration):
    """
    Anole model with memory bank functionality.
    Inherits from AnoleforConditionalGeneration and adds memory bank support.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        # Memory bank configuration
        self.use_memory_bank_layers = {7, 13 ,19, 25, 31}
        self.memory_bank_initialized = False
        self.special_token_start = 8197
        self.special_token_end = 8196
        
        # Replace decoder layers with custom layers for memory bank support
        self._replace_decoder_layers()
        
        # Set root model reference for all custom layers
        self._set_root_model_references()
        
        # Set global reference for layers to find
        import sys
        current_module = sys.modules.get(self.__class__.__module__)
        if current_module:
            current_module._current_memory_bank_model = self
        
    def _replace_decoder_layers(self):
        """Replace specific decoder layers with custom layers that support memory bank."""
        # Try different model structure paths
        layers = None
        if hasattr(self, 'model') and hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self, 'model') and hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self, 'layers'):
            layers = self.layers
        
        if layers is not None:
            for layer_idx in range(len(layers)):
                if layer_idx in self.use_memory_bank_layers:
                    # Replace with custom layer
                    config = layers[layer_idx].config if hasattr(layers[layer_idx], 'config') else self.config
                    layers[layer_idx] = CustomAnoleDecoderLayer(config, layer_idx)
                    print(f"Replaced layer {layer_idx} with CustomAnoleDecoderLayer")
        else:
            print("Warning: Could not find model layers to replace")
    
    def _set_root_model_references(self):
        """Set root model reference for all custom layers."""
        layers = None
        if hasattr(self, 'model') and hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self, 'model') and hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self, 'layers'):
            layers = self.layers
        
        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                if isinstance(layer, CustomAnoleDecoderLayer):
                    # Use weak reference to avoid circular reference issues
                    import weakref
                    layer._root_model_ref = weakref.ref(self)
                    print(f"Set root model reference for layer {layer_idx}")
    
    def reset_memory_bank(self):
        """Reset intra-step memory bank state for all layers (but keep cross-step memory bank)."""
        self.memory_bank_initialized = False
        
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, CustomAnoleAttention):
                    layer.self_attn.reset_memory_bank()
        
        print("Intra-step memory bank reset for all layers", flush=True)
    
    def reset_global_memory_bank(self):
        """Reset cross-step memory bank state for all layers (only at trajectory start)."""
        
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, CustomAnoleAttention):
                    layer.self_attn.reset_global_memory_bank()
        
        print("Cross-step memory bank reset for all layers")
    
    def store_to_global_memory_bank(self, current_step: int):
        """Store current intra-step K,V to global cross-step memory bank for all layers."""
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, CustomAnoleAttention):
                    layer.self_attn.store_to_global_memory_bank(current_step)
        
        print(f"Stored step {current_step} to global memory bank for all layers")
    
    def enable_global_memory_bank(self):
        """Enable global memory bank functionality for all layers."""
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, CustomAnoleAttention):
                    layer.self_attn.use_global_memory_bank = True
        
        print("Global memory bank enabled for all custom attention layers")
    
    def enable_memory_bank(self):
        """Enable memory bank functionality."""
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, CustomAnoleAttention):
                    layer.self_attn.use_memory_bank = True
        print("Memory bank enabled for all custom attention layers")
    
    def initialize_memory_bank(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, attention_mask: torch.Tensor):
        """
        Initialize memory bank by extracting K,V from special tokens or image tokens.
        
        Args:
            input_ids: Input token IDs containing special tokens or image tokens
            pixel_values: Pixel values for images
            attention_mask: Attention mask
        """
        print(f"Initializing memory bank with input shapes:")
        print(f"  Input IDs: {input_ids.shape}")
        print(f"  Pixel values: {pixel_values.shape}")
        print(f"  Attention mask: {attention_mask.shape}")
        
        # Check for special token pairs first
        input_ids_list = input_ids.squeeze().tolist() if input_ids.dim() > 1 else input_ids.tolist()
        pairs = []
        i = 0
        while i < len(input_ids_list):
            if input_ids_list[i] == self.special_token_start:
                for j in range(i + 1, len(input_ids_list)):
                    if input_ids_list[j] == self.special_token_end:
                        pairs.append((i, j))
                        i = j
                        break
                else:
                    break
            i += 1
        
        use_special_tokens = len(pairs) >= 3
        
        if use_special_tokens:
            print(f"Found {len(pairs)} special token pairs: {pairs}")
            print(f"Using third pair: {pairs[2]}")
        else:
            print(f"Warning: Insufficient tokens for memory bank initialization")
            return False
        
        # Ensure proper data types for initialization
        target_dtype = self.config.torch_dtype if hasattr(self.config, 'torch_dtype') else torch.bfloat16
        
        # Convert pixel_values to the correct dtype
        if pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(target_dtype)
            print(f"Converted pixel_values from {pixel_values.dtype} to {target_dtype}")
        
        # Prepare inputs for forward pass (without memory bank parameters for base model)
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }
        
        # Set memory bank parameters as instance variables for layers to access
        self._temp_use_memory_bank = True
        self._temp_is_memory_bank_init = True
        self._temp_input_ids = input_ids
        
        # Run forward pass to initialize memory bank
        with torch.no_grad():
            try:
                outputs = self.model(**model_inputs)
                self.memory_bank_initialized = True
                if hasattr(self.model, 'layers'):
                    for layer in self.model.layers:
                        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, CustomAnoleAttention):
                            print(f"Layer {layer.self_attn.layer_idx}: memory_bank_initialized = {layer.self_attn.memory_bank_initialized}")
                            # print(f"Layer {layer.self_attn.layer_idx}: stored_keys = {layer.self_attn.stored_keys}")
                            if layer.self_attn.stored_keys is not None:
                                print(f"layer stored keys shape: {layer.self_attn.stored_keys.shape}")
                            else:
                                print(f"Layer {layer.self_attn.layer_idx}: stored_keys is None - initialization failed")
                return True
            except Exception as e:
                print(f"Memory bank initialization failed: {e}")
                print(f"Input dtypes - input_ids: {input_ids.dtype}, attention_mask: {attention_mask.dtype}, pixel_values: {pixel_values.dtype}")
                return False
            finally:
                # Clean up temporary variables
                self._temp_use_memory_bank = False
                self._temp_is_memory_bank_init = False
                self._temp_input_ids = None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        use_memory_bank: bool = False,
        is_memory_bank_init: bool = False,
        current_step: Optional[int] = None,
        current_substep: Optional[str] = None,
        use_global_memory_bank: bool = False,
        **kwargs
    ):
        # Store memory bank parameters as temporary instance variables
        self._temp_use_memory_bank = use_memory_bank
        self._temp_is_memory_bank_init = is_memory_bank_init
        self._temp_input_ids = input_ids
        self._temp_current_step = current_step
        self._temp_current_substep = current_substep
        self._temp_use_global_memory_bank = use_global_memory_bank
        
        # Filter out memory bank specific parameters before calling parent
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['use_memory_bank', 'is_memory_bank_init', 'current_step', 'current_substep', 'use_global_memory_bank']}
        
        # Don't clean up temporary variables during generation - they need to persist
        # across multiple forward calls during the generation process
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            pixel_values=pixel_values,
            **filtered_kwargs
         )
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_memory_bank: bool = False,
        is_memory_bank_init: bool = False,
        current_step: Optional[int] = None,
        current_substep: Optional[str] = None,
        use_global_memory_bank: bool = False,
        **kwargs,
    ):
        """Override generate to support memory bank parameters."""
        
        # Store memory bank parameters for the entire generation process
        self._generation_use_memory_bank = use_memory_bank
        self._generation_is_memory_bank_init = is_memory_bank_init
        self._generation_current_step = current_step
        self._generation_current_substep = current_substep
        self._generation_use_global_memory_bank = use_global_memory_bank
        
        # Add memory bank parameters to generation kwargs
        if use_memory_bank:
            kwargs.update({
                'use_memory_bank': use_memory_bank,
                'is_memory_bank_init': is_memory_bank_init,
                'current_step': current_step,
                'current_substep': current_substep,
                'use_global_memory_bank': use_global_memory_bank
            })
            
            if current_step is not None:
                print(f"Step {current_step} ({current_substep}): Generating with memory bank")
                print(f"  Memory bank init: {is_memory_bank_init}")
                print(f"  Global memory bank: {use_global_memory_bank}")
                print(f"  Memory bank initialized: {self.memory_bank_initialized}")
        
        try:
            return super().generate(
                inputs=inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                **kwargs
            )
        finally:
            # Clean up generation-level parameters
            self._generation_use_memory_bank = False
            self._generation_is_memory_bank_init = False
            self._generation_current_step = None
            self._generation_current_substep = None
            self._generation_use_global_memory_bank = False
