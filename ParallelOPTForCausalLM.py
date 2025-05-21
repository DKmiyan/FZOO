# ParallelOPTForCausalLM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTConfig,OPTPreTrainedModel, BaseModelOutputWithPast, OPT_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, _EXPECTED_OUTPUT_SHAPE, _CONFIG_FOR_DOC, OPTConfig, _make_causal_mask, _expand_mask, OPTAttention,OPTLearnedPositionalEmbedding
from transformers.modeling_outputs import CausalLMOutputWithPast,BaseModelOutputWithPast
from transformers.activations import ACT2FN
from typing import Optional, List, Tuple, Union
from transformers.utils.doc import add_start_docstrings_to_model_forward, add_code_sample_docstrings
import math
import pdb

class ParallelOPTAttention(nn.Module):
    def __init__(self,embed_dim: int, num_heads: int,dropout: float = 0.0,is_decoder: bool = False,bias: bool = True,num_parallel: int = 8, freeze: bool = False, epsilon: float = 0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.num_parallel = num_parallel
        self.freeze = freeze
        self.epsilon = epsilon
        self.q_proj = ParallelLinear(self.embed_dim, self.embed_dim, bias=bias,
                                     num_parallel=num_parallel, freeze=freeze, epsilon=epsilon)
        self.k_proj = ParallelLinear(self.embed_dim, self.embed_dim, bias=bias,
                                     num_parallel=num_parallel, freeze=freeze, epsilon=epsilon)
        self.v_proj = ParallelLinear(self.embed_dim, self.embed_dim, bias=bias,
                                     num_parallel=num_parallel, freeze=freeze, epsilon=epsilon)
        self.out_proj = ParallelLinear(self.embed_dim, self.embed_dim, bias=bias,
                                       num_parallel=num_parallel, freeze=freeze, epsilon=epsilon)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [n*B, tgt_len, embed_dim]
        key_value_states: Optional[torch.Tensor] = None,  # for cross-attention, shape: [B, src_len, embed_dim]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,    # [n*B, 1, tgt_len, src_len]
        layer_head_mask: Optional[torch.Tensor] = None,     # [num_heads]
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()      

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            if past_key_value is not None:
                key_states = past_key_value[0]
                value_states = past_key_value[1]
            else:
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz) 
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # src_len = tgt_len
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        


        proj_shape = (bsz * self.num_heads, tgt_len, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # [n*B*num_heads, tgt_len, src_len]
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights have incorrect shape: {attn_weights.size()}")
        
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask should have shape {(bsz, 1, tgt_len, src_len)}, got {attention_mask.size()}")
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights,torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"layer_head_mask should have shape {(self.num_heads,)}, got {layer_head_mask.size()}")
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)  # [n*B*num_heads, tgt_len, head_dim]
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class ParallelOPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, num_parallel: int = 8, freeze: bool = False, epsilon: float = 0.01):

        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_parallel = num_parallel
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.self_attn = ParallelOPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            num_parallel=num_parallel,
            freeze=freeze, 
            epsilon=epsilon,
        )
        self.self_attn_layer_norm = ParallelLayerNorm(
            config.hidden_size, eps=1e-5, elementwise_affine=config.layer_norm_elementwise_affine,
            num_parallel=num_parallel, freeze=freeze, epsilon=epsilon
        )
        self.fc1 = ParallelLinear(
            config.hidden_size, config.ffn_dim, bias=config.enable_bias,
            num_parallel=num_parallel, freeze=freeze, epsilon=epsilon
        )
        self.fc2 = ParallelLinear(
            config.ffn_dim, config.hidden_size, bias=config.enable_bias,
            num_parallel=num_parallel, freeze=freeze, epsilon=epsilon
        )
        self.final_layer_norm = ParallelLayerNorm(
            config.hidden_size, eps=1e-5, elementwise_affine=config.layer_norm_elementwise_affine,
            num_parallel=num_parallel, freeze=freeze, epsilon=epsilon
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,   # shape: [num_parallel * B, T, E]
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states  # [N, T, E]
        
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
         # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs


class ParallelOPTDecoder(OPTPreTrainedModel): 
    def __init__(self, config, num_parallel: int = 8):
        super().__init__(config)
        self.config = config
        self.num_parallel = num_parallel
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            padding_idx=config.pad_token_id,
            num_parallel=num_parallel,
            freeze=False,
            epsilon=0.001,
        )
        self.embed_positions = ParallelOPTLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            num_parallel=num_parallel,
            freeze=False,
            epsilon=0.001,
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ParallelLinear(
                config.hidden_size, config.word_embed_proj_dim, bias=False,
                num_parallel=num_parallel, freeze=False, epsilon=0.001
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ParallelLinear(
                config.word_embed_proj_dim, config.hidden_size, bias=False,
                num_parallel=num_parallel, freeze=False, epsilon=0.001
            )
        else:
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = ParallelLayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine,
                num_parallel=num_parallel, freeze=False, epsilon=0.001
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([
            ParallelOPTDecoderLayer(config, num_parallel=num_parallel) 
            for _ in range(config.num_hidden_layers)
        ])

        self.gradient_checkpointing = False
        self.post_init() 

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # [B, T]
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[1:3]  # [B, T]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        num_parallel, batch_size, seq_length, embed_dim = inputs_embeds.size()
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        mask_seq_length = past_key_values_length + seq_length


        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        causal_attention_mask = causal_attention_mask.unsqueeze(0).expand(num_parallel, -1, -1, -1,-1)
        causal_attention_mask = causal_attention_mask.reshape(num_parallel * batch_size, *causal_attention_mask.shape[2:])

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
        inputs_embeds = inputs_embeds.reshape(num_parallel * batch_size, seq_length, embed_dim)
        pos_embeds = pos_embeds.reshape(num_parallel * batch_size, seq_length, pos_embeds.size(-1))

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if head_mask is not None and head_mask.size()[0] != len(self.layers):
            raise ValueError(
                f"The head_mask should be specified for {len(self.layers)} layers, but got {head_mask.size()[0]}."
            )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )




class ParallelOPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig, num_parallel: int = 8):
        super().__init__(config)
        self.num_parallel = num_parallel
        self.decoder = ParallelOPTDecoder(config, num_parallel=num_parallel)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

class ParallelOPTForCausalLM(OPTForCausalLM):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig, num_parallel: int = 8):
        super().__init__(config)
        self.num_parallel = num_parallel
        self.model = ParallelOPTModel(config, num_parallel=num_parallel)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[tuple, dict]:

        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        parallel_hidden_states = outputs[0]
        logits = self.lm_head(parallel_hidden_states).contiguous()
        n = self.num_parallel
        B = logits.shape[0] // n
        T = logits.shape[1]
        logits = logits.view(n, B, T, -1)
        
        loss = None
        if labels is not None:
            labels_expanded = labels.unsqueeze(0).expand(n, -1, -1)
            shift_logits = logits[:, :, :-1, :].contiguous()  # [n, B, T-1, vocab_size]
            shift_labels = labels_expanded[:, :, 1:].contiguous()  # [n, B, T-1]
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss_list = []
            for i in range(n):
                branch_loss = loss_fct(
                    shift_logits[i].view(-1, self.config.vocab_size),
                    shift_labels[i].view(-1)
                )
                loss_list.append(branch_loss)
            loss = loss_list

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ParallelEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, 
                 num_parallel: int = 8, freeze: bool = False, epsilon: float = 0.01):
        super().__init__()
        embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.register_parameter("weight", embedding.weight)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.num_parallel = num_parallel
        self.freeze = freeze
        self.epsilon = epsilon

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # input_ids: [B, T]
        pdb.set_trace()
        print("input_ids.dtype:", input_ids.dtype)
        print("Embedding_self.weight.dtype:", self.weight.dtype)  
        if self.freeze:
            base_embeds = F.embedding(input_ids, self.weight, padding_idx=self.padding_idx)
            return base_embeds.unsqueeze(0).expand(self.num_parallel, -1, -1, -1)
        else:
            weight_orig = self.weight.data.clone()  # [num_embeddings, embedding_dim]
            mu = torch.randint(0, 2, (self.num_parallel, self.num_embeddings, self.embedding_dim),
                               device=self.weight.device).float() * 2 - 1
            weight_perturbed = weight_orig.unsqueeze(0) + self.epsilon * mu  # [num_parallel, num_embeddings, embedding_dim]
            input_ids_expanded = input_ids.unsqueeze(0).expand(self.num_parallel, -1, -1)
            embeds_list = []
            for i in range(self.num_parallel):
                emb = F.embedding(input_ids_expanded[i], weight_perturbed[i], padding_idx=self.padding_idx)
                embeds_list.append(emb)  # emb: [B, T, embedding_dim]
            perturbed_embeds = torch.stack(embeds_list, dim=0)
            self.weight.data.copy_(weight_orig)
            return perturbed_embeds

class ParallelOPTLearnedPositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, num_parallel: int = 8, 
                 padding_idx: int = None, freeze: bool = False, epsilon: float = 0.01):
        super().__init__()
        self.offset = 2
        self.num_parallel = num_parallel
        self.freeze = freeze
        self.epsilon = epsilon


        embedding = nn.Embedding(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)
        self.register_parameter("weight", embedding.weight)
        self.num_embeddings = num_embeddings + self.offset 
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0) -> torch.Tensor:
        pdb.set_trace()
        print("attention_mask.dtype:", attention_mask.dtype)
        print("PositionalEmbedding_self.weight.dtype:", self.weight.dtype)      
        attention_mask = attention_mask.long()
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        positions = positions[:, past_key_values_length:]
        positions = positions + self.offset 

        if self.freeze:
            base_embeds = F.embedding(positions, self.weight, padding_idx=self.padding_idx)
            return base_embeds.unsqueeze(0).expand(self.num_parallel, -1, -1, -1)
        else:
            weight_orig = self.weight.data.clone()  # [num_embeddings, embedding_dim]
            mu = torch.randint(0, 2, (self.num_parallel, self.num_embeddings, self.embedding_dim),
                               device=self.weight.device).float() * 2 - 1
            weight_perturbed = weight_orig.unsqueeze(0) + self.epsilon * mu  # [num_parallel, num_embeddings, embedding_dim]
            positions_expanded = positions.unsqueeze(0).expand(self.num_parallel, -1, -1)
            embeds_list = []
            for i in range(self.num_parallel):
                emb = F.embedding(positions_expanded[i], weight_perturbed[i], padding_idx=self.padding_idx)
                embeds_list.append(emb)  # emb: [B, T', embedding_dim]
            perturbed_embeds = torch.stack(embeds_list, dim=0)
            self.weight.data.copy_(weight_orig)
            return perturbed_embeds



class ParallelLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 num_parallel: int = 8, freeze: bool = False, epsilon: float = 0.01):
        super().__init__(in_features, out_features, bias)
        self.num_parallel = num_parallel
        self.freeze = freeze
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n * B, T, in_features]
        n = self.num_parallel
        B = x.shape[0] // n
        T = x.shape[1]
        pdb.set_trace()
        print("ParallelLinear_x_split.dtype:", x_split.dtype)
        print("ParallelLinear_self.weight.dtype:", self.weight.dtype)        
        x_split = x.view(n, B, T, self.in_features)#.to(self.weight.dtype)

        F_all = torch.matmul(x_split, self.weight.t())
        if self.bias is not None:
            F_all = F_all + self.bias.view(1, 1, 1, -1)

        if self.freeze:
            out = F_all
        else:
            mu = torch.randint(0, 2, (n, self.out_features, self.in_features),
                               device=x.device).float() * 2 - 1
            mu = mu.to(self.weight.dtype)
            mu_t = mu.transpose(-1, -2)
            mu_t = mu_t.unsqueeze(1)
            p_all = self.epsilon * torch.matmul(x_split, mu_t)
            p_all = p_all.squeeze(3)
            out = F_all + p_all

        out = out.view(n * B, T, self.out_features)
        return out


class ParallelLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 num_parallel: int = 8, freeze: bool = False, epsilon: float = 0.01):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.num_parallel = num_parallel
        self.freeze = freeze
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.num_parallel
        B = x.shape[0] // n
        T = x.shape[1]
        E = x.shape[2]
        x_split = x.view(n, B, T, E)

        pdb.set_trace()
        print("ParallelLayerNorm_x_split.dtype:", x_split.dtype)
        print("ParallelLayerNorm_self.weight.dtype:", self.weight.dtype)
        mean = x_split.mean(dim=-1, keepdim=True) 
        var = x_split.var(dim=-1, unbiased=False, keepdim=True)  
        x_norm = (x_split - mean) / torch.sqrt(var + self.eps) 

        if self.freeze:
            effective_weight = self.weight.view(1, 1, 1, E)
            effective_bias = self.bias.view(1, 1, 1, E)
        else:

            delta = torch.randint(0, 2, (n, E), device=x.device).float() * 2 - 1
            eta = torch.randint(0, 2, (n, E), device=x.device).float() * 2 - 1
            effective_weight = self.weight.unsqueeze(0) + self.epsilon * delta  
            effective_bias = self.bias.unsqueeze(0) + self.epsilon * eta            

            effective_weight = effective_weight.view(n, 1, 1, E)
            effective_bias = effective_bias.view(n, 1, 1, E)

        out = x_norm * effective_weight + effective_bias  
        out = out.view(n * B, T, E)
        return out
