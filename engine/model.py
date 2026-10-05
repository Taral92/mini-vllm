"""Run Qwen2 forward passes against the paged KV cache.

The Hugging Face model supplies the weights and the small building blocks
(embeddings, RMSNorm, projections, RoPE, MLP). This file replaces the one part
HF's cache can't do: attention for a batch where every request is at a
different length and its keys/values live in scattered cache blocks.

One step can mix requests doing prefill (many new tokens) and decode (one new
token). Each request's new tokens are padded to the longest in the step.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .cache import PagedKVCache


@dataclass(frozen=True, slots=True)
class BatchInput:
    """What one model step needs for each scheduled request."""

    token_ids: list[list[int]]  # new tokens (prompt for prefill, 1 for decode)
    start_positions: list[int]  # position of the first new token
    block_tables: list[list[int]]  # cache blocks owned by the request


class PagedModelRunner:
    """Qwen2 forward over a batch of variable-length requests, paged KV."""

    def __init__(self, hf_model: Any, cache: PagedKVCache) -> None:
        config = hf_model.config
        if getattr(config, "use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        self.model = hf_model
        self.cache = cache
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.group = self.num_heads // self.num_kv_heads  # query heads per KV head
        self.device = next(hf_model.parameters()).device

    @torch.no_grad()
    def forward(self, batch: BatchInput) -> Tensor:
        """Write the new tokens' K/V into the cache and return next-token logits.

        Returns:
            Logits for the last new token of each request, shaped [B, vocab].
        """
        inner = self.model.model
        device = self.device
        batch_size = len(batch.token_ids)
        q_lens = [len(ids) for ids in batch.token_ids]
        ctx_lens = [s + n for s, n in zip(batch.start_positions, q_lens, strict=True)]
        max_q, max_ctx = max(q_lens), max(ctx_lens)

        # All step metadata is built on the CPU, where every size is known,
        # then copied over once. Nothing below makes the GPU report a size
        # back to Python: a boolean-mask index like k[q_valid] does exactly
        # that (it must count the Trues first), and at 2 per layer x 24 layers
        # those stalls made this step 3x slower than static batching on MPS.
        #
        # Pad new tokens to [B, max_q]. Padded query rows reuse the first
        # position so their mask is never empty; their outputs are discarded.
        size = self.cache.block_size
        ids = torch.zeros(batch_size, max_q, dtype=torch.long)
        positions = torch.zeros(batch_size, max_q, dtype=torch.long)
        ctx_slots = torch.zeros(batch_size, max_ctx, dtype=torch.long)
        ctx_valid = torch.zeros(batch_size, max_ctx, dtype=torch.bool)
        valid_rows: list[Tensor] = []  # flat [B*max_q] index of each real new token
        for row, (tokens, start, table) in enumerate(
            zip(batch.token_ids, batch.start_positions, batch.block_tables, strict=True)
        ):
            n, ctx = len(tokens), start + len(tokens)
            ids[row, :n] = torch.tensor(tokens)
            positions[row, :n] = torch.arange(start, ctx)
            positions[row, n:] = start
            pos = torch.arange(ctx)
            ctx_slots[row, :ctx] = torch.tensor(table)[pos // size] * size + pos % size
            ctx_valid[row, :ctx] = True
            valid_rows.append(torch.arange(n) + row * max_q)
        valid_flat = torch.cat(valid_rows)
        # Slots of the new tokens = the tail of each row's context slots.
        slot_mapping = torch.cat(
            [
                ctx_slots[row, start : start + n]
                for row, (start, n) in enumerate(
                    zip(batch.start_positions, q_lens, strict=True)
                )
            ]
        )

        ids, positions = ids.to(device), positions.to(device)
        ctx_slots, ctx_valid = ctx_slots.to(device), ctx_valid.to(device)
        valid_flat, slot_mapping = valid_flat.to(device), slot_mapping.to(device)

        # Causal mask over absolute positions: query at p sees keys 0..p of its
        # own request, and nothing past that request's length.
        key_pos = torch.arange(max_ctx, device=device)
        mask = (key_pos[None, None, :] <= positions[:, :, None]) & ctx_valid[:, None, :]
        mask = mask[:, None, :, :]  # [B, 1, max_q, max_ctx], broadcast over heads

        hidden = inner.embed_tokens(ids)
        cos, sin = inner.rotary_emb(hidden, positions)
        for index, layer in enumerate(inner.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            hidden = self._attention(
                layer.self_attn,
                index,
                hidden,
                cos,
                sin,
                valid_flat,
                slot_mapping,
                ctx_slots,
                mask,
            )
            hidden = residual + hidden
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

        hidden = inner.norm(hidden)
        last = torch.tensor([n - 1 for n in q_lens], device=device)
        last_hidden = hidden[torch.arange(batch_size, device=device), last]
        logits: Tensor = self.model.lm_head(last_hidden)
        return logits

    def _attention(
        self,
        attn: Any,
        layer_index: int,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        valid_flat: Tensor,
        slot_mapping: Tensor,
        ctx_slots: Tensor,
        mask: Tensor,
    ) -> Tensor:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        batch_size, max_q, _ = hidden.shape
        q = attn.q_proj(hidden).view(batch_size, max_q, self.num_heads, self.head_dim)
        k = attn.k_proj(hidden).view(
            batch_size, max_q, self.num_kv_heads, self.head_dim
        )
        v = attn.v_proj(hidden).view(
            batch_size, max_q, self.num_kv_heads, self.head_dim
        )
        # RoPE on absolute positions: a decode token at position 40 is rotated
        # as position 40, not 0 (the classic "garbage after token 1" bug).
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)

        # 1. Write: only real (non-padding) new tokens go into their slots.
        #    index_select with a precomputed index: no GPU->CPU sync.
        key_pool, value_pool = self.cache.layer(layer_index)
        flat_k = k.transpose(1, 2).reshape(
            batch_size * max_q, self.num_kv_heads, self.head_dim
        )
        flat_v = v.reshape(batch_size * max_q, self.num_kv_heads, self.head_dim)
        key_pool[slot_mapping] = flat_k.index_select(0, valid_flat)
        value_pool[slot_mapping] = flat_v.index_select(0, valid_flat)

        # 2. Read: gather each request's whole context (old + new) from its blocks.
        keys = key_pool[ctx_slots].transpose(1, 2)  # [B, kv_heads, max_ctx, d]
        values = value_pool[ctx_slots].transpose(1, 2)
        # GQA: each of the 2 KV heads serves 7 query heads.
        keys = keys.repeat_interleave(self.group, dim=1)
        values = values.repeat_interleave(self.group, dim=1)

        out = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask, scale=attn.scaling
        )
        out = out.transpose(1, 2).reshape(
            batch_size, max_q, self.num_heads * self.head_dim
        )
        projected: Tensor = attn.o_proj(out)
        return projected


def build_cache(hf_model: Any, num_blocks: int, block_size: int) -> PagedKVCache:
    """Size a PagedKVCache from the model's config, dtype and device."""
    config = hf_model.config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    param = next(hf_model.parameters())
    return PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        dtype=param.dtype,
        device=param.device,
    )
