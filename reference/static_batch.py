"""Static batching with the Hugging Face KV cache: B prompts, one forward per step.

Analogy: a tour bus. Everyone boards together and the bus only returns when
the last passenger is done, so seats of people who finished early ride empty.
That wasted seat is exactly what continuous batching fixes.

Prompts have different lengths, so they are LEFT-padded to line up their last
tokens, with an attention mask hiding the padding and position ids that start
at 0 for each prompt's first real token.
"""

from typing import Any

import torch
from torch import Tensor

from engine.sampler import sample


@torch.no_grad()
def generate_static(
    model: Any,
    prompts: list[list[int]],
    max_new: int,
    eos_id: int,
    pad_id: int = 0,
    **sampling: Any,
) -> list[list[int]]:
    """Generate for all prompts in one batch; return each row's tokens up to EOS."""
    if max_new <= 0:
        return [[] for _ in prompts]
    device = next(model.parameters()).device
    lengths = [len(p) for p in prompts]
    width = max(lengths)

    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), width, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt) :] = torch.tensor(prompt)
        mask[row, width - len(prompt) :] = 1
    input_ids, mask = input_ids.to(device), mask.to(device)
    # Position 0 = first real token of each row, whatever its padding.
    positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)

    outputs = model(
        input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=True
    )
    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)

    for step in range(max_new):
        next_tokens: Tensor = sample(outputs.logits[:, -1, :], **sampling)  # [B, 1]
        for row, token in enumerate(next_tokens.squeeze(-1).tolist()):
            if not finished[row]:
                generated[row].append(int(token))
                finished[row] = token == eos_id
        if all(finished) or step == max_new - 1:
            break
        # Finished rows keep decoding (their tokens are dropped): the cost of static batching.
        mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=-1)
        positions = positions[:, -1:] + 1
        outputs = model(
            input_ids=next_tokens,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
    return generated
