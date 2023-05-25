from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from manager import Manager


def triu_mask(size: int, device: str | None = None) -> Tensor:
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0


def greedy_search(
    manager: 'Manager', src_encs: Tensor, src_mask: Tensor | None = None, max_length: int = 512
) -> Tensor:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    path = torch.full((1, max_length), vocab.BOS, device=device)

    for i in range(1, max_length):
        tgt_encs = model.decode(src_encs.unsqueeze(0), path[:, :i], src_mask, tgt_mask[:, :i, :i])
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)
        path[0, i] = logits.log_softmax(dim=-1).argmax(dim=-1)
        if path[0, i] == vocab.EOS:
            break

    return path.squeeze(0)


def beam_search(
    manager: 'Manager',
    src_encs: Tensor,
    src_mask: Tensor | None = None,
    beam_size: int = 4,
    max_length: int = 512,
) -> Tensor:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    active = torch.ones(beam_size, dtype=torch.bool, device=device)
    paths = torch.full((beam_size, max_length), vocab.BOS, device=device)
    probs = torch.zeros(beam_size, device=device)

    i, init_size = 0, beam_size
    while (i := i + 1) < max_length and beam_size > 0:
        tgt_encs = model.decode(
            src_encs.expand(beam_size, -1, -1), paths[active, :i], src_mask, tgt_mask[:, :i, :i]
        )
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)
        scores = probs[active].unsqueeze(1) + logits.log_softmax(dim=-1)
        if i == 1:
            scores = scores[0]

        topv, topi = torch.topk(scores.flatten(), beam_size)
        if beam_size < init_size:
            active[~active] |= probs[~active] < topv.max() / i
            active_count = int(active.count_nonzero())
            if active_count > beam_size:
                beam_size = active_count
                topv, topi = torch.topk(scores.flatten(), beam_size)

        reorder = topi // vocab.size()
        paths[active] = paths[active][reorder]
        paths[active, i] = topi % vocab.size()
        probs[active] = topv

        terminated = paths[:, i] == vocab.EOS
        probs[terminated] /= i
        active &= ~terminated
        beam_size = int(active.count_nonzero())

    return paths[probs.argmax()]
