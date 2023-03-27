from manager import triu_mask
import torch

def greedy_decode(manager, src_encs, src_mask, max_length=512):
    path = torch.full((1, max_length), manager.vocab.bos, device=manager.device)
    tgt_mask = triu_mask(max_length, device=manager.device)
    for i in range(1, max_length):
        logits = manager.model.decode(src_encs,
            path[:, :i], src_mask, tgt_mask[:, :i, :i])
        lprobs = manager.model.generator(logits[:, -1, :])
        path[0, i] = torch.argmax(lprobs, dim=-1).unsqueeze(0)
        if path[0, i] == manager.vocab.eos: break
    return path.squeeze(0)

def beam_decode(manager, src_encs, src_mask, beam_width, max_length=512):
    if beam_width == 1: return greedy_decode(manager, src_encs, src_mask, max_length)

    finished = torch.zeros(1, dtype=torch.bool, device=manager.device)
    paths = torch.full((1, max_length), manager.vocab.bos, device=manager.device)
    probs = torch.zeros(1, device=manager.device)
    tgt_mask = triu_mask(max_length, device=manager.device)

    curr_width =  1
    for i in range(1, max_length):
        logits = manager.model.decode(src_encs.expand(curr_width, -1, -1),
            paths[~finished, :i], src_mask, tgt_mask[:, :i, :i])
        lprobs = manager.model.generator(logits[:, -1, :])
        scores = probs[~finished].unsqueeze(1) + lprobs
        if i == 1: # increase curr_width to beam_width
            finished = finished.repeat(beam_width)
            paths = paths.repeat(beam_width, 1)
            probs = probs.repeat(beam_width)
            curr_width = beam_width

        candidates = paths[~finished]
        topv, topi = torch.topk(scores.flatten(), curr_width)
        if finished.any():
            for j in range(curr_width):
                finished[finished == True] ^= probs[finished] < (topv[j] / i)
            nonzero_count = (~finished).count_nonzero()
            if nonzero_count > curr_width:
                curr_width = nonzero_count
                topv, topi = torch.topk(scores.flatten(), curr_width)

        paths[~finished] = candidates[
            torch.div(topi, manager.model.vocab_size, rounding_mode='trunc')
        ]
        paths[~finished, i] = topi % manager.model.vocab_size
        probs[~finished] = topv

        terminated = (paths[:, i] == manager.vocab.eos)
        probs[terminated] /= i
        finished |= terminated
        curr_width = (~finished).count_nonzero()
        if finished.all(): break

    return paths[probs.argmax()]
