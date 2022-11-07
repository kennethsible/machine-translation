from manager import device, triu_mask
import torch

def greedy_search(model, vocab, src_encs, src_mask, max_length):
    path = torch.full((1, max_length), vocab.bos, device=device)
    tgt_mask = triu_mask(max_length, device=device)
    for i in range(1, max_length):
        logits = model.decode(src_encs, path[:, :i], src_mask, tgt_mask[:, :i, :i])
        lprobs = model.generator(logits[:, -1, :])
        path[0, i] = torch.argmax(lprobs, dim=-1).unsqueeze(0)
        if path[0, i] == vocab.eos: break
    return path.squeeze(0)

def beam_search(model, vocab, src_encs, src_mask, max_length, beam_size):
    assert beam_size > 0 and max_length > 0
    finished = torch.zeros(1, dtype=torch.bool, device=device)
    paths = torch.full((1, max_length), vocab.bos, device=device)
    probs = torch.zeros(1, device=device)
    tgt_mask = triu_mask(max_length, device=device)
    beam_size, max_size = 1, beam_size

    for i in range(1, max_length):
        logits = model.decode(src_encs.expand(beam_size, -1, -1),
            paths[~finished, :i], src_mask, tgt_mask[:, :i, :i])
        lprobs = model.generator(logits[:, -1, :])
        scores = probs[~finished].unsqueeze(1) + lprobs
        if i == 1: # increase beam_size to max_size
            finished = finished.repeat(max_size)
            paths = paths.repeat(max_size, 1)
            probs = probs.repeat(max_size)
            beam_size = max_size

        candidates = paths[~finished]
        topv, topi = torch.topk(scores.flatten(), beam_size)
        if finished.any():
            for j in range(beam_size):
                finished[finished == True] ^= probs[finished] < (topv[j] / i)
            nonzero_count = (~finished).count_nonzero()
            if nonzero_count > beam_size:
                beam_size = nonzero_count
                topv, topi = torch.topk(scores.flatten(), beam_size)

        paths[~finished] = candidates[
            torch.div(topi, model.vocab_size, rounding_mode='trunc')
        ]
        paths[~finished, i] = topi % model.vocab_size
        probs[~finished] = topv

        terminated = (paths[:, i] == vocab.eos)
        probs[terminated] /= i
        finished |= terminated
        beam_size = (~finished).count_nonzero()
        if finished.all(): break

    return paths[probs.argmax()]