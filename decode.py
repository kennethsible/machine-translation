from manager import triu_mask
import torch

def greedy_decode(manager, src_encs, src_mask, max_length=512):
    path = torch.full((1, max_length), manager.vocab.BOS, device=manager.device)
    tgt_mask = triu_mask(max_length, device=manager.device)
    for i in range(1, max_length):
        tgt_encs = manager.model.decode(path[:, :i], tgt_mask[:, :i, :i], src_encs, src_mask)
        scores = manager.model.generator(tgt_encs[:, -1, :])
        path[0, i] = torch.argmax(scores, dim=-1).unsqueeze(0)
        if path[0, i] == manager.vocab.EOS: break
    return path.squeeze(0)

def beam_decode(manager, src_encs, src_mask, beam_width, max_length=512):
    if beam_width == 1:
        return greedy_decode(manager, src_encs, src_mask, max_length)

    vocab_dim = manager.model.generator.weights.size(0)
    inactive = torch.zeros(1, dtype=torch.bool, device=manager.device)
    paths = torch.full((1, max_length), manager.vocab.BOS, device=manager.device)
    probs = torch.zeros(1, device=manager.device)
    tgt_mask = triu_mask(max_length, device=manager.device)

    beam_width, max_width = 1, beam_width
    for i in range(1, max_length):
        tgt_encs = manager.model.decode(paths[~inactive, :i], tgt_mask[:, :i, :i],
            src_encs.expand(beam_width, -1, -1), src_mask)
        scores = manager.model.generator(tgt_encs[:, -1, :]) + probs[~inactive].unsqueeze(1)
        if i == 1:
            inactive = inactive.repeat(max_width)
            paths = paths.repeat(max_width, 1)
            probs = probs.repeat(max_width)
            beam_width = max_width

        candidates = paths[~inactive]
        topv, topi = torch.topk(scores.flatten(), beam_width)
        if inactive.any():
            for j in range(beam_width):
                inactive[inactive == True] ^= probs[inactive] < (topv[j] / i)
            nonzero_count = (~inactive).count_nonzero()
            if nonzero_count > beam_width:
                beam_width = nonzero_count
                topv, topi = torch.topk(scores.flatten(), beam_width)

        paths[~inactive] = candidates[torch.div(topi, vocab_dim, rounding_mode='trunc')]
        paths[~inactive, i] = topi % vocab_dim
        probs[~inactive] = topv

        finished = (paths[:, i] == manager.vocab.EOS)
        probs[finished] /= i
        inactive |= finished
        beam_width = (~inactive).count_nonzero()
        if inactive.all(): break

    return paths[probs.argmax()]
