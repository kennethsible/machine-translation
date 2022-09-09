from manager import subsequent_mask
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def greedy_search(model, memory, src_mask=None, max_len=256, start_word=0, end_word=1):
    path = torch.full((1, 1), start_word, device=device)
    for i in range(1, max_len + 1):
        logits = model.decode(memory, src_mask, path, subsequent_mask(i))
        lprobs = model.generator(logits[:, -1])
        next_word = torch.argmax(lprobs, dim=-1)
        if next_word == end_word: break
        path = torch.cat([path, next_word.unsqueeze(0)], dim=-1)
    return path.squeeze(0)[1:]

def beam_search(model, memory, beam_size, src_mask=None, max_len=256, start_word=0, end_word=1):
    probs = torch.zeros(1, device=device)
    paths = torch.full((1, max_len + 1), start_word, device=device)
    finished = torch.zeros(1, dtype=torch.bool, device=device)

    for i in range(1, max_len + 1):
        logits = model.decode(memory.expand((~finished).count_nonzero(), -1, -1),
            src_mask, paths[~finished, :i], subsequent_mask(i))
        scores = probs[~finished].unsqueeze(1) + model.generator(logits[:, -1])
        if i == 1: # increase capacity to beam_size
            probs = probs.repeat(beam_size)
            paths = paths.repeat(beam_size, 1)
            finished = finished.repeat(beam_size)

        candidates = paths[~finished]
        topv, topi = torch.topk(scores.flatten(), beam_size)
        if any(finished): # length normalization
            for j in range(beam_size):
                finished[finished.nonzero(as_tuple=True)] ^= probs[finished] < (topv[j] / i)
            if (~finished).count_nonzero() > beam_size:
                beam_size = (~finished).sum()
                topv, topi = torch.topk(scores.flatten(), beam_size)

        probs[~finished] = topv
        paths[~finished] = candidates[topi // model.vocab_size]
        paths[~finished, i] = topi % model.vocab_size

        finished |= paths[:, i] == end_word
        beam_size = (~finished).count_nonzero()
        probs[paths[:, i] == end_word] /= i
        if all(finished): break

    best_path = paths[probs.argmax()]
    return best_path[1:(best_path == end_word).nonzero()]
