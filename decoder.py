import warnings, torch

warnings.filterwarnings('ignore', category=UserWarning)

def triu_mask(size, device=None):
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0

def greedy_search(manager, src_encs, src_mask, max_length=512):
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    path = torch.full((1, max_length), vocab.BOS, device=device)

    for i in range(1, max_length):
        tgt_encs = model.decode(path[:, :i], tgt_mask[:, :i, :i],
            src_encs.unsqueeze(0), src_mask)
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)
        path[0, i] = logits.log_softmax(dim=-1).argmax(dim=-1)
        if path[0, i] == vocab.EOS: break

    return path.squeeze(0)

def beam_search(manager, src_encs, src_mask, beam_size, max_length=512):
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    active = torch.ones(beam_size, dtype=torch.bool, device=device)
    paths = torch.full((beam_size, max_length), vocab.BOS, device=device)
    probs = torch.zeros(beam_size, device=device)

    i, init_size = 0, beam_size
    while (i := i + 1) < max_length and beam_size > 0:
        tgt_encs = model.decode(paths[active, :i], tgt_mask[:, :i, :i],
            src_encs.expand(beam_size, -1, -1), src_mask)
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)
        scores = probs[active].unsqueeze(1) + logits.log_softmax(dim=-1)
        if i == 1: scores = scores[0]

        topv, topi = torch.topk(scores.flatten(), beam_size)
        if beam_size < init_size:
            active[~active] |= probs[~active] < topv.max() / i
            active_count = active.count_nonzero()
            if active_count > beam_size:
                beam_size = active_count
                topv, topi = torch.topk(scores.flatten(), beam_size)

        paths[active] = paths[active][topi // vocab.size()]
        paths[active, i], probs[active] = topi % vocab.size(), topv

        terminated = paths[:, i] == vocab.EOS
        probs[terminated] /= i
        active &= ~terminated
        beam_size = active.count_nonzero()

    return paths[probs.argmax()]
