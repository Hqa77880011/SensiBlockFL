import math
import torch


def allocate(capacities, scores, budget, minimum):
    cap = torch.tensor(capacities, dtype=torch.long)
    budget = max(0, min(int(budget), int(cap.sum())))
    result = torch.minimum(cap, torch.full_like(cap, minimum))
    if int(result.sum()) > budget:
        result.zero_()
    left = budget - int(result.sum())
    while left:
        active = result < cap
        weights = scores.double().cpu().clone()
        weights[~active] = 0
        if not weights.sum():
            weights = active.double()
        quota = left * weights / weights.sum()
        full = torch.minimum(quota.floor().long(), cap - result)
        result += full
        left -= int(full.sum())
        if not left:
            break
        order = sorted(
            range(len(cap)), key=lambda i: (-float(quota[i] - quota[i].floor()), i)
        )
        for i in order:
            if left and result[i] < cap[i]:
                result[i] += 1
                left -= 1
    return result.tolist()


def compress(updates, gradients, ema, config, method, generator):
    sparse = [name for name, value in updates.items() if value.ndim > 1]
    block = config["block_size"]
    capacities = [math.ceil(updates[n].numel() / block) for n in sparse]
    budget = int(round(config["rate"] * sum(capacities)))
    scores = torch.zeros(len(sparse), dtype=torch.float64)
    if gradients is not None:
        for i, name in enumerate(sparse):
            score = abs(float(torch.sum(gradients[name] * updates[name]))) / (
                updates[name].numel() ** 0.5 + 1e-8
            )
            ema[name] = (
                config["beta"] * ema.get(name, 0.0) + (1 - config["beta"]) * score
            )
            scores[i] = ema[name]
    probability = torch.softmax(scores / config["temperature"], 0)
    counts = allocate(capacities, probability, budget, config["minimum"])
    result = {
        n: u.clone() if u.ndim <= 1 else torch.zeros_like(u) for n, u in updates.items()
    }
    index_bytes = max(1, math.ceil(math.log2(max(sum(capacities), 2)) / 8))
    num_values = sum(u.numel() for u in updates.values() if u.ndim <= 1)
    num_blocks = 0
    if method == "topk":
        vector = torch.cat([updates[n].flatten() for n in sparse])
        k = int(round(config["rate"] * len(vector)))
        indices = torch.argsort(vector.abs(), descending=True, stable=True)[:k]
        values = torch.zeros_like(vector)
        values[indices] = vector[indices]
        start = 0
        for name in sparse:
            size = updates[name].numel()
            result[name] = values[start : start + size].reshape_as(updates[name])
            start += size
        return result, 4 * (num_values + k) + 4 * k, counts
    for name, k in zip(sparse, counts):
        pieces = updates[name].flatten().split(block)
        norm = [float(p.norm()) / (p.numel() ** 0.5 + 1e-8) for p in pieces]
        order = (
            torch.randperm(len(pieces), generator=generator).tolist()
            if method == "random"
            else sorted(range(len(pieces)), key=lambda b: (-norm[b], b))
        )
        flat = result[name].view(-1)
        for b in order[:k]:
            start = b * block
            flat[start : start + len(pieces[b])] = pieces[b]
            num_values += len(pieces[b])
            num_blocks += 1
    return result, 4 * num_values + index_bytes * num_blocks, counts
