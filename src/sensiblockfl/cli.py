import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import datasets, models
from .common import load, save, table, plot
from .compression import compress


def network(classes):
    model = models.resnet18(num_classes=classes)
    model.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def prepare(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10 if config["dataset"] == "cifar10" else datasets.CIFAR100
    train = dataset(directory / "raw", train=True, download=True)
    test = dataset(directory / "raw", train=False, download=True)
    rng = np.random.default_rng(config["seed"])
    labels = np.array(train.targets)
    partitions = [[] for _ in range(config["clients"])]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        sizes = rng.multinomial(
            len(indices), rng.dirichlet(np.full(config["clients"], config["alpha"]))
        )
        for client, part in zip(partitions, np.split(indices, np.cumsum(sizes)[:-1])):
            client.extend(part.tolist())
    if min(map(len, partitions)) < 2:
        raise ValueError(
            "Dirichlet split produced a client with fewer than two samples; choose another seed"
        )
    np.savez_compressed(
        directory / "cifar.npz",
        x=np.asarray(train.data),
        y=labels,
        test_x=np.asarray(test.data),
        test_y=np.array(test.targets),
    )
    save(directory / "partitions.json", partitions)
    save(directory / "preparation.json", config)


def loader(x, y, batch, shuffle, device, augment=False, generator=None):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float() / 255, torch.as_tensor(y)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, generator=generator
    )


def accuracy(model, batches, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / total


def train(config, data, output, method):
    if not 0 <= config["eta"] < 1 or config["temperature"] <= 0:
        raise ValueError("Require eta in [0,1) and positive temperature")
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    data = Path(data)
    archive = np.load(data / "cifar.npz")
    parts = load(data / "partitions.json")
    prep = load(data / "preparation.json")
    if any(prep[k] != config[k] for k in ["dataset", "clients", "alpha"]):
        raise ValueError("Dataset partition settings differ; rerun prepare")
    device = config["device"]
    global_model = network(int(archive["y"].max()) + 1).to(device)
    trainable = dict(global_model.named_parameters())
    zeros = {n: torch.zeros_like(p) for n, p in trainable.items()}
    residuals = [{n: v.clone() for n, v in zeros.items()} for _ in parts]
    emas = [{} for _ in parts]
    control = [{n: v.clone() for n, v in zeros.items()} for _ in parts]
    global_control = {n: v.clone() for n, v in zeros.items()}
    momentum = {n: v.clone() for n, v in zeros.items()}
    history = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "config.json", dict(config, method=method))
    cumulative = 0
    for round_id in range(config["rounds"]):
        selected = rng.choice(len(parts), config["participation"], replace=False)
        transmitted, weights, buffer_states, control_deltas = [], [], [], []
        duration = probe_duration = 0.0
        counts = []
        for i in selected:
            indices = np.array(parts[i])
            probe_idx = indices[: min(config["batch_size"], len(indices) // 4 or 1)]
            fit_idx = indices[len(probe_idx) :]
            batches = loader(
                archive["x"][fit_idx],
                archive["y"][fit_idx],
                config["batch_size"],
                True,
                device,
                generator=generator,
            )
            local = copy.deepcopy(global_model).train()
            optimizer = torch.optim.SGD(
                local.parameters(),
                lr=config["lr"],
                momentum=0.0 if method == "scaffold" else config["momentum"],
                weight_decay=config["weight_decay"],
            )
            start = time.perf_counter()
            steps = 0
            for _ in range(config["local_epochs"]):
                for x, y in batches:
                    x, y = x.to(device), y.to(device)
                    padded = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
                    crop_y, crop_x = rng.integers(0, 9, size=2)
                    x = padded[:, :, crop_y : crop_y + 32, crop_x : crop_x + 32]
                    if rng.random() < 0.5:
                        x = x.flip(-1)
                    loss = nn.functional.cross_entropy(local(x), y)
                    if method == "fedprox":
                        loss += (
                            config["proximal"]
                            / 2
                            * sum(
                                (p - trainable[n].detach()).square().sum()
                                for n, p in local.named_parameters()
                            )
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    if method == "scaffold":
                        for n, p in local.named_parameters():
                            p.grad.add_(global_control[n] - control[i][n])
                    optimizer.step()
                    steps += 1
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            duration += time.perf_counter() - start
            update = {
                n: p.detach() - trainable[n].detach()
                for n, p in local.named_parameters()
            }
            if method == "scaffold":
                new = {
                    n: control[i][n]
                    - global_control[n]
                    - update[n] / (steps * config["lr"])
                    for n in update
                }
                control_deltas.append({n: new[n] - control[i][n] for n in new})
                control[i] = new
            start = time.perf_counter()
            if method in ("fedavg", "fedprox", "scaffold"):
                sent = update
                nbytes = sum(4 * p.numel() for p in update.values())
                allocation = []
            else:
                u = {
                    n: v + (0 if method == "no_ef" else residuals[i][n])
                    for n, v in update.items()
                }
                grads = None
                if method not in ("topk", "random", "uniform", "no_sensitivity"):
                    px, py = next(
                        iter(
                            loader(
                                archive["x"][probe_idx],
                                archive["y"][probe_idx],
                                len(probe_idx),
                                False,
                                device,
                            )
                        )
                    )
                    local.eval()
                    local.zero_grad(set_to_none=True)
                    nn.functional.cross_entropy(
                        local(px.to(device)), py.to(device)
                    ).backward()
                    grads = {n: p.grad.detach() for n, p in local.named_parameters()}
                sent, nbytes, allocation = compress(
                    u, grads, emas[i], config, method, generator
                )
                residuals[i] = {n: u[n] - sent[n] for n in u}
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            probe_duration += time.perf_counter() - start
            if method == "scaffold":
                nbytes += sum(4 * p.numel() for p in control_deltas[-1].values())
            cumulative += nbytes
            cosine = sum((sent[n] * momentum[n]).sum() for n in sent) / (
                sum(v.square().sum() for v in sent.values()).sqrt()
                * sum(v.square().sum() for v in momentum.values()).sqrt()
                + 1e-8
            )
            modifier = (
                1.0
                if method
                in (
                    "fedavg",
                    "fedprox",
                    "scaffold",
                    "uniform",
                    "random",
                    "topk",
                    "no_aggregation",
                )
                else float(
                    torch.clamp(
                        1 + config["eta"] * cosine, 1 - config["eta"], 1 + config["eta"]
                    )
                )
            )
            weights.append(len(fit_idx) * modifier)
            transmitted.append(sent)
            counts.append(allocation)
            buffer_states.append(dict(local.named_buffers()))
        weights = np.array(weights) / sum(weights)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                delta = sum(float(w) * sent[n] for w, sent in zip(weights, transmitted))
                p.add_(delta)
                momentum[n].mul_(config["server_momentum"]).add_(delta)
                if method == "scaffold":
                    global_control[n].add_(
                        sum(
                            len(parts[int(i)]) * d[n]
                            for i, d in zip(selected, control_deltas)
                        )
                        / sum(map(len, parts))
                    )
            for n, b in global_model.named_buffers():
                values = [bs[n] for bs in buffer_states]
                b.copy_(
                    sum(float(w) * v for w, v in zip(weights, values))
                    if b.is_floating_point()
                    else torch.stack(values).max(0).values
                )
        val = accuracy(
            global_model,
            loader(archive["test_x"], archive["test_y"], 256, False, device),
            device,
        )
        history.append(
            dict(
                round=round_id + 1,
                method=method,
                accuracy=val,
                uplink_bytes=cumulative,
                client_training_s=duration,
                compression_s=probe_duration,
                extra_compute_fraction=probe_duration / max(duration, 1e-12),
            )
        )
        table(output / "history.csv", history)
        save(
            output / "allocations.json",
            dict(round=round_id + 1, clients=selected, blocks=counts),
        )
    torch.save(
        dict(
            model=global_model.cpu().state_dict(),
            classes=int(archive["y"].max()) + 1,
            config=config,
        ),
        output / "model.pt",
    )
    save(output / "metrics.json", history[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "compare", "evaluate", "plot"]
    )
    p.add_argument("--config", default="configs/cifar10.json")
    p.add_argument("--data", default="data/cifar10")
    p.add_argument("--output", default="runs/sensiblockfl")
    p.add_argument(
        "--method",
        choices=[
            "sensiblockfl",
            "fedavg",
            "fedprox",
            "scaffold",
            "topk",
            "uniform",
            "random",
            "no_ef",
            "no_sensitivity",
            "no_aggregation",
        ],
        default="sensiblockfl",
    )
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--inputs", nargs="*")
    p.add_argument("--checkpoint")
    a = p.parse_args()
    c = load(a.config)
    import json

    for item in a.set:
        key, value = item.split("=", 1)
        if key not in c:
            raise ValueError(key)
        c[key] = json.loads(value)
    if a.command == "prepare":
        prepare(c, a.data)
    elif a.command == "train":
        train(c, a.data, a.output, a.method)
    elif a.command == "compare":
        for method in [
            "fedavg",
            "fedprox",
            "scaffold",
            "topk",
            "random",
            "uniform",
            "sensiblockfl",
            "no_ef",
            "no_sensitivity",
            "no_aggregation",
        ]:
            train(c, a.data, Path(a.output) / method, method)
    elif a.command == "evaluate":
        rows = [load(Path(d) / "metrics.json") for d in a.inputs]
        table(Path(a.output) / "comparison.csv", rows)
        save(Path(a.output) / "comparison.json", rows)
        target = c.get("target_accuracy", 0.5)
        crossings = []
        import csv

        for d in a.inputs:
            history = list(csv.DictReader((Path(d) / "history.csv").open()))
            crossed = next((r for r in history if float(r["accuracy"]) >= target), None)
            crossings.append(
                dict(
                    method=history[0]["method"],
                    target_accuracy=target,
                    rounds_to_target=int(crossed["round"]) if crossed else None,
                    bytes_to_target=int(crossed["uplink_bytes"]) if crossed else None,
                )
            )
        save(Path(a.output) / "time_to_target.json", crossings)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")


if __name__ == "__main__":
    main()
