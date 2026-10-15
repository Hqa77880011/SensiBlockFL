import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(clean(rows))


def auc(y, score):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    n, p = (~y).sum(), y.sum()
    if not n or not p:
        return None
    return float((rankdata(score)[y].sum() - p * (p + 1) / 2) / (p * n))


def roc(y, score):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    if y.all() or not y.any():
        return None, None
    order = np.argsort(-score, kind="stable")
    yy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tpr = np.r_[0.0, np.cumsum(yy)[ends] / y.sum()]
    fpr = np.r_[0.0, np.cumsum(~yy)[ends] / (~y).sum()]
    return fpr, tpr


def classification(y, prob, risk=None, bins=15):
    y, prob = np.asarray(y, int), np.asarray(prob, float)
    pred = prob.argmax(-1)
    correct = pred == y
    f1 = []
    recalls = []
    for k in np.unique(np.r_[y, pred]):
        tp = np.sum((y == k) & (pred == k))
        denom = np.sum(y == k) + np.sum(pred == k)
        f1.append(2 * tp / max(denom, 1))
        if np.any(y == k):
            recalls.append(tp / np.sum(y == k))
    conf = prob.max(-1)
    ece = 0.0
    for lo, hi in zip(
        np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]
    ):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    risk = 1 - conf if risk is None else np.asarray(risk)
    order = np.argsort(risk, kind="stable")
    curve = np.cumsum(~correct[order]) / np.arange(1, len(y) + 1)
    return dict(
        accuracy=correct.mean(),
        balanced_accuracy=np.mean(recalls),
        macro_f1=np.mean(f1),
        ece=ece,
        aurc=curve.mean(),
        error_auroc=auc(~correct, risk),
    )


def grouped_split(groups, seed=0, ratios=(0.7, 0.15, 0.15)):
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three independent groups are required")
    np.random.default_rng(seed).shuffle(unique)
    a = min(max(1, int(len(unique) * ratios[0])), len(unique) - 2)
    b = min(max(a + 1, int(len(unique) * (ratios[0] + ratios[1]))), len(unique) - 1)
    mapping = {
        str(g): s
        for block, s in zip(np.split(unique, [a, b]), ["train", "val", "test"])
        for g in block
    }
    return np.array([mapping[str(g)] for g in groups])


def plot(path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (
        load(path)
        if str(path).endswith(".json")
        else list(csv.DictReader(Path(path).open(encoding="utf-8")))
    )
    rows = rows if isinstance(rows, list) else [rows]
    keys = [
        k
        for k in rows[0]
        if any(isinstance(r.get(k), (int, float)) for r in rows)
        and all(r.get(k) is None or isinstance(r.get(k), (int, float)) for r in rows)
    ]
    if not keys:
        keys = [
            k
            for k in rows[0]
            if all(
                r.get(k) not in (None, "")
                and str(r[k]).replace(".", "", 1).replace("-", "", 1).isdigit()
                for r in rows
            )
        ]
    if not keys:
        raise ValueError("No numeric measurements to plot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = [
        str(r.get("method", r.get("variant", r.get("id", i))))
        for i, r in enumerate(rows)
    ]
    for key in keys:
        fig, ax = plt.subplots(figsize=(max(5, len(rows) * 0.6), 3.5))
        ax.bar(
            labels,
            [float(r[key]) if r.get(key) is not None else float("nan") for r in rows],
            color="#417b97",
        )
        ax.set_ylabel(key.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output / f"{key}.png", dpi=180)
        fig.savefig(output / f"{key}.svg")
        plt.close(fig)


def report():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/seeds")
    args = parser.parse_args()
    groups = {}
    condition_fields = [
        "method",
        "variant",
        "endpoint",
        "subset",
        "channels",
        "mask",
        "snr_db",
        "factor",
        "intervention",
        "reviewed_masks",
        "occlusion",
    ]
    for filename in args.inputs:
        contents = load(filename)
        records = contents if isinstance(contents, list) else [contents]
        for record in records:
            identity = tuple(
                (key, record[key]) for key in condition_fields if key in record
            )
            groups.setdefault(identity, []).append(record)
    rows = []
    for identity, records in groups.items():
        row = dict(identity)
        row["independent_runs"] = len(records)
        keys = sorted(
            {
                key
                for record in records
                for key, value in record.items()
                if key not in condition_fields and isinstance(value, (int, float))
            }
        )
        for key in keys:
            values = [
                record[key]
                for record in records
                if isinstance(record.get(key), (int, float))
                and np.isfinite(record[key])
            ]
            row[key + "_mean"] = float(np.mean(values)) if values else None
            row[key + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        rows.append(row)
    save(Path(args.output) / "summary.json", rows)
    table(Path(args.output) / "summary.csv", rows)
    plot(Path(args.output) / "summary.json", Path(args.output) / "figures")
