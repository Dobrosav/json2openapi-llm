import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_stats():
    summary_path = os.path.join(RESULTS_DIR, "ollama_summary.json")
    if os.path.exists(summary_path):
        ollama = load_json(summary_path)
    else:
        ollama = {
            name: load_json(os.path.join(RESULTS_DIR, f"ollama_{name}.json"))["stats"]
            for name in ["few_shot", "cot", "cove"]
        }

    codet5 = load_json(os.path.join(RESULTS_DIR, "transformer_codet5.json"))["stats"]

    return {
        "CodeT5-small\n(fine-tuned)": codet5,
        "Llama 3\nFew-Shot": ollama["few_shot"],
        "Llama 3\nCoT": ollama["cot"],
        "Llama 3\nCoVe": ollama["cove"],
    }


def plot_metrics_comparison(models):
    labels = list(models.keys())
    metrics = [
        ("yaml_parse_rate", "YAML Parse (%)"),
        ("oas_schema_valid_rate", "OAS 3.0 Valid (%)"),
        ("json_validation_accuracy", "JSON Valid (%)"),
        ("exact_match_rate", "Exact Match (%)"),
    ]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]

    x = np.arange(len(labels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (key, title) in enumerate(metrics):
        vals = [models[m][key] for m in labels]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=title, color=colors[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.0,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Uspešnost (%)")
    ax.set_title("Poređenje performansi modela na test skupu (N = 100)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "01_metrike_poredjenje.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Sačuvan grafik: {out}")


def plot_latency_and_tradeoff(models):
    labels = list(models.keys())
    short_labels = [l.replace("\n", " ") for l in labels]
    latencies = [models[m]["avg_latency_s"] for m in labels]
    exact = [models[m]["exact_match_rate"] for m in labels]
    colors = ["#2E86AB", "#F18F01", "#2A9D8F", "#C73E1D"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Levi panel: Prosečna latencija
    bars = ax1.bar(labels, latencies, color=colors, width=0.55, edgecolor="white")
    for bar, v in zip(bars, latencies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.25,
            f"{v:.2f} s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_ylabel("Prosečna latencija po primeru (s)")
    ax1.set_title("Prosečno vreme inferencije (niže je bolje)")
    ax1.set_ylim(0, max(latencies) * 1.18)
    ax1.set_axisbelow(True)

    # Desni panel: Trade-off (Exact Match vs. Latencija)
    for i, (lat, ex, name) in enumerate(zip(latencies, exact, short_labels)):
        ax2.scatter(lat, ex, s=150, color=colors[i], edgecolor="black", linewidth=0.8, zorder=3)
        offset_x = 0.25 if lat < max(latencies) * 0.85 else -2.8
        offset_y = -2.2 if ex > 97 else 1.2
        ax2.annotate(
            f"{name}\n({ex:.0f}%, {lat:.2f}s)",
            (lat, ex),
            xytext=(lat + offset_x, ex + offset_y),
            fontsize=8.5,
            fontweight="bold",
        )

    ax2.set_xlabel("Prosečna latencija po primeru (s)")
    ax2.set_ylabel("Exact Match (%)")
    ax2.set_title("Odnos tačnosti i brzine (gore-levo je optimalno)")
    ax2.set_xlim(0, max(latencies) * 1.15)
    ax2.set_ylim(75, 105)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "02_tacnost_vs_latencija.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Sačuvan grafik: {out}")


def plot_training_curve():
    log_path = os.path.join(RESULTS_DIR, "codet5_training_log.json")
    if not os.path.exists(log_path):
        print("Preskačem krivu učenja: nije pronađen results/codet5_training_log.json")
        return

    logs = load_json(log_path)
    train_steps = [(e["step"], e["loss"]) for e in logs if "loss" in e]
    eval_epochs = [(e["epoch"], e["eval_loss"]) for e in logs if "eval_loss" in e]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    if train_steps:
        steps, losses = zip(*train_steps)
        ax1.plot(steps, losses, marker="o", markersize=3.5, color="#2E86AB", linewidth=1.8)
        ax1.set_xlabel("Trening korak (step)")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("CodeT5-small: Training Loss po koracima")

    if eval_epochs:
        epochs, ev_losses = zip(*eval_epochs)
        ax2.plot(epochs, ev_losses, marker="s", markersize=7, color="#C73E1D", linewidth=2.0)
        for ep, val in zip(epochs, ev_losses):
            ax2.annotate(
                f"{val:.6f}",
                (ep, val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        ax2.set_xlabel("Epoha")
        ax2.set_ylabel("Validation Loss")
        ax2.set_title("CodeT5-small: Validation Loss po epohama")
        ax2.set_xticks(epochs)
        ax2.set_ylim(0, max(ev_losses) * 1.35 if max(ev_losses) > 0 else 0.0001)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "03_codet5_trening_kriva.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Sačuvan grafik: {out}")


def print_error_breakdown():
    print("\n=== ANALIZA GREŠAKA PO TEHNICI (za diskusiju u izveštaju) ===")
    for name in ["few_shot", "cot", "cove"]:
        path = os.path.join(RESULTS_DIR, f"ollama_{name}.json")
        if not os.path.exists(path):
            continue
        records = load_json(path)["records"]
        yaml_fail = [r for r in records if not r["yaml_valid"]]
        oas_fail = [r for r in records if r["yaml_valid"] and not r["oas_valid"]]
        json_fail = [r for r in records if r["oas_valid"] and not r["json_valid"]]
        exact_only_fail = [r for r in records if r["json_valid"] and not r["exact_match"]]

        print(f"\n[{name.upper()}]")
        print(f"  - YAML sintaksne greške:                 {len(yaml_fail)}")
        print(f"  - Nevalidna OpenAPI 3.0 šema:            {len(oas_fail)}")
        print(f"  - Validna šema, ali JSON pada validaciju:{len(json_fail)}")
        print(f"  - JSON validan, ali nije Exact Match:    {len(exact_only_fail)}")

        for bucket_name, bucket in [("YAML/OAS/JSON greška", yaml_fail + oas_fail + json_fail)]:
            for sample in bucket[:2]:
                err = (sample.get("error") or "").replace("\n", " ")[:140]
                print(f"    Primer #{sample['index']} ({bucket_name}): {err}")


if __name__ == "__main__":
    models = load_all_stats()
    plot_metrics_comparison(models)
    plot_latency_and_tradeoff(models)
    plot_training_curve()
    print_error_breakdown()