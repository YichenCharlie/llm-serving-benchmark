import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


result_dir = Path("results/raw")
processed_dir = Path("results/processed")
figure_dir = Path("figures")


files = {
    "Prefix Cache OFF": "prefix_cache_off.json",
    "Prefix Cache ON": "prefix_cache_on.json",
}


rows = []

for condition, file_name in files.items():
    file_path = result_dir / file_name

    with open(file_path, "r") as f:
        data = json.load(f)

    rows.append(
        {
            "condition": condition,
            "output_throughput": data["output_throughput"],
            "mean_ttft_ms": data["mean_ttft_ms"],
            "p99_ttft_ms": data["p99_ttft_ms"],
            "mean_tpot_ms": data["mean_tpot_ms"],
            "p99_tpot_ms": data["p99_tpot_ms"],
        }
    )


df = pd.DataFrame(rows)

print(df)

df.to_csv(
    processed_dir / "prefix_cache_summary.csv",
    index=False,
)


def make_bar_plot(
    labels,
    values,
    ylabel,
    title,
    save_path,
):
    plt.figure(figsize=(8, 5.2))

    bars = plt.bar(labels, values)

    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=12)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    # Add value labels above bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()


make_bar_plot(
    df["condition"],
    df["mean_ttft_ms"],
    "Mean TTFT (ms)",
    "Mean TTFT: Prefix Caching OFF vs ON",
    figure_dir / "mean_ttft_prefix_cache.png",
)


make_bar_plot(
    df["condition"],
    df["p99_ttft_ms"],
    "P99 TTFT (ms)",
    "P99 TTFT: Prefix Caching OFF vs ON",
    figure_dir / "p99_ttft_prefix_cache.png",
)


make_bar_plot(
    df["condition"],
    df["output_throughput"],
    "Output Throughput (tok/s)",
    "Output Throughput: Prefix Caching OFF vs ON",
    figure_dir / "output_throughput_prefix_cache.png",
)


make_bar_plot(
    df["condition"],
    df["mean_tpot_ms"],
    "Mean TPOT (ms)",
    "Mean TPOT: Prefix Caching OFF vs ON",
    figure_dir / "mean_tpot_prefix_cache.png",
)