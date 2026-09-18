import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

result_dir = Path("results/raw")
processed_dir = Path("results/processed")
figure_dir = Path("figures")

processed_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

files = [
    "baseline_single_concurrency.json",
    "concurrency_2.json",
    "concurrency_4.json",
    "concurrency_8.json",
    "concurrency_16.json",
]

rows = []

for file_name in files:
    file_path = result_dir / file_name

    with open(file_path, "r") as f:
        data = json.load(f)

    rows.append({
        "concurrency": data["max_concurrency"],
        "output_throughput": data["output_throughput"],
        "mean_ttft_ms": data["mean_ttft_ms"],
        "p99_ttft_ms": data["p99_ttft_ms"],
        "mean_tpot_ms": data["mean_tpot_ms"],
        "p99_tpot_ms": data["p99_tpot_ms"],
    })

df = pd.DataFrame(rows)
df = df.sort_values("concurrency").reset_index(drop=True)

print(df)

# 保存处理后的表格
df.to_csv(processed_dir / "concurrency_summary.csv", index=False)


def make_line_plot(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8, 5.2))
    plt.plot(x, y, marker="o", linewidth=2, markersize=7)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=12)
    plt.xticks(x, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()


# 1. Output Throughput vs Concurrency
make_line_plot(
    x=df["concurrency"],
    y=df["output_throughput"],
    xlabel="Concurrency",
    ylabel="Output Throughput (tok/s)",
    title="Output Throughput vs Concurrency",
    save_path=figure_dir / "throughput_vs_concurrency.png",
)

# 2. Mean TTFT vs Concurrency
make_line_plot(
    x=df["concurrency"],
    y=df["mean_ttft_ms"],
    xlabel="Concurrency",
    ylabel="Mean TTFT (ms)",
    title="Mean TTFT vs Concurrency",
    save_path=figure_dir / "mean_ttft_vs_concurrency.png",
)

# 3. P99 TTFT vs Concurrency
make_line_plot(
    x=df["concurrency"],
    y=df["p99_ttft_ms"],
    xlabel="Concurrency",
    ylabel="P99 TTFT (ms)",
    title="P99 TTFT vs Concurrency",
    save_path=figure_dir / "p99_ttft_vs_concurrency.png",
)

# 4. Mean TPOT vs Concurrency
make_line_plot(
    x=df["concurrency"],
    y=df["mean_tpot_ms"],
    xlabel="Concurrency",
    ylabel="Mean TPOT (ms)",
    title="Mean TPOT vs Concurrency",
    save_path=figure_dir / "mean_tpot_vs_concurrency.png",
)

print("Done.")
print("CSV saved to:", processed_dir / "concurrency_summary.csv")
print("Figures saved to:", figure_dir)