import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------

result_dir = Path("results/raw")
processed_dir = Path("results/processed")
figure_dir = Path("figures")


# -----------------------------
# Input files
# -----------------------------

files = [
    "context_128.json",
    "context_512.json",
    "context_1024.json",
    "context_2048.json",
]


# -----------------------------
# Load benchmark results
# -----------------------------

rows = []

for file_name in files:
    file_path = result_dir / file_name

    with open(file_path, "r") as f:
        data = json.load(f)

    rows.append({
        "input_length": data["total_input_tokens"] // data["num_prompts"],
        "output_throughput": data["output_throughput"],
        "mean_ttft_ms": data["mean_ttft_ms"],
        "p99_ttft_ms": data["p99_ttft_ms"],
        "mean_tpot_ms": data["mean_tpot_ms"],
        "p99_tpot_ms": data["p99_tpot_ms"],
    })


# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame(rows)

df = df.sort_values("input_length").reset_index(drop=True)

print(df)


# -----------------------------
# Save processed CSV
# -----------------------------

df.to_csv(
    processed_dir / "context_length_summary.csv",
    index=False
)


# -----------------------------
# Helper function for plotting
# -----------------------------

def make_line_plot(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8, 5.2))

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        markersize=7
    )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.title(
        title,
        fontsize=14,
        pad=12
    )

    plt.xticks(x, fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# -----------------------------
# Figure 1: Mean TTFT
# -----------------------------

make_line_plot(
    x=df["input_length"],
    y=df["mean_ttft_ms"],
    xlabel="Input Length (tokens)",
    ylabel="Mean TTFT (ms)",
    title="Mean TTFT vs Input Length",
    save_path=figure_dir / "ttft_vs_context_length.png",
)


# -----------------------------
# Figure 2: P99 TTFT
# -----------------------------

make_line_plot(
    x=df["input_length"],
    y=df["p99_ttft_ms"],
    xlabel="Input Length (tokens)",
    ylabel="P99 TTFT (ms)",
    title="P99 TTFT vs Input Length",
    save_path=figure_dir / "p99_ttft_vs_context_length.png",
)


# -----------------------------
# Figure 3: Mean TPOT
# -----------------------------

make_line_plot(
    x=df["input_length"],
    y=df["mean_tpot_ms"],
    xlabel="Input Length (tokens)",
    ylabel="Mean TPOT (ms)",
    title="Mean TPOT vs Input Length",
    save_path=figure_dir / "tpot_vs_context_length.png",
)


# -----------------------------
# Figure 4: Output Throughput
# -----------------------------

make_line_plot(
    x=df["input_length"],
    y=df["output_throughput"],
    xlabel="Input Length (tokens)",
    ylabel="Output Throughput (tok/s)",
    title="Output Throughput vs Input Length",
    save_path=figure_dir / "output_throughput_vs_context_length.png",
)


print("\nDone.")
print(
    "CSV saved to:",
    processed_dir / "context_length_summary.csv"
)
print("Figures saved to:", figure_dir)