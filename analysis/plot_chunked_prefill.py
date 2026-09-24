import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


result_dir = Path("results/raw")
processed_dir = Path("results/processed")
figure_dir = Path("figures")


files = {
    "OFF": "chunked_prefill_off.json",
    "ON / 4096": "chunked_prefill_on.json",
    "ON / 2048": "chunked_prefill_2048.json",
    "ON / 1024": "chunked_prefill_1024.json",
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
            "mean_itl_ms": data["mean_itl_ms"],
            "p99_itl_ms": data["p99_itl_ms"],
            "mean_tpot_ms": data["mean_tpot_ms"],
            "p99_tpot_ms": data["p99_tpot_ms"],
        }
    )


df = pd.DataFrame(rows)

print("Raw summary:")
print(df)

df.to_csv(
    processed_dir / "chunked_prefill_summary.csv",
    index=False,
)


# ============================================================
# Figure 1: Normalized Multi-Metric Line Plot
# ============================================================

# Use Chunked Prefill OFF as the 100% baseline.
baseline = df.iloc[0]

normalized_df = pd.DataFrame(
    {
        "condition": df["condition"],
        "Output Throughput": (
            df["output_throughput"] / baseline["output_throughput"] * 100
        ),
        "Mean TTFT": (
            df["mean_ttft_ms"] / baseline["mean_ttft_ms"] * 100
        ),
        "P99 TTFT": (
            df["p99_ttft_ms"] / baseline["p99_ttft_ms"] * 100
        ),
        "Mean ITL": (
            df["mean_itl_ms"] / baseline["mean_itl_ms"] * 100
        ),
        "P99 ITL": (
            df["p99_itl_ms"] / baseline["p99_itl_ms"] * 100
        ),
    }
)

print("\nNormalized metrics (OFF = 100%):")
print(normalized_df)


plt.figure(figsize=(8, 5.2))

x = range(len(normalized_df))

metrics = [
    "Output Throughput",
    "Mean TTFT",
    "P99 TTFT",
    "Mean ITL",
    "P99 ITL",
]

for metric in metrics:
    plt.plot(
        x,
        normalized_df[metric],
        marker="o",
        linewidth=2,
        markersize=7,
        label=metric,
    )

plt.axhline(
    y=100,
    linestyle="--",
    linewidth=1.5,
    alpha=0.6,
)

plt.xlabel("Chunked Prefill Configuration", fontsize=12)
plt.ylabel("Relative Metric (% of OFF baseline)", fontsize=12)

plt.title(
    "Normalized Performance Across Chunked Prefill Configurations",
    fontsize=14,
    pad=12,
)

plt.xticks(
    x,
    normalized_df["condition"],
    fontsize=11,
)

plt.yticks(fontsize=11)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend(
    fontsize=9,
)

plt.tight_layout()

plt.savefig(
    figure_dir / "chunked_prefill_normalized_metrics.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 2: Throughput vs P99 ITL Trade-off Scatter Plot
# ============================================================

plt.figure(figsize=(8, 5.2))

plt.scatter(
    df["p99_itl_ms"],
    df["output_throughput"],
    s=90,
)

for _, row in df.iterrows():
    plt.annotate(
        row["condition"],
        (
            row["p99_itl_ms"],
            row["output_throughput"],
        ),
        xytext=(7, 6),
        textcoords="offset points",
        fontsize=10,
    )

plt.xlabel(
    "P99 ITL (ms, lower is better)",
    fontsize=12,
)

plt.ylabel(
    "Output Throughput (tok/s, higher is better)",
    fontsize=12,
)

plt.title(
    "Throughput vs Decode Tail Latency",
    fontsize=14,
    pad=12,
)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    figure_dir / "throughput_vs_p99_itl.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 3: P99 ITL vs Token Budget
# ============================================================

on_df = df[df["condition"] != "OFF"].copy()

token_budgets = [4096, 2048, 1024]

on_df["token_budget"] = token_budgets


plt.figure(figsize=(8, 5.2))

plt.plot(
    on_df["token_budget"],
    on_df["p99_itl_ms"],
    marker="o",
    linewidth=2,
    markersize=7,
    label="Chunked Prefill ON",
)

# OFF baseline for reference
plt.axhline(
    y=baseline["p99_itl_ms"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Chunked Prefill OFF baseline",
)

plt.xlabel(
    "max_num_batched_tokens",
    fontsize=12,
)

plt.ylabel(
    "P99 ITL (ms)",
    fontsize=12,
)

plt.title(
    "P99 ITL vs Scheduler Token Budget",
    fontsize=14,
    pad=12,
)

plt.xticks(
    token_budgets,
    fontsize=11,
)

plt.yticks(fontsize=11)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend(fontsize=10)

plt.tight_layout()

plt.savefig(
    figure_dir / "p99_itl_vs_token_budget.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


print("\nGenerated figures:")
print("1. figures/chunked_prefill_normalized_metrics.png")
print("2. figures/throughput_vs_p99_itl.png")
print("3. figures/p99_itl_vs_token_budget.png")