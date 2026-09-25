from pathlib import Path
import json
import re
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "results" / "raw"
PROCESSED_DIR = ROOT / "results" / "processed"
FIGURE_DIR = ROOT / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment configuration
# ============================================================

EXPERIMENTS = {
    "queue": {
        "label": "Sequence limit",
        "benchmark": RAW_DIR / "queue.json",
        "server_log": RAW_DIR / "queue_server.log",
        "max_num_seqs": 4,
        "kv_blocks": None,
        "preemptions": 0,
    },

    "kv_512": {
        "label": "512 blocks",
        "benchmark": RAW_DIR / "preempt_512.json",
        "server_log": RAW_DIR / "preempt_512_server.log",
        "max_num_seqs": 8,
        "kv_blocks": 512,
        "preemptions": 1,
    },

    "kv_2048": {
        "label": "2048 blocks",
        "benchmark": RAW_DIR / "preempt_2048.json",
        "server_log": RAW_DIR / "preempt_2048_server.log",
        "max_num_seqs": 8,
        "kv_blocks": 2048,
        "preemptions": 0,
    },
}


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ============================================================
# JSON loading
# ============================================================

def load_json(path: Path):
    """
    Load a vLLM benchmark result.

    Usually the file contains one JSON object.
    If multiple lines exist, use the last valid JSON line.
    """

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in reversed(lines):

            try:
                return json.loads(line)

            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"Could not parse JSON file: {path}"
    )


# ============================================================
# Server-log parsing
# ============================================================

ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-9;]*m"
)

LOG_PATTERN = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*?"
    r"Running:\s*(\d+)\s*reqs,\s*"
    r"Waiting:\s*(\d+)\s*reqs,\s*"
    r"GPU KV cache usage:\s*([\d.]+)%"
)


def parse_server_log(
    path: Path,
    config_name: str,
):
    """
    Extract scheduler statistics from vLLM logs.

    Fields:
        timestamp
        running requests
        waiting requests
        GPU KV-cache utilization
    """

    rows = []

    for line in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        # Remove terminal color codes
        line = ANSI_ESCAPE.sub("", line)

        match = LOG_PATTERN.search(line)

        if not match:
            continue

        rows.append({
            "config": config_name,
            "timestamp": match.group(1),
            "running": int(match.group(2)),
            "waiting": int(match.group(3)),
            "kv_cache_usage_pct": float(
                match.group(4)
            ),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No scheduler statistics found in {path}"
        )

    # ------------------------------------------
    # Convert HH:MM:SS into elapsed seconds
    # ------------------------------------------

    parsed_times = []

    day_offset = 0
    previous_time = None

    for value in df["timestamp"]:

        current_time = datetime.strptime(
            value,
            "%H:%M:%S",
        )

        # Handle midnight rollover
        if (
            previous_time is not None
            and current_time < previous_time
        ):
            day_offset += 1

        current_time += timedelta(
            days=day_offset
        )

        parsed_times.append(
            current_time
        )

        previous_time = current_time

    df["_datetime"] = parsed_times

    # ------------------------------------------
    # Keep only active benchmark period
    # ------------------------------------------

    active = (
        df["running"]
        + df["waiting"]
    ) > 0

    if active.any():

        first_active = active.idxmax()

        last_active = (
            active[::-1].idxmax()
        )

        df = df.loc[
            first_active:last_active
        ].copy()

    start_time = df["_datetime"].iloc[0]

    df["elapsed_s"] = (
        df["_datetime"]
        - start_time
    ).dt.total_seconds()

    df = df.drop(
        columns="_datetime"
    )

    return df


# ============================================================
# Benchmark summary
# ============================================================

summary_rows = []

for config_name, config in EXPERIMENTS.items():

    result = load_json(
        config["benchmark"]
    )

    summary_rows.append({

        "config": config_name,
        "label": config["label"],

        "max_num_seqs":
            config["max_num_seqs"],

        "kv_blocks":
            config["kv_blocks"],

        "preemptions":
            config["preemptions"],

        "duration_s":
            result["duration"],

        "request_throughput":
            result["request_throughput"],

        "output_throughput":
            result["output_throughput"],

        "mean_ttft_ms":
            result["mean_ttft_ms"],

        "p99_ttft_ms":
            result["p99_ttft_ms"],

        "mean_tpot_ms":
            result["mean_tpot_ms"],

        "p99_tpot_ms":
            result["p99_tpot_ms"],

        "mean_itl_ms":
            result["mean_itl_ms"],

        "p99_itl_ms":
            result["p99_itl_ms"],
    })


summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    PROCESSED_DIR
    / "stage8_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Scheduler trace dataset
# ============================================================

trace_frames = []

for config_name, config in EXPERIMENTS.items():

    trace = parse_server_log(
        config["server_log"],
        config_name,
    )

    trace_frames.append(
        trace
    )


trace_df = pd.concat(
    trace_frames,
    ignore_index=True,
)

trace_path = (
    PROCESSED_DIR
    / "stage8_scheduler_trace.csv"
)

trace_df.to_csv(
    trace_path,
    index=False,
)


# ============================================================
# Helpers
# ============================================================

def save_figure(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        bbox_inches="tight",
    )

    plt.close()


def get_trace(config):

    return (
        trace_df[
            trace_df["config"] == config
        ]
        .sort_values("elapsed_s")
        .reset_index(drop=True)
    )


def get_result(config):

    return (
        summary[
            summary["config"] == config
        ]
        .iloc[0]
    )


# ============================================================
# Figure 1
# Scheduler Queue Dynamics
# ============================================================

queue_trace = get_trace(
    "queue"
)

fig, ax = plt.subplots()

ax.plot(
    queue_trace["elapsed_s"],
    queue_trace["running"],
    marker="o",
    linewidth=2.2,
    label="Running requests",
)

ax.plot(
    queue_trace["elapsed_s"],
    queue_trace["waiting"],
    marker="s",
    linewidth=2.2,
    label="Waiting requests",
)

# Scheduler sequence limit
ax.axhline(
    4,
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    color="gray",
)

# Put the limit directly on the chart
ax.text(
    queue_trace["elapsed_s"].max() * 0.98,
    4.08,
    "Sequence limit = 4",
    ha="right",
    va="bottom",
    fontsize=10,
    color="gray",
)

ax.set_title(
    "Scheduler Queue Dynamics under Sequence Limit\n"
    "Concurrency = 8"
)

ax.set_xlabel(
    "Elapsed time (s)"
)

ax.set_ylabel(
    "Number of requests"
)

ax.set_ylim(
    bottom=0,
    top=max(
        queue_trace["running"].max(),
        queue_trace["waiting"].max(),
        4,
    ) + 1
)

ax.legend(
    loc="upper left"
)

save_figure(
    "scheduler_queue_dynamics.png"
)


# ============================================================
# Figure 2
# Request Scheduling under KV-Cache Pressure
# ============================================================

trace_512 = get_trace(
    "kv_512"
)

trace_2048 = get_trace(
    "kv_2048"
)

fig, ax = plt.subplots()

color_512 = COLORS[0]
color_2048 = COLORS[1]


# ------------------------------------------
# 512 blocks
# ------------------------------------------

ax.plot(
    trace_512["elapsed_s"],
    trace_512["running"],
    linewidth=2.2,
    marker="o",
    color=color_512,
    label="512 blocks — Running",
)

ax.plot(
    trace_512["elapsed_s"],
    trace_512["waiting"],
    linewidth=2.0,
    linestyle="--",
    marker="o",
    color=color_512,
    label="512 blocks — Waiting",
)


# ------------------------------------------
# 2048 blocks
# ------------------------------------------

ax.plot(
    trace_2048["elapsed_s"],
    trace_2048["running"],
    linewidth=2.2,
    marker="s",
    color=color_2048,
    label="2048 blocks — Running",
)

ax.plot(
    trace_2048["elapsed_s"],
    trace_2048["waiting"],
    linewidth=2.0,
    linestyle="--",
    marker="s",
    color=color_2048,
    label="2048 blocks — Waiting",
)


ax.set_title(
    "Request Scheduling under Different KV-Cache Capacities\n"
    "3072-token input, 512-token output, concurrency = 8"
)

ax.set_xlabel(
    "Elapsed time (s)"
)

ax.set_ylabel(
    "Number of requests"
)

ax.set_ylim(
    0,
    9,
)

ax.legend(
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    frameon=True,
)

save_figure(
    "request_scheduling_kv_capacity.png"
)


# ============================================================
# Figure 3
# KV Cache Utilization
# ============================================================

fig, ax = plt.subplots()

ax.plot(
    trace_512["elapsed_s"],
    trace_512["kv_cache_usage_pct"],
    linewidth=2.3,
    marker="o",
    color=color_512,
    label="512 blocks",
)

ax.plot(
    trace_2048["elapsed_s"],
    trace_2048["kv_cache_usage_pct"],
    linewidth=2.3,
    marker="s",
    color=color_2048,
    label="2048 blocks",
)

ax.set_title(
    "KV-Cache Utilization over Time\n"
    "Similar utilization percentages can represent very different absolute capacities"
)

ax.set_xlabel(
    "Elapsed time (s)"
)

ax.set_ylabel(
    "KV-cache utilization (%)"
)

ax.set_ylim(
    0,
    100,
)

ax.legend(
    loc="upper right"
)

save_figure(
    "kv_cache_utilization.png"
)


# ============================================================
# Figure 4
# TTFT Comparison
# ============================================================

result_512 = get_result(
    "kv_512"
)

result_2048 = get_result(
    "kv_2048"
)


metrics = [
    "Mean TTFT",
    "P99 TTFT",
]

values_512 = [
    result_512["mean_ttft_ms"] / 1000,
    result_512["p99_ttft_ms"] / 1000,
]

values_2048 = [
    result_2048["mean_ttft_ms"] / 1000,
    result_2048["p99_ttft_ms"] / 1000,
]


x = np.arange(
    len(metrics)
)

width = 0.34


fig, ax = plt.subplots()


bars_512 = ax.bar(
    x - width / 2,
    values_512,
    width,
    label="512 blocks",
)

bars_2048 = ax.bar(
    x + width / 2,
    values_2048,
    width,
    label="2048 blocks",
)


ax.set_title(
    "KV-Cache Pressure Dramatically Increases TTFT\n"
    "512 blocks triggered preemption; 2048 blocks did not"
)

ax.set_ylabel(
    "Time to first token (s)"
)

ax.set_xticks(x)

ax.set_xticklabels(
    metrics
)

ax.legend(
    loc="upper right"
)


# ------------------------------------------
# Value labels
# ------------------------------------------

for bars in [
    bars_512,
    bars_2048,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            height,

            f"{height:.2f}s",

            ha="center",
            va="bottom",
            fontsize=10,
        )


save_figure(
    "ttft_kv_capacity.png"
)


# ============================================================
# Figure 5
# Throughput vs Mean TTFT
# ============================================================

fig, ax = plt.subplots()


points = [

    {
        "label": "512 blocks",

        "ttft_s":
            result_512[
                "mean_ttft_ms"
            ] / 1000,

        "throughput":
            result_512[
                "output_throughput"
            ],

        "preemptions":
            int(
                result_512[
                    "preemptions"
                ]
            ),

        "marker": "o",

        "color": color_512,
    },

    {
        "label": "2048 blocks",

        "ttft_s":
            result_2048[
                "mean_ttft_ms"
            ] / 1000,

        "throughput":
            result_2048[
                "output_throughput"
            ],

        "preemptions":
            int(
                result_2048[
                    "preemptions"
                ]
            ),

        "marker": "s",

        "color": color_2048,
    },
]


for point in points:

    ax.scatter(
        point["ttft_s"],
        point["throughput"],
        s=150,
        marker=point["marker"],
        color=point["color"],
        zorder=3,
    )

    # Put labels in different directions
    # so they do not leave the plot area.
    if point["label"] == "2048 blocks":

        offset = (
            12,
            -18,
        )

        vertical_alignment = "top"

    else:

        offset = (
            -8,
            12,
        )

        vertical_alignment = "bottom"


    ax.annotate(

        (
            f'{point["label"]}\n'
            f'Preemptions = '
            f'{point["preemptions"]}'
        ),

        (
            point["ttft_s"],
            point["throughput"],
        ),

        xytext=offset,

        textcoords="offset points",

        fontsize=10,

        va=vertical_alignment,

        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )


ax.set_title(
    "Throughput–Latency Impact of KV-Cache Capacity\n"
    "More KV capacity enables higher throughput and substantially lower TTFT"
)

ax.set_xlabel(
    "Mean TTFT (s)"
)

ax.set_ylabel(
    "Output throughput (tokens/s)"
)


# ------------------------------------------
# Give annotations extra space
# ------------------------------------------

x_max = max(
    point["ttft_s"]
    for point in points
)

y_max = max(
    point["throughput"]
    for point in points
)


ax.set_xlim(
    0,
    x_max * 1.08,
)

ax.set_ylim(
    0,
    y_max * 1.15,
)


# ------------------------------------------
# Ideal direction
# ------------------------------------------

ax.text(
    0.02,
    0.88,
    "Better ↖",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
)


save_figure(
    "throughput_vs_ttft.png"
)


# ============================================================
# Console summary
# ============================================================

print()

print(
    "Stage 8 plots generated successfully."
)

print()

print(
    "Processed data:"
)

print(
    f"  {summary_path}"
)

print(
    f"  {trace_path}"
)

print()

print(
    "Figures:"
)


figure_names = [

    "scheduler_queue_dynamics.png",

    "request_scheduling_kv_capacity.png",

    "kv_cache_utilization.png",

    "ttft_kv_capacity.png",

    "throughput_vs_ttft.png",
]


for filename in figure_names:

    print(
        f"  {FIGURE_DIR / filename}"
    )


print()

print(
    "Benchmark summary:"
)


print(

    summary[
        [
            "label",

            "output_throughput",

            "mean_ttft_ms",

            "p99_ttft_ms",

            "mean_tpot_ms",

            "p99_itl_ms",

            "preemptions",
        ]
    ].to_string(
        index=False
    )
)