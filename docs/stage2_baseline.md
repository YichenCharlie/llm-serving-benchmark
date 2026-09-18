# Stage 2: Baseline Benchmark

## Stage Goal

Stage 2 的目标是测试 LLM serving system
在固定条件下的基础性能，并建立后续实验的 baseline。

实验目标：

> 给 vLLM 连续发送一批固定为 512-token 输入、128-token 输出的 synthetic
> requests，测量这套 serving system 在单并发条件下的基础性能。

## Experiment Environment

Hardware: - GPU: NVIDIA GeForce RTX 3090 24GB

Software: - vLLM: 0.11.2 - PyTorch: 2.9.0+cu128 - CUDA: 12.8

Model: - Qwen2.5-3B-Instruct

## Workload Design

Dataset: - random synthetic dataset

Configuration:

  Parameter             Value
  --------------------- ------------
  Input length          512 tokens
  Output length         128 tokens
  Number of requests    100
  Maximum concurrency   1
  Request rate          inf

## Baseline Result

  Metric                    Value
  ------------------------- --------------
  Request throughput        0.799 req/s
  Output token throughput   102.32 tok/s
  Total token throughput    511.62 tok/s
  Mean TTFT                 62.64 ms
  P99 TTFT                  72.42 ms
  Mean TPOT                 9.35 ms
  P99 TPOT                  9.49 ms

## Metric Explanation

TTFT: 用户发送请求后，到第一个 token 返回的时间。

TPOT: 第一个 token 后，每生成一个 token 的平均时间。

Throughput: 系统单位时间处理能力。

P99: 衡量尾延迟，表示 99% 请求低于该延迟。

## Observation

单并发条件下： - 没有多个请求竞争 - TTFT 较低 - TPOT 稳定 - Tail latency
较小

该 baseline 将作为后续 concurrency scaling、context length analysis 和
prefix caching 实验的比较基准。
