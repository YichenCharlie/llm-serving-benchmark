# LLM Serving Performance Benchmark

A hands-on project for studying LLM inference and serving performance using vLLM.

## Current Status

Stage 1 completed:

- Deployed Qwen2.5-3B-Instruct on an NVIDIA RTX 3090
- Set up vLLM 0.11.2 with PyTorch 2.9.0 and CUDA 12.8
- Started an OpenAI-compatible vLLM API server
- Successfully sent chat completion requests through HTTP API

## Environment

- GPU: NVIDIA GeForce RTX 3090 24GB
- OS: Ubuntu 22.04
- Python: 3.12
- PyTorch: 2.9.0+cu128
- CUDA: 12.8
- vLLM: 0.11.2
- Model: Qwen2.5-3B-Instruct

## Roadmap

- [x] Stage 1: Model deployment and API serving
- [x] Stage 2: Baseline benchmark
- [x] Stage 3: Request load / concurrency experiments
- [ ] Stage 4: Context length experiments
- [ ] Stage 5: Performance analysis and visualization
- [ ] Stage 6: Prefix caching
- [ ] Stage 7: Chunked prefill
