# Environment Setup

This document records the environment used for the **LLM Serving Performance Benchmark with vLLM** project and provides a reproducible setup procedure.

## 1. Reference Environment

The experiments in this repository were conducted with:

| Component | Version / Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB |
| OS | Ubuntu 22.04 |
| Python | 3.12 |
| PyTorch | 2.9.0+cu128 |
| CUDA | 12.8 |
| vLLM | 0.11.2 |
| Model | Qwen2.5-3B-Instruct |

The project was developed and benchmarked on a single NVIDIA GPU.

---

## 2. Clone the Repository

```bash
git clone https://github.com/YichenCharlie/llm-serving-benchmark.git
cd llm-serving-benchmark
```

---

## 3. Check the GPU Environment

Before installing Python packages, confirm that the NVIDIA driver and GPU are available:

```bash
nvidia-smi
```

The original experiments used an NVIDIA RTX 3090 24GB.

---

## 4. Create a Python Virtual Environment

Python 3.12 was used for the original experiments.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Check the Python version:

```bash
python --version
```

Expected major/minor version:

```text
Python 3.12
```

---

## 5. Install Python Dependencies

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The requirements file includes the CUDA 12.8 PyTorch index and pins the two core runtime components used in the experiments:

```text
torch==2.9.0
vllm==0.11.2
```

The remaining packages are used for data processing, plotting, HTTP requests, model download, and benchmark utilities.

---

## 6. Verify the Installation

Verify PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

The original environment reported approximately:

```text
PyTorch: 2.9.0+cu128
CUDA: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

Verify vLLM:

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
```

Expected:

```text
vLLM: 0.11.2
```

---

## 7. Download the Model

The original project used:

```text
Qwen2.5-3B-Instruct
```

The model was downloaded through ModelScope and stored locally.

Example:

```bash
modelscope download \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local_dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

If a different local path is used, update the model path in the serving and benchmark commands accordingly.

The original model path was:

```text
/root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

---

## 8. Start the vLLM Server

The repository provides:

```text
scripts/start_server.sh
```

The base server configuration used in the later experiments is:

```bash
vllm serve /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  "$@"
```

Start the server:

```bash
./scripts/start_server.sh
```

Additional vLLM arguments can be appended directly.

Example:

```bash
./scripts/start_server.sh \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching
```

---

## 9. Verify the Server

Check whether the server is responding:

```bash
curl http://localhost:8000/v1/models
```

The metrics endpoint can also be inspected:

```bash
curl http://localhost:8000/metrics
```

During this project, `/metrics` was used to verify settings such as:

- Prefix Caching
- KV-cache configuration
- GPU block count
- Preemption counters

---

## 10. Run a Smoke Benchmark

After the server is ready, run a small benchmark before reproducing the full experiments:

```bash
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000 \
  --model /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 10 \
  --request-rate inf \
  --max-concurrency 1 \
  --ignore-eos
```

If all requests complete successfully, the serving environment is ready.

---

## 11. Reproduce the Project Experiments

Detailed commands, workload settings, results, and analysis are stored in:

```text
docs/
├── stage1_setup.md
├── stage2_baseline.md
├── stage3_concurrency.md
├── stage4_context_length.md
├── stage6_prefix_caching.md
├── stage7_chunked_prefill.md
└── stage8_scheduler_preemption.md
```

The repository also contains:

```text
results/raw/         Raw benchmark outputs and server logs
results/processed/   Processed CSV results
analysis/            Python analysis / plotting scripts
figures/             Generated experiment figures
```

---

## 12. Environment Verification Checklist

Before running the full benchmarks, verify:

```text
[ ] NVIDIA GPU visible through nvidia-smi
[ ] Python 3.12
[ ] torch == 2.9.0 (+cu128)
[ ] torch.cuda.is_available() == True
[ ] vLLM == 0.11.2
[ ] Qwen2.5-3B-Instruct downloaded locally
[ ] vLLM server starts successfully
[ ] /v1/models responds
[ ] /metrics responds
[ ] smoke benchmark completes successfully
```

---

## Notes

This repository records a controlled ML Systems study rather than a production deployment.

The numerical benchmark results in the repository were collected on the original RTX 3090 environment. Re-running the same workloads on a different GPU, driver, model, or software stack may produce different absolute performance values.
