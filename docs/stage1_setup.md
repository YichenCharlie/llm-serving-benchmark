# Stage 1: Environment Setup and vLLM Deployment

## Stage Goal

Stage 1 的目标是搭建完整的 LLM serving 环境，并成功在 GPU
服务器上部署一个开源大语言模型。

最终完成： - 配置 GPU 推理环境 - 创建 Python virtual environment - 使用
vLLM 部署 Qwen2.5-3B-Instruct - 通过 OpenAI-compatible API
成功发送请求并获得模型回复

## Hardware Environment

-   Cloud Platform: AutoDL
-   GPU: NVIDIA GeForce RTX 3090 24GB

## Software Environment

-   OS: Linux
-   Python: 3.12
-   PyTorch: 2.9.0+cu128
-   CUDA: 12.8
-   vLLM: 0.11.2

## Python Virtual Environment (venv)

使用 Python virtual environment 隔离项目依赖。

创建：

``` bash
python -m venv .venv
```

进入：

``` bash
source .venv/bin/activate
```

退出：

``` bash
deactivate
```

## tmux Usage

tmux 用于保持远程服务器上的长期运行任务。

创建：

``` bash
tmux new -s vllm
```

查看：

``` bash
tmux ls
```

进入：

``` bash
tmux attach -t vllm
```

退出但保持运行： Ctrl + B，然后按 D

## Model Selection

模型： Qwen2.5-3B-Instruct

3B 表示约 30 亿参数。

选择原因： - 开源 - 支持 instruction tuning - 可以在 RTX 3090 单卡运行 -
适合作为 LLM serving 学习对象

## ModelScope

ModelScope 用于模型存储、下载和管理。主要是hugging face我怕国内的服务器连不上，国内源不稳定。

流程：

Model Repository → Download Model Weights → Local Storage → vLLM Load
Model → GPU Inference

## vLLM Introduction

vLLM 是用于 LLM serving 的高性能推理框架。

主要优化： - KV Cache management - Continuous batching - Request
scheduling - Memory efficiency

## Stage 1 Result

成功完成： - AutoDL GPU 环境配置 - venv 环境创建 - vLLM 安装 -
Qwen2.5-3B-Instruct 部署 - API 请求测试

Stage 1 completed.
