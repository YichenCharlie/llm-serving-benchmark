#!/bin/bash

vllm serve /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
