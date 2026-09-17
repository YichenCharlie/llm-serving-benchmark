#!/bin/bash

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Explain KV cache in one sentence."
      }
    ],
    "max_tokens": 64
  }'
