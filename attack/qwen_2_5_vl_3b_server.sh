#!/bin/bash

python3 server.py \
    --lvlm_name=qwen2_5_vl_3b \
    --llm_name=qwen2_5_7b \
    --llm_7b_name=qwen2_5_7b \
    --load_clip \
    --load_sd \
    --load_sem_relevance \
    --port=8000
