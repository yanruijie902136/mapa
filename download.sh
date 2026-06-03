#!/bin/bash

: "${HF_TOKEN:?Set HF_TOKEN before running this script}"

hf download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir models/Qwen_Qwen2.5-VL-3B-Instruct \
    --max-workers 8

hf download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir models/Qwen_Qwen3-VL-4B-Instruct \
    --max-workers 8

hf download Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/Qwen_Qwen2.5-7B-Instruct \
    --max-workers 8

hf download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/meta-llama_Llama-3.1-8B-Instruct \
    --max-workers 8

hf download cais/HarmBench-Mistral-7b-val-cls \
    --local-dir models/cais_HarmBench-Mistral-7b-val-cls \
    --max-workers 8

hf download stabilityai/stable-diffusion-3.5-medium \
    --local-dir models/stable_diffusion_3.5_medium \
    --max-workers 8

hf download openai/clip-vit-base-patch32 \
    --local-dir models/clip_vit_base_patch32 \
    --max-workers 8

hf download princeton-nlp/sup-simcse-roberta-large \
    --local-dir models/sup_simcse_roberta_large \
    --max-workers 8
