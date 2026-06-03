#/bin/bash

python3 main.py \
    --dataset=harmbench \
    --task_i_start_from=0 \
    --num_tasks=60 \
    --experiment_name=qwen2_5_vl_3b_harmbench \
    --port=8000 \
    --target_model_name=qwen2_5_vl_3b \
    --attacker_model_name=qwen2_5_7b \
    --inner_judge_model_name=llama3_1_8b \
    --final_judge_model_name=harmbench_mistral_cls \
    --num_attempts=1 \
    --rounds=10

python3 main.py \
    --dataset=jailbreakbench \
    --task_i_start_from=0 \
    --num_tasks=60 \
    --experiment_name=qwen2_5_vl_3b_jailbreakbench \
    --port=8000 \
    --target_model_name=qwen2_5_vl_3b \
    --attacker_model_name=qwen2_5_7b \
    --inner_judge_model_name=llama3_1_8b \
    --final_judge_model_name=harmbench_mistral_cls \
    --num_attempts=1 \
    --rounds=10
