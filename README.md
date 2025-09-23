# MAPA: Multi-Turn Adaptive Prompting Attack on Large Vision-Language Models

## How to run the project
1. Download the models from Hugging Face and save them under the ```models``` folder. The mapping between the model name and model folder name is conducted by the ```model_name_to_path_type``` function in ```attack/hf_models/utils.py```.
2. Create a Python virtual environment and enable it.
3. Go into the ```attack``` folder by ```cd attack```.
4. Install the dependencies according to ```requirements.txt```.
6. Run the model server. For example, ```python server.py --lvlm_name=llava --load_clip --load_sd --load_sem_relevance --port=8000```. There are three options for lvlm_name: "llava", "qwen" and "llama".
7. Run the attack. For example, ```python main.py --dataset=harmbench --task_i_start_from=0 --num_tasks=60 --experiment_name=llava_test --port=8000```. Possible options of the ```dataset``` argument include "harmbench", "harmbench_tiny" and "jailbreakbench".
