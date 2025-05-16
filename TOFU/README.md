# TOFU

## Installation
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## Get the finetuned model
You can add the newer models in the <kbd style="background-color: #f2f2f2;">model_config.yaml</kbd> file. Finetuning can be done as following command:
```bash
master_port=18765
split=full
model=Llama2-7b
lr=1e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Uwanted information identification
You can find the code of information identification process in the <kbd style="background-color: #f2f2f2;">identification_TOFU</kbd> file with bert model
We also provide the full instruction of information identification process with GPT-4o in <kbd style="background-color: #f2f2f2;">Instruction for Identifying Important Words (TOFU)</kbd> file

## Get the unlearned model
- Ensure that the <kbd style="background-color: #f2f2f2;">model_path</kbd> is correctly set in the <kbd style="background-color: #f2f2f2;">forget.yaml</kbd> configuration file.
- You can also modify the <kbd style="background-color: #f2f2f2;">save_dir</kbd> to change the path where the unlearned model will be saved.
- The <kbd style="background-color: #f2f2f2;">fill_mask</kbd> field in <kbd style="background-color: #f2f2f2;">forget.yaml</kbd> controls whether the unlearning process leverages our unwanted information identification module. The default value is True.
-  You can choose the model used for the identification process by setting the <kbd style="background-color: #f2f2f2;">classifier</kbd> field in the same file. Available options are: 'gpt' and 'bert'
-  To unlearn a model on a forget set, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```
When using **Targeted Preference Optimization (TPO)** for unlearning, the following `β` values achieve the best forget quality across different models and TOFU benchmark settings:
| Model         | Forget 01 | Forget 05 | Forget 10 |
|---------------|-----------|-----------|-----------|
| LLaMa-2 7B     | 0.32      | 0.32      | 0.23      |
| LLaMa-3.2 3B   | 0.30      | 0.27      | 0.19      |

- Once the unlearning process is complete, the results will be saved in <kbd style="background-color: #f2f2f2;">${save_dir}/checkpoint/aggregate_stat.txt</kbd>.







