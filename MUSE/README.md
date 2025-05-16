# MUSE

## Installation
To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate muse
```
## Get the data & original models
You can find the source of data & original model from the [MUSE Bench](https://muse-bench.github.io/)

## Get the unlearned model
Run <kbd style="background-color: #f2f2f2;">unlearn.py</kbd> in the <kbd style="background-color: #f2f2f2;">baselines</kbd> folder. You can enable or disable the information identification process by setting the <kbd style="background-color: #f2f2f2;">flag</kbd> argument in the <kbd style="background-color: #f2f2f2;">unlearn.py</kbd> file. Set it to <code>True</code> to use identification, or <code>False</code> to skip it. You can also update the path to the filtered dataset by modifying the <kbd style="background-color: #f2f2f2;">common_words_file</kbd> argument, or use the default file we provide at <kbd style="background-color: #f2f2f2;">../data/news/raw/forget_common_words_bert.json</kbd>.

### Uwanted information identification
You can find the code of Uwanted information identification process in the <kbd style="background-color: #f2f2f2;">identification_MUSE</kbd> file with bert model

```bash
# news
beta=0.2
python unlearn.py --algo tpo_gdr --model_dir muse-bench/MUSE-News_target --tokenizer_dir meta-llama/Llama-2-7b-hf --data_file ../data/news/raw/forget.txt --retain_data_file ../data/news/raw/retain1.txt --common_words_file ../data/news/raw/forget_common_words_bert.json --out_dir /output/news/tpo_gdr --max_len 2048 --epochs 10 --lr 1e-5 --per_device_batch_size 2 --beta ${beta} --coeff 1 --npo_coeff 0.1

# books
beta=0.2
python unlearn.py --algo tpo_gdr --model_dir muse-bench/MUSE-News_target --tokenizer_dir meta-llama/Llama-2-7b-hf --data_file ../data/books/raw/forget.txt --retain_data_file ../data/books/raw/retain1.txt --common_words_file ../data/books/raw/forget_common_words_bert.json --out_dir /output/books/tpo_gdr --max_len 2048 --epochs 10 --lr 1e-5 --per_device_batch_size 2 --beta ${beta} --coeff 1 --npo_coeff 0.1
```
- `algo`: Unlearning algorithm to run (`simnpo`, `simnpo_gdr`, `ga`, `ga_gdr`, `ga_klr`, `npo`, `npo_gdr`, `npo_klr`, or `tv`).
- `model_dir`: Directory of the target model.
- `tokenizer_dir`: Directory of the tokenizer.
- `data_file`: Forget set.
- `retain_data_file`: Retain set for GDR/KLR regularizations if required by the algorithm.
- `out_dir`: Directory to save the unlearned model (default: `ckpt`).
- `max_len`: Maximum input length (default: 2048).
- `per_device_batch_size`, `epochs`, `lr`: Hyperparameters.

## Evaluate the unlearned model
- To evaluate your unlearned model(s), run `eval.py` from the root of this repository with the following command-line arguments:

- `--model_dirs`: A list of directories containing the unlearned models. These can be either HuggingFace model directories or local storage paths.
- `--names`: A unique name assigned to each unlearned model in `--model_dirs`. The length of `--names` should match the length of `--model_dirs`.
- `--corpus`: The corpus to use for evaluation. Options are `news` or `books`.
- `--out_file`: The name of the output file. The file will be in CSV format, with each row corresponding to an unlearning method from `--model_dirs`, and columns representing the metrics specified by `--metrics`.
- `--tokenizer_dir` (Optional): The directory of the tokenizer. Defaults to `meta-llama/Llama-2-7b-hf`, which is the default tokenizer for LLaMA.
- `--metrics` (Optional): The metrics to evaluate. Options are `verbmem_f` (VerbMem Forget), `privleak` (PrivLeak), `knowmem_f` (KnowMem Forget), and `knowmem_r` (Knowmem Retain, i.e., Utility). Defaults to evaluating all these metrics.
- `--temp_dir` (Optional): The directory for saving intermediate computations. Defaults to `temp`.

- Run the following command with placeholder values:

```python
python eval.py \
--model_dirs "repo/model1" "repo/model2" \
--names "model1" "model2" \
--corpus books \
--out_file "out.csv"
```
