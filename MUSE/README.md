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

```bash
# news
beta=0.2
python unlearn.py --algo tpo_gdr --model_dir muse-bench/MUSE-News_target --tokenizer_dir meta-llama/Llama-2-7b-hf --data_file ../data/news/raw/forget.txt --retain_data_file ../data/news/raw/retain1.txt --common_words_file ../data/news/raw/forget_common_words_bert.json --out_dir /output/news/tpo_gdr --max_len 2048 --epochs 10 --lr 1e-5 --per_device_batch_size 2 --beta ${beta} --coeff 1 --npo_coeff 0.1

# books
beta=0.2
python unlearn.py --algo tpo_gdr --model_dir muse-bench/MUSE-News_target --tokenizer_dir meta-llama/Llama-2-7b-hf --data_file ../data/books/raw/forget.txt --retain_data_file ../data/books/raw/retain1.txt --common_words_file ../data/books/raw/forget_common_words_bert.json --out_dir /output/books/tpo_gdr --max_len 2048 --epochs 10 --lr 1e-5 --per_device_batch_size 2 --beta ${beta} --coeff 1 --npo_coeff 0.1
```
