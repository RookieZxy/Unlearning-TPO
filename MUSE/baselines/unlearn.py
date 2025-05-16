import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)

from baselines import it_unlearn, tv_unlearn, finetune

import argparse
from os.path import basename, dirname, join as pathjoin

CORPUS="books"
PATH = '../data'
FORGET=f"{PATH}/{CORPUS}/raw/forget.txt"
RETAIN=f"{PATH}/{CORPUS}/raw/retain1.txt"

TARGET_DIR="muse-bench/MUSE-Books_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

MAX_LEN=2048
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=2 # 2 GPUs
FT_EPOCHS=10
FT_LR='1e-5'
ALGO = 'spo_gdr' 
GRADIENT_ACCUMULATION_STEPS = 8

def main():
    args = get_args()
    print(args.out_dir)
    if 'npo' in args.out_dir:
        args.out_dir = args.out_dir.replace(args.algo, f'{args.algo}-{args.beta}')
    if 'lpo' in args.out_dir:
        args.out_dir = args.out_dir.replace(args.algo, f'{args.algo}-{args.beta}_pl_coeff-{args.pl_coeff}_npo_coeff-{args.npo_coeff}')

        
    if args.flag:
        if 'bert' in args.common_words_file:
            args.out_dir = args.out_dir.replace(args.algo, f'{args.algo}-bert')
        else:
            args.out_dir = args.out_dir.replace(args.algo, f'{args.algo}-gpt')

    if args.algo == 'kn':
        raise NotImplementedError()

    elif args.algo == 'tv':
        ft_model_dir = pathjoin(dirname(args.out_dir), basename(args.out_dir) + "_ft")
        finetune(
            args.model_dir, args.data_file, ft_model_dir,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir
        )
        tv_unlearn(
            args.model_dir, args.out_dir,
            some_pt_model_dir=args.model_dir,
            some_ft_model_dir=ft_model_dir,
            alpha=args.alpha
        )

    else:
        it_unlearn(
            args.model_dir, args.data_file, args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=args.algo,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            beta=args.beta,
            coeff=args.coeff,
            npo_coeff=args.npo_coeff,
            pl_coeff=args.pl_coeff,
            gamma=args.gamma,
            flag = args.flag,
            common_words_file=args.common_words_file
        )

    return


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    parser.add_argument('--algo', default=ALGO, type=str)
    parser.add_argument(
        '--model_dir', type=str, default=TARGET_DIR,
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default=LLAMA_DIR,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str, default=FORGET,
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        default=f'/data/XZhou/unlearning/results/MUSE/{CORPUS}/{ALGO}',
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )
    parser.add_argument(
        '--max_len', type=int, default=MAX_LEN,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # Gradient ascent & Gradient difference
    parser.add_argument('--per_device_batch_size', type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS)
    
    parser.add_argument(
        '--retain_data_file', type=str, default=RETAIN,
        help="Path to the retain set file. Required if algo is gradient difference (gd)."
    )
    parser.add_argument(
        '--lr', type=float, default=LR,
        help="Learning rate if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help="Number of epochs of training if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )

    # Task vector
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help="Scaling coefficient scales the task vector if algo is task vector (tv)."
    )

    parser.add_argument(
        '--beta', type=float, default=0.2,
        help="for npo"
    )
    
    parser.add_argument(
        '--coeff', type=float, default=1,
        help="for retain loss"
    )

    parser.add_argument(
        '--pl_coeff', type=float, default=0,
        help="for retain loss"
    )

    parser.add_argument(
        '--npo_coeff', type=float, default=0.1,
        help="for forget loss"
    )

    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help="for simnpo"
    )

    parser.add_argument(
        '--flag', type=bool, default=True,
        help="Whether using information indentification"
    )
    parser.add_argument(
        '--common_words_file', type=str, default="../data/news/raw/forget_common_words_bert.json",
        help="Path to common words"
    )
    args = parser.parse_args()

    if args.algo == 'gd':
        # gradient difference. Retain set is required
        assert args.retain_data_file is not None, "Gradient difference selected. Retain set required."

    if args.resume_from_checkpoint:
        assert args.algo not in {'tv'}, "Cannot resume from checkpoint if the method is task vector."

    return args


if __name__ == '__main__':
    main()
