import json
import numpy as np
import subprocess
import argparse
import os

from pathlib import Path
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def examine(json_path, amount, num_examples):
    total_lines = count_lines_wc(json_path)
    if type(amount) == float:  # percentage of lines
        amount = int(amount * total_lines)
    os.makedirs('./examples', exist_ok=True)
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
    out_path = f'{script_path}/{Path(json_path).parent.parent.parent.stem}_{Path(json_path).parent.parent.stem}_top_and_bottom_{amount}.txt'

    if num_examples:
        assert num_examples <= amount, 'Num examples exceeds amount of lines examined'
        top_range = range(0, amount)[-num_examples:]
        bottom_range = range(total_lines - amount, total_lines)[:num_examples]
    else:
        top_range, bottom_range = range(0, amount), range(total_lines - amount, total_lines)
    with open(out_path, 'w') as out:
        with open(json_path, 'r') as file:
            # Process only top and bottom examples:
            for i, line in enumerate(file):
                if i in top_range:
                    if i == 0 or i == top_range[0]:
                        out.write(f"{'*' * 30}\033[32mTop {amount} Examples\033[0m{'*' * 30}\n{'#' * 80}\n\n\n")
                    out.write(examine_line(line))
                elif i in bottom_range:
                    if i == total_lines - amount:
                        out.write(f"{'*' * 30}\033[31mBottom {amount} Examples\033[0m{'*' * 30}\n{'#' * 80}\n\n\n")
                    out.write(examine_line(line))
    return 0

def read_tokens(file_path, idx_range):
    start, end = idx_range
    data = list(np.memmap(file_path, dtype='uint16')[start:end])
    text = TOKENIZER.decode(data)
    return text

def count_lines_wc(file_path):
    return int(subprocess.check_output(['wc', '-l', file_path]).split()[0])

def examine_line(line):
    line = json.loads(line)
    score, metadata = f"{(line['score'][0]):.04f}", line['metadata']
    path, idx_range = metadata['path'], metadata['memmap_idx_range']
    text = read_tokens(path, idx_range)
    formatted_line = f'Score: {score}\n\tText: {text}\n\n'
    return formatted_line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to combined score json')
    parser.add_argument('amount', help='How many lines to examine')
    parser.add_argument('--num-examples', default=None, type=int, help='Number of top/bottom examples to print. If none, print all')
    args = parser.parse_args()

    args.amount = float(args.amount) if '.' in args.amount else int(args.amount)
    examine(args.path, args.amount, args.num_examples)