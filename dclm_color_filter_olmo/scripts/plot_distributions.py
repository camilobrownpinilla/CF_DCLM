import argparse
import matplotlib.pyplot as plt
import json
import os
import numpy as np

from pathlib import Path

def plot(score_path):
    data = []
    with open(score_path, 'r') as file:
        for line in file:
            score = json.loads(line)['score'][0]
            data.append(score)

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "distributions"))
    model_name = f'{Path(score_path).parent.parent.parent.stem}_{Path(score_path).parent.parent.stem}'
    os.makedirs(script_path, exist_ok=True)
    
    # Freedman-Diaconis method to set bin size
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    bins = int((max(data) - min(data)) / bin_width)
    bins = max(bins, 1)
    plt.title(f'{model_name} Score Distribution')
    plt.hist(data, bins=bins, color='#e6294d', edgecolor='none')
    
    # Draw vertical line at cutoff for top 8 billion scores
    cutoff_count = 8_000_000_000 // 2048
    if len(data) >= cutoff_count:
        sorted_data = sorted(data)
        cutoff_index = len(data) - cutoff_count
        cutoff_score = sorted_data[cutoff_index]
        plt.axvline(x=cutoff_score, color='black', linestyle='--', label='Top 8B cutoff')
        plt.legend()
        
    plt.savefig(f'{script_path}/{model_name}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the distribution of scores')
    parser.add_argument('path', type=str, help='Path to combined and sorted scores')

    args = parser.parse_args()
    plot(args.path)