# prepare_data.py
import os
import csv
import json
import random
import argparse
import pandas as pd

def build_char_map(samples):
    """Build character to index mapping."""
    chars = set()
    for s in samples:
        for c in s:
            chars.add(c)
    chars = sorted(chars)
    char2i = {c: i+1 for i, c in enumerate(chars)}
    return char2i

def main(args):
    os.makedirs('data', exist_ok=True)
    os.makedirs('python/artifacts', exist_ok=True)

    # Load prefix.csv
    df = pd.read_csv(args.prefix_csv, encoding='utf-8')

    # Combine prefix columns into one column
    prefix_cols = [col for col in df.columns if 'input' in col.lower()]
    df['input'] = df[prefix_cols].fillna('').agg(''.join, axis=1)
    df['target'] = df['target']

    # Keep only necessary columns
    df = df[['input', 'target']]

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split train/test
    test_size = args.test_count if args.test_count else int(0.15 * len(df))
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    # Save train.csv
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8')
    # Save test.csv
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8')

    # Save representation samples (first 200 prefixes)
    with open('data/repr_samples.txt', 'w', encoding='utf-8') as f:
        for x in df['input'][:200]:
            f.write(x + '\n')

    # Build character map
    all_inputs = list(df['input']) + list(df['target'])
    char_map = build_char_map(all_inputs)
    with open('python/artifacts/char_map.json', 'w', encoding='utf-8') as f:
        json.dump(char_map, f, ensure_ascii=False, indent=2)

    # Save labels (unique target words)
    words = sorted(df['target'].unique())
    with open('python/labels.txt', 'w', encoding='utf-8') as f:
        for w in words:
            f.write(w + '\n')

    print(f"Prepared train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_csv', default='../data/training_samples_full.csv',
                        help='Input CSV with columns: prefix_1,...,prefix_n, correct words')
    parser.add_argument('--test_count', type=int, default=None,
                        help='Number of test rows (default 15% of data)')
    args = parser.parse_args()
    main(args)
