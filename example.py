import argparse
import os
from textdsets.dataset import *
from textdsets.text8 import *

def logging_sep(char = "-"):
    return os.get_terminal_size().columns * char

parser = argparse.ArgumentParser()
parser.add_argument('--token_type', type=TokenType, default='char', help='split type', choices=list(TokenType))
parser.add_argument('--vocabulary_size', type=int, default=None, help='max. size of the vocabulary')
parser.add_argument('--sentence_length', type=int, default=180, help='number of tokens in each sentence')
args = vars(parser.parse_args())


dset = ChunckTextDataset(DatasetType.TEXT8, **args)
dset_train, dset_valid, dset_test = split_dataset(dset, split_ratios=TEXT8_SPLITS_RATIOS, shuffle=False)

print(logging_sep("="))
print(dset)
print(logging_sep("="))
print(dset_train)
print(logging_sep("="))
print("Text8 Dataset: Number of Tokens:")
print(f"Warning: in order to keep sentences of length {args['sentence_length']} without padding, "
      f"resulting number of tokens in each split may be slightly less than the "
      f"original characters splits 90M/5M/5M.")
print(f"\t train: {sum([len(x) for x in dset_train]):.6E}")
print(f"\t valid: {sum([len(x) for x in dset_valid]):.6E}")
print(f"\t test: {sum([len(x) for x in dset_test]):.6E}")
print(logging_sep("="))
print(f"Samples:")
for k in range(10):
    print(f"[{k+1}] `{''.join([dset.vocabulary[t] for t in dset_train[k]])}`")