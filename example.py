import argparse

from textdsets.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=TokenType, default='word', help='split type', choices=list(TokenType))
parser.add_argument('--vocabulary_size', type=int, default=10000, help='max. size of the vocabulary')
args = vars(parser.parse_args())

from textdsets.dataset import ChunckTextDataset, DatasetType

dset = ChunckTextDataset(DatasetType.TEXT8, vocabulary_size=args['vocabulary_size'])

dset_train, dset_valid, dset_test = split_dataset(dset, train_ratio=0.9, shuffle=True)

print(dset)
print(dset_train)