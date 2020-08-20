import hashlib
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .parser import *
from .text8 import read_and_download_text8_data


class DatasetType(Enum):
    TEXT8 = "text8"


BASE_ARGS_EXCEPTIONS = ['self', 'root']


def get_hash_from_args(args: Dict, exceptions=None):
    if exceptions is None:
        exceptions = BASE_ARGS_EXCEPTIONS
    filtered_opt_dict = {k: v for k, v in args.items() if k not in exceptions}
    opt_string = ",".join(("{}={}".format(*i) for i in filtered_opt_dict.items()))
    return hashlib.md5(opt_string.encode('utf-8')).hexdigest()


class ChunckTextDataset(Dataset):
    """
    A simple text dataset that splits a raw texts into chucks of equal number of tokens.
    Tokens can represent characters, subwords or words based on the TokenType.
    Each dataset needs an implementation of the function `read_and_download_data`.
    """

    @property
    def filename(self):
        return f"dataset-{self.id}.tar"

    def __init__(self,
                 dataset_type: DatasetType,
                 root: str = 'data/',
                 token_type: TokenType = TokenType.CHAR,
                 vocabulary_size: Optional[int] = None,
                 sentence_length: int = 180):

        self.root = root
        self.token_type = token_type
        self.dataset_type = dataset_type
        self.id = f"{dataset_type.value}-{get_hash_from_args(locals())}"

        self.space_char = {TokenType.CHAR: "",
                           TokenType.WORD: " ",
                           TokenType.SUB: ""
                           }[self.token_type]

        if not self.restore_if_available():
            # choose the dataset downloader and reader
            read_and_download_data = {
                DatasetType.TEXT8: read_and_download_text8_data
            }[dataset_type]

            # read the raw text dump
            print("(Download and) read the data..")
            raw_text = read_and_download_data(root=root)

            # extract tokens
            print("Tokenizing..")
            tokens = Tokenizer(token_type)(raw_text)

            # build vocabulary
            print("Building vocabulary..")
            self.vocabulary = build_vocabulary_from_tokens(tokens, max=vocabulary_size)

            # encode tokens
            print("Encode tokens..")
            encoded_tokens = encode_tokens(tokens, self.vocabulary)

            # drop the last split
            encoded_tokens = encoded_tokens[:sentence_length * (len(encoded_tokens) // sentence_length)]

            # build an array where each row corresponds to a `split`
            self.data = np.array(encoded_tokens).reshape(-1, sentence_length)

            # save data to file for fast loading
            self.save()

    def __repr__(self):
        return f"`{self.dataset_type.value}` {type(self).__name__}: " \
               f"N = {self.data.shape[0]}, L = {self.data.shape[1]}, " \
               f"V = {len(self.vocabulary)}, size = {self.data.nbytes / 1e6} MB"

    def state_dict(self):
        return {
            'data': self.data,
            'vocabulary': self.vocabulary
        }

    def __getitem__(self, item):
        x = self.data[item]
        return torch.tensor(x)

    def __len__(self):
        return self.data.shape[0]

    def decode_tokens(self, tokens: List[int]):
        return self.space_char.join([self.vocabulary[t] for t in tokens])

    @property
    def path(self):
        return os.path.join(self.root, self.dataset_type.value, self.filename)

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        state = torch.load(self.path)
        self.data = state['data']
        self.vocabulary = state['vocabulary']

    def restore_if_available(self) -> bool:
        if os.path.exists(self.path):
            self.load()
            return True
        else:
            return False


class SubsetDataset():
    """Create a data by indexing a parent dataset with an index"""

    def __init__(self, dataset: Dataset, index: List[int], label: str = "subset"):
        self.dataset = dataset
        self.index = index
        self.label = label

    def __getitem__(self, item):
        return self.dataset[self.index[item]]

    def __len__(self):
        return len(self.index)

    def __repr__(self):
        return f"`{self.label}` {type(self).__name__} dataset with Index of size {len(self.index)}. Parent Dataset:\n{self.dataset.__repr__()}"


def split_dataset(dataset: Dataset,
                  split_ratios: Tuple[float, float, float],
                  shuffle: bool = True) \
        -> Tuple[SubsetDataset, SubsetDataset, SubsetDataset]:
    """
    Split a dataset into train/valid/test splits given the `split_ratios`.
    **warnings** test set = validation set
    :param dataset: dataset to be sliced
    :param train_ratio: ratio of training items
    :param shuffle: shuffle dataset before slicing
    :return: (train, valid, test) datasets
    """
    assert sum(split_ratios) == 1
    assert all([s > 0 and s < 1 for s in split_ratios])
    index = list(range(len(dataset)))
    if shuffle:
        random.shuffle(index)

    # split the dataset index into train/valid/test
    def discrtz(N, ratio):
        return int(ratio * N)

    n_train, n_valid, n_test = map(partial(discrtz, len(dataset)), split_ratios)
    train_index = index[:n_train]
    valid_index = index[n_train:-n_test]
    test_index = index[-n_test:]

    return SubsetDataset(dataset, train_index, label="train"), \
           SubsetDataset(dataset, valid_index, label="valid"), \
           SubsetDataset(dataset, test_index, label="test")
