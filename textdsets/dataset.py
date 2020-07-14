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
    filename = "dataset.tar"

    def __init__(self,
                 dataset_type: DatasetType,
                 root: str = 'data/',
                 split_type: TokenType = TokenType.WORD,
                 vocabulary_size: Optional[int] = None,
                 sentence_length: int = 40):
        self.root = root
        self.dataset_type = dataset_type
        self.id = f"{dataset_type.value}-{get_hash_from_args(locals())}"

        if not self.restore_if_available():
            # choose the dataset downloader and reader
            read_and_download_data = {
                DatasetType.TEXT8: read_and_download_text8_data
            }[dataset_type]

            # read the raw text dump
            raw_text = read_and_download_data(root=root)

            # extract tokens
            tokens = Tokenizer(split_type)(raw_text)

            # build vocabulary
            self.vocabulary = build_vocabulary_from_tokens(tokens, max=vocabulary_size)

            # encode tokens
            tokens = encode_tokens(tokens, self.vocabulary)

            # drop the last split
            tokens = tokens[:sentence_length * (len(tokens) // sentence_length)]

            # build an array where each row corresponds to a `split`
            self.data = np.array(tokens).reshape(-1, sentence_length)

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
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]

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


def split_dataset(dataset: Dataset, train_ratio: float, shuffle: bool = True) \
        -> Tuple[SubsetDataset, SubsetDataset, SubsetDataset]:
    """
    Tokenizer a dataset into train/test sets.
    **warnings** test set = validation set
    :param dataset: dataset to be sliced
    :param train_ratio: ratio of training items
    :param shuffle: shuffle dataset before slicing
    :return: (train, valid, test) datasets
    """
    assert train_ratio < 1 and train_ratio > 0
    index = list(range(len(dataset)))
    if shuffle:
        random.shuffle(index)

    n_train = int(train_ratio * len(dataset))
    train_index = index[:n_train]
    test_index = index[n_train:]

    return SubsetDataset(dataset, train_index, label="train"), \
           SubsetDataset(dataset, test_index, label="valid"), \
           SubsetDataset(dataset, test_index, label="test")
