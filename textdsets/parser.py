from collections import Counter
from enum import Enum
from itertools import chain
from typing import *

from .utils import Token


def tokenize_words(text) -> List[str]:
    return text.split()


def tokenize_characters(text) -> List[str]:
    return list(text)


def tokenize_subwords(text) -> List[str]:
    raise NotImplementedError


class TokenType(Enum):
    WORD = "word"
    CHAR = "char"
    SUB = "sub"


class Tokenizer():
    """A small class to tokenize raw `text` into `tokens`"""

    def __init__(self, token_type: TokenType):
        self.split_fn = {
            TokenType.WORD: tokenize_words,
            TokenType.CHAR: tokenize_characters,
            TokenType.SUB: tokenize_subwords
        }[token_type]

    def __call__(self, *args, **kwargs):
        return self.split_fn(*args, **kwargs)


def build_vocabulary_from_tokens(tokens: List[str], max: Optional[int] = None) -> List[str]:
    """Build the list """
    base_tokens = sorted([(t.name, t.value) for t in Token], key=lambda x: [1])
    if max is not None:
        max -= len(base_tokens)
    tokens_count = Counter(tokens).most_common(max)
    return [t[0] for t in chain(base_tokens, tokens_count)]


def encode_tokens(tokens: List[str], vocabulary: List[str]) -> List[int]:
    vocabulary = {v: k for k, v in enumerate(vocabulary)}
    return [vocabulary.get(t, Token.UNK.value) for t in tokens]


def decode_tokens(tokens: List[str], vocabulary: List[str]) -> List[int]:
    vocabulary = {k: v for k, v in enumerate(vocabulary)}
    return [vocabulary.get(t, "<TOKEN_NOT_FOUND>") for t in tokens]
