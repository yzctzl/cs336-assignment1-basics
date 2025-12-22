import pickle
from collections.abc import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        # for encode merge, need to use bytes as key to find the order
        self.vocab_inv = {v: k for k, v in vocab.items()}

        # map each merge pair to its rank for efficient lookup
        self.merges = {pair: rank for rank, pair in enumerate(merges)}

        # can't find in None, if special_tokens is None set to empty list
        self.special_tokens = special_tokens if special_tokens else []
        # sort the list to keep longer token is processed first
        self.special_tokens.sort(key=len, reverse=True)

        # cache ids for words to speed up encoding
        self.word_ids: dict[str, list[int]] = {}


    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def _bpe_merge(self, word: str) -> list[int]:
        # process special token
        if word in self.special_tokens:
            return [self.vocab_inv[word.encode("utf8")]]

        # convert word to bytes list
        word_bytes = [bytes([b]) for b in word.encode("utf-8", errors="replace")]

        while len(word_bytes) > 1:
            # find the pair with the minimum rank
            min_rank = float("inf")
            min_pair: tuple[bytes, bytes] | None = None
            for pair in zip(word_bytes[:-1], word_bytes[1:]):
                rank = self.merges.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            # if no pair/rank is found, break while loop
            if min_rank == float("inf"):
                break

            # merge the pair with the minimum rank to word_bytes
            i = 0
            new_word_bytes = []
            while i < len(word_bytes):
                if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i + 1]) == min_pair:
                    new_word_bytes.append(word_bytes[i] + word_bytes[i + 1])
                    i += 2
                else:
                    new_word_bytes.append(word_bytes[i])
                    i += 1
            word_bytes = new_word_bytes

        return [self.vocab_inv[b] for b in word_bytes]


    def _pre_token(self, chunk: str) -> list[list[str]]:
        words_list = []
        # same as pre-tokenization in bpe_train, split with special token
        if self.special_tokens:
            # use capture group to keep sprcial tokens in docs
            split_pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            raw_docs = re.split(split_pattern, chunk)
            docs = [doc for doc in raw_docs if doc]
        else:
            docs = [chunk]

        # to keep consistency with train_bpe, should pre-tokenize the text
        for doc in docs:
            if doc in self.special_tokens:
                words = [doc]
            else:
                words_scanner = COMPILED_PAT.finditer(doc)
                words = [word.group(0) for word in words_scanner]

            words_list.append(words)

        return words_list


    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        words_list = self._pre_token(text)
        for words in words_list:
            for word in words:
                if word not in self.word_ids:
                    self.word_ids[word] = self._bpe_merge(word)
                ids.extend(self.word_ids[word])

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # if call encode_iterable with file, the iterable is a line
        for line in iterable:
            words_list = self._pre_token(line)
            for words in words_list:
                for word in words:
                    if word not in self.word_ids:
                        self.word_ids[word] = self._bpe_merge(word)
                    yield from self.word_ids[word]


    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.vocab[t] for t in tokens).decode("utf-8", errors="replace")
