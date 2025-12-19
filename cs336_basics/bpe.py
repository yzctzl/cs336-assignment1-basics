import os
from collections import Counter, defaultdict

from pretokenization import parallelize_pre_tokenizer


def merge(
    freq_table: Counter[tuple[bytes, ...]],
    stats: defaultdict[tuple[bytes, bytes], int],
    pair: tuple[bytes, bytes]
) -> tuple[Counter, defaultdict]:
    """
    """
    # save the list of pair (original key, replace key)
    replace = []
    p1, p2 = pair
    p = p1 + p2

    for word, freq in freq_table.items():
        if p1 in word and p2 in word:
            new_word: list[bytes] = []
            word_len = len(word)
            for i in range(word_len):
                if i < word_len - 1 and word[i] == p1 and word[i + 1] == p2:
                    stats[pair] -= freq
                    if i != 0:
                        stats[(word[i - 1], p)] += freq
                    if i + 2 != word_len - 1:
                        stats[(p, word[i + 2])] += freq
                    new_word.append(p)
                else:
                    new_word.append(word[i])
            if p in new_word:
                replace.append((word, new_word))

    assert stats[pair] == 0, "算法错了？"
    stats.pop(pair)
    for old, new in replace:
        freq_table[new] = freq_table[old]
        freq_table.pop(old)

    return freq_table, stats


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # initialize our vocabulary with our special token <|endoftext|> and the 256 byte values
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in zip(range(len(special_tokens)), special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    cur_vocab_size = len(vocab)
    assert cur_vocab_size < vocab_size, "the vocab_size is too small, the minimal is 256 + len(special_tokens)"

    # BPE merges. Each list item is a tuple of bytes (<token1>, <token2>)
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as file:
        stats: dict[tuple[bytes, bytes], int] = defaultdict(int)
        freq_table = parallelize_pre_tokenizer(file)

        for key in freq_table:
            if len(key) > 1:
                for b1, b2 in zip(key[:-1], key[1:]):
                    stats[(b1, b2)] += 1

        while cur_vocab_size <= vocab_size:
            pair = max(stats, key=lambda p: (stats[p], p))

            vocab[cur_vocab_size] = b"".join(pair)
            merges.append(pair)
            cur_vocab_size += 1

            freq_table, stats = merge(freq_table, stats, pair)

    return vocab, merges
