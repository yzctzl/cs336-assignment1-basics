import os
from collections import Counter, defaultdict

from .pretokenization import parallelize_pre_tokenizer


def merge(
    freq_table: Counter[tuple[bytes, ...]],
    stats: defaultdict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair: tuple[bytes, bytes]
) -> tuple[Counter, defaultdict, dict]:
    """
    Given a frequency table, a stats table, a pair_to_words table, and a pair,
    merge the pair in the frequency table and update the stats table and pair_to_words table.

    Args:
        freq_table (Counter[tuple[bytes, ...]]): Frequency table of the words.
        stats (defaultdict[tuple[bytes, bytes], int]): Stats table of the words.
        pair_to_words (dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]): Pair to words table.
        pair (tuple[bytes, bytes]): Pair to merge.
    """
    p1, p2 = pair

    # use the unload-convert-upload pattern, unload the old word,
    # convert the old word to new word, upload the new word
    # this can avoid the problem of word loss and 
    # avoid handle the complex case of the stats and freq_table update 
    words = list(pair_to_words[pair])
    for word in words:
        # unload the old word
        freq = freq_table[word]
        for w1, w2 in zip(word[:-1], word[1:]):
            stats[(w1, w2)] -= freq
            pair_to_words[(w1, w2)].discard(word)

        # convert the old word to new word
        i = 0
        new_word = []
        while i < len(word):
            if i + 1 < len(word) and word[i] == p1 and word[i + 1] == p2:
                new_word.append(p1 + p2)
                # skip the pair
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        # update the frequency table
        freq_table[new_word] += freq
        freq_table.pop(word)

        # upload the new word
        for w1, w2 in zip(new_word[:-1], new_word[1:]):
            stats[(w1, w2)] += freq
            pair_to_words[(w1, w2)].add(new_word)

    # unload the pair
    stats.pop(pair)
    pair_to_words.pop(pair)

    return freq_table, stats, pair_to_words


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
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(2**8)}
    for i, token in zip(range(len(special_tokens)), special_tokens):
        vocab[2**8 + i] = token.encode("utf-8")
    cur_vocab_size = len(vocab)
    assert cur_vocab_size < vocab_size, "the vocab_size is too small, the minimal is 256 + len(special_tokens)"

    # BPE merges. Each list item is a tuple of bytes (<token1>, <token2>)
    merges: list[tuple[bytes, bytes]] = []
    # record the words of pair in pair_to_words
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    with open(input_path, "rb") as file:
        stats: dict[tuple[bytes, bytes], int] = defaultdict(int)
        freq_table = parallelize_pre_tokenizer(file)

        # count the frequency of each pair and record the words of pair in pair_to_words
        for key, freq in freq_table.items():
            if len(key) > 1:
                for b1, b2 in zip(key[:-1], key[1:]):
                    stats[(b1, b2)] += freq
                    pair_to_words[(b1, b2)].add(key)

        # train the BPE
        while cur_vocab_size < vocab_size:
            # find the pair with the highest frequency and break ties by lexicographic order
            pair = max(stats, key=lambda p: (stats[p], p))
            # add the pair to the vocabulary
            vocab[cur_vocab_size] = b"".join(pair)
            # add the pair to the merges
            merges.append(pair)
            cur_vocab_size += 1

            # merge the pair
            freq_table, stats, pair_to_words = merge(freq_table, stats, pair_to_words, pair)

    return vocab, merges
