import heapq
import os
from collections import defaultdict

from tqdm import tqdm

from .pretokenization import parallelize_pre_tokenizer


class PairItem:
    """
    Wrapper class for byte bp heap items. 
    Items with higher count and lexicographical values will have priority.
    """
    __slots__ = ['freq', 'pair']
    def __init__(self, freq, pair):
        self.pair = pair
        self.freq = freq

    def __lt__(self, other):
        # frequency first
        if self.freq != other.freq:
            return self.freq > other.freq
        # break ties by lexicographic order
        return self.pair > other.pair


def merge(
    freq_table: dict[int, int],
    stats: defaultdict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[int]],
    pair: tuple[bytes, bytes],
    idx_to_word: list[tuple[bytes, ...]]
) -> set[tuple[bytes, bytes]]:
    """
    """
    p1, p2 = pair
    new_token = p1 + p2
    touched_pairs = {pair}

    # use the unload-convert-upload pattern, unload the old word,
    # convert the old word to new word, upload the new word
    # this can avoid the problem of word loss and 
    # avoid handle the complex case of the stats and freq_table update 
    words_indices = set(pair_to_words.get(pair, ()))
    for idx in words_indices:
        word = idx_to_word[idx]
        # unload the old word
        freq = freq_table[idx]
        for w in zip(word[:-1], word[1:]):
            stats[w] -= freq
            pair_to_words[w].discard(idx)
            touched_pairs.add(w)

        # convert the old word to new word
        i = 0
        new_word_list = []
        while i < len(word):
            if i + 1 < len(word) and word[i] == p1 and word[i + 1] == p2:
                new_word_list.append(new_token)
                # skip the pair
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1
        new_word = tuple(new_word_list)

        # update index manager, use the old idx directly do not add new idx
        idx_to_word[idx] = new_word

        # upload the new word
        for w in zip(new_word[:-1], new_word[1:]):
            stats[w] += freq
            pair_to_words[w].add(idx)
            touched_pairs.add(w)

    # unload the pair
    stats.pop(pair, None)
    pair_to_words.pop(pair, None)

    return touched_pairs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"],
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
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    cur_vocab_size = len(vocab)
    assert cur_vocab_size < vocab_size, "the vocab_size is too small, the minimal is 256 + len(special_tokens)"

    # BPE merges. Each list item is a tuple of bytes (<token1>, <token2>)
    merges: list[tuple[bytes, bytes]] = []

    # To efficiently perform BPE training, we maintain the following data structures:
    # 1. idx_to_word : Store unique words (sequences of tokens) and their indices.
    # 2. pair_to_words: An inverted index mapping each adjacent token pair to the indices of words containing it.
    # 3. stats: Records the total frequency of each token pair across the entire corpus.
    # Optimization: When merging the most frequent pair, we only need to find the affected words using pair_to_words,
    # update their token sequences, and synchronize stats and pair_to_words, avoiding a full corpus re-scan.
    idx_to_word: list[tuple[bytes, ...]] = []
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    stats: dict[tuple[bytes, bytes], int] = defaultdict(int)

    raw_freq_table = parallelize_pre_tokenizer(input_path, special_tokens=special_tokens)

    print("counting the pre-tokens and initing heap...")
    # new freq_table is a map from tuple index to frequency
    freq_table: dict[int, int] = defaultdict(int)
    for word, freq in raw_freq_table.items():
        idx = len(idx_to_word)

        # construct the index manager
        idx_to_word.append(word)
        # construct new freq_table
        freq_table[idx] = freq

        # count the frequency of each pair and record the words of pair in pair_to_words
        if len(word) > 1:
            for w in zip(word[:-1], word[1:]):
                stats[w] += freq
                pair_to_words[w].add(idx)

    del raw_freq_table

    # init the priority queue with all pairs that have a positive frequency.
    # using heapify allows for efficient O(N) construction of the heap.
    bp_heap = [PairItem(f, p) for p, f in stats.items()]
    heapq.heapify(bp_heap)

    pbar = tqdm(total=vocab_size - cur_vocab_size, desc="BPE merging", unit="merge")
    # train the BPE
    while cur_vocab_size < vocab_size:
        if not bp_heap:
            break
        # heap pop to get max pair
        item = heapq.heappop(bp_heap)
        pair, freq = item.pair, item.freq

        """
            MECHANISM: Lazy Update (Lazy Deletion) in Priority Queue
            --------------------------------------------------------
            In BPE training, the frequency of pairs changes constantly as we merge them. 
            However, Python's `heapq` module does not support efficient item updates or 
            removals (finding an item takes O(N) time).

            To maintain O(log N) performance, we use a "Lazy Update" strategy:

            1. PUSHING (Trigger): Whenever a pair's frequency is updated in the `stats` 
            dictionary, we simply push a NEW (frequency, pair) tuple into the heap. 
            We do NOT search for or remove the old entries already present in the heap.

            2. POPPING (Validation): When we extract the maximum element via `heappop()`:
            - We compare the frequency from the heap entry with the current "Source of 
                Truth" (the `stats` dictionary).
            - If `stats[pair] != popped_frequency`, it means the heap entry is "stale" 
                (outdated) because the pair's frequency has changed since this entry 
                was pushed.
            - We simply discard the stale entry and continue popping until we find a 
                match or the heap is empty.

            This effectively simulates a dynamic priority queue with O(log N) updates 
            and O(log N) extractions, avoiding costly O(N) searches.
        """
        if stats.get(pair, 0) != freq or freq <= 0:
            continue

        # add the pair to the vocabulary
        vocab[cur_vocab_size] = b"".join(pair)
        # add the pair to the merges
        merges.append(pair)
        cur_vocab_size += 1

        # merge the pair
        touched = merge(freq_table, stats, pair_to_words, pair, idx_to_word)
        # only update the touched stats
        for p in touched:
            f = stats[p]
            if f > 0:
                heapq.heappush(bp_heap, PairItem(f, p))

        pbar.update()
    pbar.close()

    return vocab, merges
