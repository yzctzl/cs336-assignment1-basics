import multiprocessing
import os
from collections import Counter
from typing import BinaryIO

import regex as re
from tqdm import tqdm

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes = b"<|endoftext|>",
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, desired_num_chunks):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenization(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """
    a coarse-grained tokenization over the corpus
    that helps us count how often pairs of characters appear.
    """
    # read chunk and decode to utf-8 with error replace
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="replace")

    # When we pretokenize and count, we end up with the frequency table.
    # e.g. {low: 5, lower: 2, widest: 3, newest: 6}
    frequency_table: Counter[tuple[bytes, ...]] = Counter()

    # chunk like [Doc 1]<|endoftext|>[Doc2], should split on the special token <|endoftext|>
    # using re.split with "|".join(special_tokens) as the delimiter
    # with careful use of re.escape since | may occur in the special tokens
    split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    docs = re.split(split_pattern, chunk)

    # use re.finditer to avoid storing the pre-tokenized words
    # as you construct your mapping from pre-tokens to their counts
    for doc in docs:
        word_iter = PAT.finditer(doc)
        # It is convenient to represent as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}.
        all_matches = map(lambda m: tuple(bytes([b]) for b in m.group().encode("utf-8")), word_iter)

        # counting words
        frequency_table.update(all_matches)

    return frequency_table


def pre_tokenization_wrapper(args):
    return pre_tokenization(*args)


def parallelize_pre_tokenizer(
    input_path: str | os.PathLike,
    cpu_count: int | None = os.cpu_count(),
    special_tokens: list[str] = ["<|endoftext|>"],
) -> Counter[tuple[bytes, ...]]:
    """
    speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing.
    """
    cpu_count = max(cpu_count - 1, 2) if cpu_count is not None else 16 - 1
    print(f"parallelize pre-tokenizing with {cpu_count} processes...")
    # find 8x chunks for load balance
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, 8 * cpu_count)

    # do not read chunks and decode them here, bcuz there is 8x chunks
    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args.append((input_path, start, end, special_tokens))

    # callback for pool to update table
    frequency_table = Counter()

    # parallelize pre-tokenization
    with multiprocessing.Pool(cpu_count) as pool:
        # multiprocessing do not have multi-args version imap so need a wrapper
        result_iter = tqdm(
            pool.imap(pre_tokenization_wrapper, args), 
            total=len(args), 
            desc="Pre-tokenizing",
            unit="chunk"
        )
        for c in result_iter:
            frequency_table.update(c)

    return frequency_table
