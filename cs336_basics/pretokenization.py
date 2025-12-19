import multiprocessing
import os
from collections import Counter
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
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
    chunk: str,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> Counter[tuple[bytes, ...]]:
    """
    a coarse-grained tokenization over the corpus 
    that helps us count how often pairs of characters appear.
    """

    # When we pretokenize and count, we end up with the frequency table.
    # e.g. {low: 5, lower: 2, widest: 3, newest: 6}
    frequency_table: Counter[tuple[bytes, ...]] = Counter()

    # chunk like [Doc 1]<|endoftext|>[Doc2], should split on the special token <|endoftext|>
    # using re.split with "|".join(special_tokens) as the delimiter 
    # with careful use of re.escape since | may occur in the special tokens
    split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    docs = re.split(split_pattern, chunk)

    # It is convenient to represent as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}.
    def str_to_bytes_tuple(s: str):
        return tuple(bytes([b]) for b in s.encode("utf-8"))

    # use re.finditer to avoid storing the pre-tokenized words 
    # as you construct your mapping from pre-tokens to their counts
    compiled = re.compile(PAT)
    for doc in docs:
        word_iter = compiled.finditer(doc)
        # counting
        all_matches = (str_to_bytes_tuple(match.group(0)) for match in word_iter)
        frequency_table += Counter(all_matches)

    return frequency_table


def parallelize_pre_tokenizer(
    file: BinaryIO,
    cpu_count: int | None = os.cpu_count(),
) -> Counter[tuple[bytes, ...]]:
    """
    speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing.
    """
    cpu_count = cpu_count if cpu_count is not None else 16
    boundaries = find_chunk_boundaries(file, cpu_count)

    # file.seek(0)  # chunk_boundaries[0] == 0
    # chunks = [(file.read(boundaries[i + 1] - boundaries[i]).decode(errors="ignore")) for i in range(cpu_count)]
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="replace")
        chunks.append(chunk)
    assert len(chunks) == cpu_count, "Wrong chunks numbers, you should check the boundaries list."

    frequency_table = Counter()
    with multiprocessing.Pool(cpu_count) as pool:
        result_iter = pool.imap(pre_tokenization, chunks)
        for c in result_iter:
            frequency_table += c

    return frequency_table

