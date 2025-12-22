import os
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode(tokenizer: Tokenizer, sample: str):
    with open(sample, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        text = file.read().decode("utf-8", errors="replace")

    encoded = tokenizer.encode(text)
    return encoded, file_size


def encode_iter(tokenizer: Tokenizer, sample: str):
    with open(sample, encoding="utf-8", errors="replace") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        start_time = time.perf_counter()
        tokens = list(tokenizer.encode_iterable(file))
        end_time = time.perf_counter()
        np.save(sample, np.array(tokens, dtype=np.uint16))

        speed = file_size / (end_time - start_time) / 1000 / 1000  # MB/s

    return tokens, speed


owt_vocab = "data/owt_train_vocab.pkl"
owt_merge = "data/owt_train_merges.pkl"
tss_vocab = "data/TinyStoriesV2-GPT4-train_vocab.pkl"
tss_merge = "data/TinyStoriesV2-GPT4-train_merges.pkl"

owt = Tokenizer.from_files(owt_vocab, owt_merge, ["<|endoftext|>"])
tss = Tokenizer.from_files(tss_vocab, tss_merge, ["<|endoftext|>"])


def main():
    # 1
    owt_encoded, owt_size = encode(owt, "data/owt_sample.txt")
    print(f'owt sample compress ratio: {owt_size / len(owt_encoded)}')
    tss_encoded, tss_size = encode(tss, "data/TinyStoriesV2-GPT4-sample.txt")
    print(f'tss sample compress ratio: {tss_size / len(tss_encoded)}')

    # 2
    tss_encoded, tss_size = encode(owt, "data/TinyStoriesV2-GPT4-sample.txt")
    print(f'tss sample with owt tokenizer compress ratio: {tss_size / len(tss_encoded)}')

    # 3/4
    _, speed = encode_iter(tss, "data/TinyStoriesV2-GPT4-valid.txt")
    print(f"tss encode speed: {speed} MB/sec")


if __name__ == "__main__":
    main()
