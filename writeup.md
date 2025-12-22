# writeup
## unicode1

1. '\x00'
2. "'\\\\x00'"
3. this char isn't printable

## unicode2

1. the utf16/utf32 are inefficient than utf8 because they with a smaller compression_ratio and they are padding 0x00 ahead, meaning the tokenizer need to learn the padding, this is inappropriate.

2. the utf8 is variable-width encoding, example: "牛"

3. the high 3 bit for 2 bytes utf8 encoding must be `110`, e.g.: `b'\xF0\xF1'`

## train_bpe_tinystories

1. with a 16-core cpu, it takes about ~100s and max ~120MB memory to train a 10000 tokens bpe tokenizer on TinyStoriesV2-GPT4-train.txt. the longest token is b' accomplishment'. 

2. use cProfile to monitor the train, the pre-token's multi-processing.Pool takes 95% time. use `py-spy record --pid $(pgrep -f "python.*bpe_profile.py") --output cs336_basics/experiments/bpe_profile.svg`, is mucher faster.

## train_bpe_expts_owt

1. with a 1-core cpu, it takes about ~1200s and max ~9.6GB memory to tarin a 32000 tokens bpe tokenizer on owt_tarin.txt. the longest token b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82', it represent a utf-8 sting: "ÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃ"

2. they have half of merges is shared.

## tokenizer_experiments

1. TinyStories's compression ratio is 4.08 and OpenWebText's is 4.4

2. the compression ratio reduce to 4.05

3. ~6.6MB/s, need ~35h to encode Piles dataset

4. memory/storage efficient and fits most vocabularies