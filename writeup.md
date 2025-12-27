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

2. they have half of merges are shared.

## tokenizer_experiments

1. TinyStories's compression ratio is 4.08 and OpenWebText's is 4.4

2. the compression ratio reduce to 4.05

3. ~6.6MB/s, need ~35h to encode The Pile dataset

4. memory/storage efficient and fits most vocabularies


## transformer_accounting

1. GPT-2 XL have 2,127,057,600 trainable parameters, requires ~8.5GB memory if save as float32
```
token_embedding: vocab_size * d_model
Transformer_blocks: block * num_layers
    RMSNorm: d_model
    MHSA: 4 * d_model^2
    RMSNorm: d_model
    SwiGLU-FFN: 3 * d_model * d_ff
RMSNorm: d_model
Linear: d_model * vocab_size
```

2. matrix multiplies require 4,513,336,524,800 FLOPS in total
```
Transformer_blocks: block * num_layers
    MHSA:
        qkv_proj: (seq_len, d_model) (d_model, 3 * d_model)
        QK^T: (seq_len, d_model) (d_model, seq_len)
        scoreV: (seq_len, seq_len) (seq_len, d_model)
        output_proj: (seq_len, d_model) (d_model, d_model)
    SwiGLU-FFN:
        gate: (seq_len, d_model) (d_model, 2 * d_ff)
        w2_proj: (seq_len, d_ff) (d_ff, d_model)
lm_head: (seq_len, d_model) (d_model, vocab_size)

Other: Embedding/RMSNorm/RoPE/Softmax are O(d)
```

3. FFN requires ~3.0T FLOPS, MHSA require ~1.33T FLOPS

4. as the model increases, FFN takes up more proportionally, MHSA takes up less.
```
GPT-2 small:  FFN:   173 946 175 488 FLOPS, MHSA:  82 141 249 536 FLOPS
GPT-2 Medium: FFN:   618 475 290 624 FLOPS, MHSA: 257 698 037 760 FLOPS
GPT-2 Large:  FFN: 1 449 551 462 400 FLOPS, MHSA: 555 661 393 920 FLOPS
```

5. need 1.495 227 957 × 10^14 FLOPS for one forward if GPT-2 XL increased the context length to 16,384. MHSA takes 9.856 949 944 × 10^13 FLOPS, FFN takes 4.830 658 560 × 10^12 FLOPS.

