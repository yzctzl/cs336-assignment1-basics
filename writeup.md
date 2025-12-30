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
GPT-2 small:  FFN:   173 946 175 488 FLOPS, MHSA:  82 141 249 536 FLOPS
GPT-2 Medium: FFN:   618 475 290 624 FLOPS, MHSA: 257 698 037 760 FLOPS
GPT-2 Large:  FFN: 1 449 551 462 400 FLOPS, MHSA: 555 661 393 920 FLOPS
```

5. need 1.495 227 957 × 10^14 FLOPS for one forward if GPT-2 XL increased the context length to 16,384. MHSA takes 9.856 949 944 × 10^13 FLOPS, FFN takes 4.830 658 560 × 10^12 FLOPS.

## learning_rate_tuning

1. use lr = 1e2 is faster decay than lr = 1e1, but when use lr = 1e3 it diverged.

## adamwAccounting

1. hyperparameters: batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff = 4 × d_model
```
Parameters:
    token_embedding: vocab_size * d_model
    Transformer block: * num_layers
        RMSNorm(s): 2 * d_model
        MHSA: 4 * d_model * d_model
        FFN: 3 * d_model * 4 * d_model
    final RMSNorm: d_model
    output embedding: d_model * vocab_size

Gradients:
    1 * Parameters

Optimizer State:
    m,v: 2 * Parameters

Activations(intermediate outputs of each layer stored during the forward pass for gradient calculation in the backward pass):
    Transformer block: num_layers
        RMSNorm(s): 2 * batch_size * context_length * d_model
        MultiheadSelfAttention:
            QKV proj: batch_size * context_length * 3 * d_model
            Q^TK matmul: batch_size * num_heads * context_length * context_length
            softmax: batch_size * num_heads * context_length * context_length
            weighted sum of values: batch_size * context_length * d_model
            output projection: batch_size * context_length * d_model
        FFN:
            gate: 2 * batch_size * context_length * d_ff
            W2: batch_size * context_length * d_model
    final RMSNorm: batch_size * context_length * d_model
    output embedding: batch_size * context_length * vocab_size
    cross entropy: batch_size * context_length * vocab_size
```

2. 
```
GPT-2 XL-shaped model memory
= Parameters + Gradients + Optimizer State + Activations
= 4 * Parameters + Activations
= 8 508 230 400 float32 + 3 879 438 336 float32 * batch_size
= 34.03 GB + 15.52 GB * batch_size
=> 80GB
=> batch_size = 2.96 
```

3. 14 * Parameters FLOPS
```
weight dency: θ - lr_t * m / (sqrt(v) + eps), 2
m update: beta1 * m + (1 - beta1) * grad​, 3
v update: beta2 * v + (1 - beta2) * grad^2​, 4
lr_t: ~0
θ update: θ = θ - lr_t * m / (sqrt(v) + eps)​​, 5
Per Param = 14 FLOPS
```

4. 
```
our GPT-2 XL-shaped model has 2.125B parameters
FLOPS/Token = (2 + 4) * 2.125B = 12.75B
A100 FP32 FLOPS = 19.5 * 10^12 FLOPS * 0.5 = 9.75 * 10^12 FLOPS
Tokens = 400,000 * 1024 * 1024 = 419.3B
Time = 12.75B * 419.3B / (9.75 * 10^12) = 17.4 years

GPT-2 XL model has 1.558 B parameters
Times = 6 * 1.558B * 419.3B / (9.75 * 10^12) = 12.75 years
```

## learning_rate

1. search from big to small: 5e-2 -> 1e-2 -> 5e-3 -> 1e-3 -> 5e-4, get valid loss 1.61 with:`batch_size: 32*2, step: 2500, lr: 4e-3->5e-5, warmup: 200, context_length: 256, d_model: 512, num_layers: 4, num_heads: 16, d_ff: 1344` on my laptop w/ rtx 2060 maxq in 30mins.

2. "at the edge of stability": lr_max is in 1e-2 -> 1e-3


## batch_size_experiment

1. testd batch_size: `1, 2, 4, 8, 16, 32, 32*2, 32*4`, the more larger batch_size the more better valid loos can get.

## generate
```
User: Once upon a time
Assistant: , there was a little girl named Sue. Sue loved to play with her toys. One day, she found a big box in her room. She was very excited to see what was inside.
Sue opened the box and found a big, soft pillow. She wanted to wrap the pillow around her pillow. She put the pillow on her pillow and went to her room. Sue was very happy to see her pillow.
Sue put the pillow on her pillow and went to play with her toys. She had a lot of fun with her pillow. When she was done, she put the pillow on her pillow. She was very proud of her pillow. Sue knew that her pillow was special and special.
<|endoftext|>
User: Once upon a time
Assistant: , there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, she found a big, red apple in her yard. She was very happy and wanted to eat it all by herself.
Lily took the apple to her mom and said, "Mom, can I have a bite of the apple?" Her mom smiled and said, "Yes, you can have one." Lily was very happy and took the apple home.
Lily ate the apple and felt very happy. She knew that her mom would always be there to help her. And every time she ate her apple, she would say, "Thank you, Mom!" And they lived happily ever after.
<|endoftext|>
```


