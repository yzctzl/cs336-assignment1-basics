import json
import sys

import click
import torch
from einops import rearrange

from cs336_basics.config import Configures
from cs336_basics.data import load_checkpoint
from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


class Generator:
    def __init__(self, cfg: Configures) -> None:
        self.cfg = cfg
        self.tokenizer = Tokenizer.from_files(**cfg.tokenizer.model_dump())

        model = TransformerLM(**cfg.model.model_dump())
        model.to(cfg.model.device)
        load_checkpoint(cfg.infer.checkpoint, model)
        # self.model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.model.eval()

    def top_p_logits(self, probs: torch.Tensor, top_p: float):
        # Sort probabilities and track original indices
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # Compute cumulative probability distribution
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Identify indices where cumulative probability exceeds top_p
        mask = cum_probs > top_p
        # Shift mask right to include the first token exceeding top_p
        mask[..., 1:] = mask[..., :-1].clone()
        # Ensure the highest-probability token is never masked
        mask[..., 0] = False

        # Zero out probabilities outside the top_p threshold
        sorted_probs[mask] = 0.0
        # Renormalize the remaining probabilities
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample an index from the filtered distribution
        next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
        # Map back to original vocabulary index
        return torch.gather(sorted_indices, -1, next_token_idx)

    @torch.inference_mode()
    def streamer(self, prompt: str, max_gens: int, temperature: float, top_p: float):
        _prompt = self.tokenizer.encode(prompt)
        _prompt = torch.tensor(_prompt, dtype=torch.long, device=self.cfg.model.device)
        tokens = rearrange(_prompt, "seq_len -> 1 seq_len")
        context_length = self.cfg.model.context_length

        for _ in range(max_gens):
            if tokens.shape[-1] >= context_length:
                tokens = tokens[:, -context_length:]

            logits = self.model(tokens)
            logits = logits[:, -1, :] / (temperature + 1e-9)
            probs = softmax(logits, dim=-1)

            next_token = self.top_p_logits(probs, top_p)

            tokens = torch.cat((tokens, next_token), dim=1)

            next_word = self.tokenizer.decode([next_token.item()])  # pyright: ignore[reportArgumentType]
            yield next_word
            if next_word in self.cfg.tokenizer.special_tokens:
                return


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True)
@click.option("-l", "--length", type=click.INT, default=512, help="Max LM output length.")
@click.option("-t", "--temperature", type=click.FLOAT, default=0.8, help="Temperature range from 0.0 to 1.0")
@click.option("-p", "--top_p", type=click.FLOAT, default=0.8, help="The Top-p")
def main(config, length, temperature, top_p):
    try:
        with open(config) as f:
            conf = json.load(f)
        cfg = Configures(**conf)
    except Exception as e:
        print(f"check your config: {e}")
        return

    gen = Generator(cfg)
    while True:
        try:
            prompt = input("User: ")
            print("Assistant: ", end="")
            for word in gen.streamer(prompt.strip(), length, temperature, top_p):
                print(word, end="", flush=True)
            print()
        except KeyboardInterrupt or EOFError:
            print("\nSYSTEM: BYE!")
            sys.exit(0)


if __name__ == "__main__":
    main()
