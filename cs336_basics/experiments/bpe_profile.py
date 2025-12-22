import argparse
import cProfile
import os
import pickle
import pstats
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def save_tokenizer(vocab, merges, input_path, output_dir):
    filename = Path(input_path).stem
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, f"{filename}_vocab.pkl")
    merges_path = os.path.join(output_dir, f"{filename}_merges.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved to {vocab_path} and {merges_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile BPE training.")
    parser.add_argument(
        "--input_path", type=str, default="data/TinyStoriesV2-GPT4-valid.txt", help="Path to the input corpus."
    )
    parser.add_argument("--vocab_size", type=int, default=10000, help="Target vocabulary size.")
    parser.add_argument("--use_cprofile", action="store_true", default=False, help="Enable cProfile.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the tokenizer.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"File not found: {args.input_path}")
        return

    print(f"Profiling BPE training on {args.input_path}...")

    # profiler enable
    profiler = None
    if args.use_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

    # timer start
    start_time = time.perf_counter()

    # run bpe training
    vocab, merges = None, None
    try:
        # Note: special_tokens is hardcoded here, can be added to argparse if needed
        vocab, merges = train_bpe(
            input_path=args.input_path, vocab_size=args.vocab_size, special_tokens=["<|endoftext|>"]
        )
    except Exception:
        import traceback

        traceback.print_exc()
        return

    # timer stop
    end_time = time.perf_counter()

    if profiler:
        profiler.disable()

    # save tokenizer
    save_tokenizer(vocab, merges, args.input_path, args.output_dir)

    # print stats
    total_time_seconds = end_time - start_time
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    print("\n[Training Stats]")
    print(f"- Duration: {total_time_seconds:.4f} seconds ({total_time_seconds / 3600:.6f} hours)")
    print(f"- Max RSS (System): {rusage.ru_maxrss / 1024:.2f} MB")

    if vocab:
        longest_token = max(vocab.values(), key=len)
        print(f"- Longest token: {longest_token}")

    # cprofile stats
    if args.use_cprofile and profiler:
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        print("\n--- Top 30 functions by CUMULATIVE time ---")
        stats.sort_stats("cumtime").print_stats(30)
        print("\n--- Top 30 functions by TOTAL (internal) time ---")
        stats.sort_stats("tottime").print_stats(30)


def compare_saved_tokenizers(path1_prefix: str = "data/owt_train", path2_prefix: str = "data/TinyStoriesV2-GPT4-train"):
    def load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    vocab1 = load_pkl(f"{path1_prefix}_vocab.pkl")
    merges1 = load_pkl(f"{path1_prefix}_merges.pkl")
    vocab2 = load_pkl(f"{path2_prefix}_vocab.pkl")
    merges2 = load_pkl(f"{path2_prefix}_merges.pkl")

    tokens1 = set(vocab1.keys())
    tokens2 = set(vocab2.keys())
    m_rules1 = set(merges1)
    m_rules2 = set(merges2)

    print(f"Comparison: {path1_prefix} vs {path2_prefix}")
    print(f"Vocab sizes: {len(tokens1)} vs {len(tokens2)}")
    print(f"Shared tokens: {len(tokens1 & tokens2)}")
    print(f"Unique to {path1_prefix}: {len(tokens1 - tokens2)}")
    print(f"Unique to {path2_prefix}: {len(tokens2 - tokens1)}")

    print(f"\nMerge rules: {len(m_rules1)} vs {len(m_rules2)}")
    print(f"Shared merges: {len(m_rules1 & m_rules2)}")
    print(f"Unique merges to {path1_prefix}: {len(m_rules1 - m_rules2)}")
    print(f"Unique merges to {path2_prefix}: {len(m_rules2 - m_rules1)}")


if __name__ == "__main__":
    main()
