from debertinha.data import download_single_file
from debertinha.tokenization import tokenize_dataset_from_index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and tokenize index")
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Index of the shard to download and tokenize",
    )
    args = parser.parse_args()

    print(f"Downloading index {args.index}")
    download_single_file(args.index)
    print(f"Tokenizing index {args.index}")
    tokenize_dataset_from_index(args.index)
    print("DONE!")
