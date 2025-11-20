import os

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from debertinha.data import DATA_DIR, index_to_filename, base_dir

tokenizer = AutoTokenizer.from_pretrained("deberta-v3-xsmall-pt-emb-reset")
TOKENIZED_DIR = os.path.join(base_dir, "tokenized_data")


def tokenized_index_to_filename(index: int) -> str:
    return f"tokenized-{index:05d}"


def prepare_dataset(raw_ds: Dataset) -> Dataset:
    raw_ds = raw_ds.select(indices=range(10))

    # 2. Tokenize WITHOUT truncation/padding
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            return_special_tokens_mask=True,
            padding=False,
            truncation=False,  # <— important
        )

    tokenized_ds = raw_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_ds.column_names,
    )

    tokenized_ds = tokenized_ds.remove_columns(["token_type_ids", "attention_mask"])

    # 3. Group tokens into 512-token blocks
    block_size = 512

    def group_texts(examples):
        # Concatenate within the batch
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Drop small remainder to keep shapes nice (optional)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=1_000,
    )
    lm_ds = lm_ds.select(range(len(lm_ds) - 1))

    return lm_ds


def tokenize_dataset_from_index(index: int):
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Reading from {filepath}")

    dataset = load_dataset("parquet", data_files=filepath, split="train")

    print("Preparing dataset")

    tokenized_dataset = prepare_dataset(dataset)

    output_filepath = os.path.join(TOKENIZED_DIR, tokenized_index_to_filename(index))
    tokenized_dataset.save_to_disk(output_filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize already saved shards")

    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Index number of the shard to tokenize",
    )
    args = parser.parse_args()

    print(f"Tokenize index {args.index}")

    tokenize_dataset_from_index(index=args.index)

    print(f"Done! Tokenized saved to {TOKENIZED_DIR}")
