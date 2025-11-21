from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import ftfy
from tqdm.auto import tqdm


def train_tokenizer(
    dataset_name: str,
    model_name: str,
    vocab_size: int,
    output_name: str,
):
    dataset = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    def batch_iterator(batch_size=10_000):
        for i in tqdm(range(0, len(dataset), batch_size)):
            text = dataset[i : i + batch_size]["text"]
            text = [ftfy.fix_text(t) for t in text]
            yield text

    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=vocab_size
    )

    new_tokenizer.save_pretrained(output_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train tokenizer")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="dominguesm/wikipedia-ptbr-20230601",
        help="Dataset to be used to train tokenizer",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="Model to use the same type of tokenizer",
    )
    parser.add_argument(
        "-s",
        "--vocab_size",
        type=int,
        default=52000,
        help="Size of the tokenizer vocabulary",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="deberta-pt-52k-tokenizer",
        help="Name of the tokenizer",
    )
    args = parser.parse_args()

    train_tokenizer(
        dataset_name=args.dataset,
        model_name=args.model,
        vocab_size=args.vocab_size,
        output_name=args.output,
    )
