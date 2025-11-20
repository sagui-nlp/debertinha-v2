from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import ftfy
from tqdm.auto import tqdm

dataset = load_dataset("dominguesm/wikipedia-ptbr-20230601")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")

dataset = concatenate_datasets([dataset["train"], dataset["test"]])


def batch_iterator(batch_size=10_000):
    for i in tqdm(range(0, len(dataset), batch_size)):
        text = dataset[i : i + batch_size]["text"]
        text = [ftfy.fix_text(t) for t in text]
        yield text


new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=52000)

new_tokenizer.save_pretrained("deberta-pt-52k-tokenizer")
