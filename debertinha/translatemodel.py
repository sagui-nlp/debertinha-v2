from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch import nn


def translate_model(tokenizer_name: str, model_name: str, output_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # base model (can also be AutoModelForSequenceClassification, etc.)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # --- reset embeddings to match new vocab ---
    old_emb = model.get_input_embeddings()
    new_vocab_size = len(tokenizer)
    hidden_size = old_emb.embedding_dim

    # input embeddings
    new_input_emb = nn.Embedding(new_vocab_size, hidden_size)
    model.set_input_embeddings(new_input_emb)

    # output embeddings (for MLM / LM heads)
    old_output_emb = model.get_output_embeddings()
    if old_output_emb is not None:
        new_output_emb = nn.Linear(hidden_size, new_vocab_size, bias=False)
        model.set_output_embeddings(new_output_emb)
        model.tie_weights()  # keep input/output tied if the model expects that

    # update config to be consistent
    model.config.vocab_size = new_vocab_size
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
    model.config.eos_token_id = getattr(tokenizer, "eos_token_id", None)
    model.config.mask_token_id = getattr(tokenizer, "mask_token_id", None)

    model.save_pretrained(output_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Give model new embeddings")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="deberta-pt-52k-tokenizer",
        help="Name of the tokenizer",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="deberta-v3-xsmall", help="Model to use"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="deberta-v3-xsmall-pt-emb-reset",
        help="Name of the output model",
    )
