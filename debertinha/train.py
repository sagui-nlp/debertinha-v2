import os

from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments


from debertinha.tokenization import (
    tokenizer,
    TOKENIZED_DIR,
    tokenized_index_to_filename,
)


def get_mlm_data_collator() -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )


def get_model() -> AutoModelForMaskedLM:
    return AutoModelForMaskedLM.from_pretrained("deberta-v3-xsmall-pt-emb-reset")


def freeze_model_body(model: AutoModelForMaskedLM):
    # ---- 1) Freeze encoder (DeBERTa body) ----
    # For DebertaV2ForMaskedLM, the encoder is under `model.deberta`
    for param in model.deberta.parameters():
        param.requires_grad = False

    # ---- 2) Make sure embeddings + LM head are trainable ----
    # Input embeddings
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = True

    # Output embeddings / LM head
    out_emb = model.get_output_embeddings()
    if out_emb is not None:
        for p in out_emb.parameters():
            p.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable params in phase 1:", trainable)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train on shard")

    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Index number of the shard to train",
    )
    args = parser.parse_args()

    filename = tokenized_index_to_filename(args.index)
    print(f"Loading {filename}")
    training_dataset = load_from_disk(os.path.join(TOKENIZED_DIR, filename))
    print(training_dataset)

    print("Data collator")
    data_collator = get_mlm_data_collator()

    print("loading model...")
    model = get_model()
    freeze_model_body(model)

    training_args = TrainingArguments(
        output_dir="deberta-pt-mlm",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        fp16=True,
        report_to="none",
        max_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
    )

    trainer.train()
