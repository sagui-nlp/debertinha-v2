import os
import math
from pathlib import Path
import json

from datasets import load_from_disk, Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments


from debertinha.tokenization import (
    tokenizer,
    TOKENIZED_DIR,
    tokenized_index_to_filename,
)


TRAINING_SHARDS = 10
META_FILE = Path(TOKENIZED_DIR) / "training_meta.json"


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


def load_shard(index: int) -> Dataset:
    filename = tokenized_index_to_filename(index)
    print(f"Loading {filename}")
    training_dataset = load_from_disk(os.path.join(TOKENIZED_DIR, filename))
    print(training_dataset)
    return training_dataset


def compute_total_steps(first_shard_len: int, args: TrainingArguments) -> int:
    # Effective batch size per update step
    eff_bs = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * max(1, args.world_size if hasattr(args, "world_size") else 1)
    )

    steps_per_shard = math.ceil(first_shard_len / eff_bs)
    total_steps = steps_per_shard * TRAINING_SHARDS * args.num_train_epochs
    return total_steps


def get_training_args() -> TrainingArguments:
    base_args = dict(
        output_dir="deberta-pt-mlm",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        fp16=True,
        report_to="none",
        save_total_limit=2,
        max_steps=1,
    )

    # Create a dummy TrainingArguments to compute steps
    tmp_args = TrainingArguments(**base_args)

    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        total_steps = meta["total_steps"]
    else:
        # Only happens on index 0
        total_steps = compute_total_steps(len(training_dataset), tmp_args)
        META_FILE.write_text(json.dumps({"total_steps": total_steps}))

    base_args.pop("max_steps")
    training_args = TrainingArguments(
        **base_args,
        max_steps=total_steps,  # global number of optimizer steps
    )
    return training_args


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

    training_dataset = load_shard(index=args.index)

    print("Data collator")
    data_collator = get_mlm_data_collator()

    print("loading model...")
    model = get_model()
    freeze_model_body(model)

    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
    )

    if args.index == 0:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)

    trainer.save_state()
