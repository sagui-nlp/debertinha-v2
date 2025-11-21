# DeBERTinha-v2

In 2023 we released [sagui-nlp/debertinha-ptbr-xsmall](https://huggingface.co/sagui-nlp/debertinha-ptbr-xsmall), however training the model for downstream tasks was not simple and the number of tokens used for training was not very large. With **DeBERTinha-v2** we aim at fixing those issues.

## Training pipeline

### Tokenizer

The first we need to train is a new tokenizer. To do that we use the **dominguesm/wikipedia-ptbr-20230601** dataset.
Training can be done using our prepared script:

```python
uv run debertinha/tok_train.py
```

### Translating model

We will use **microsoft/deberta-v3-xmall** as our base model again. However we want to keep the weights of the model and generate new embeddings that match the vocabulary size of our tokenizer. We can do that by running:

```python
uv run debertinha/translatemodel.py
```


### Dataset

This time we will use the **ClassiCC-Corpus/ClassiCC-PT** from the **maritaca.ai** team, as this is a well curated and clean dataset for the Portuguese language.

The dataset has 120B tokens that are divided into 58 shards. Downloading all the dataset consumes **165GB** so it is very large for storing in memory. To fix this issue we came up with a simple plan of downloading and tokenizing the data as we need them. For instance, we can download and tokenize the first shard using the script:

```python
uv run debertinha/data.py --index 0
uv run debertinha/tokenization.py --index 0
```

Because the training objective - MLM that will be discussed later - allows us to concatenate all the data and split it into chunks of 512 tokens, if we drop the last tokenized chunk none of the examples will have padding tokens added to it, thus we can safely remove the column **attention_mask**. Also, because **DeBERTa** does not train on **Next Sentence Prediction** (NSP), we can safely remove the **type_token_id** column as well. This allows us to keep only the **input_ids** and **special_tokens_map** in memory.

### Training 

For the first part of the training we want to tune the new embeddings to the model. To achieve that we will start with a **Masked Language Modeling** objective where only the embeddings and the **lm_head** will be tuned, all the other parameters will remain frozen.

After completing the embeddings tuning we can move on to the normal **DeBERTa-v3** training of using **RTD**.

Training on a shard will be done by the script:

```python
uv run debertinha/train.py --index 0
```

When the **index** is zero all the state of the training is newly initialized. After that we can pass the next indices and training will continue from where it stopped.

To avoid holding all the dataset and waiting for downloading and tokenization, we first prepare the index 0 of the dataset and process the next index while the training is happening, like so:

```python
# prepare index 0
uv run debertinha/download_and_tokenize.py --index 0

# prepare index 1 in the background
uv run debertinha/download_and_tokenize.py --index 1 &

# train on index 0
uv run debertinha/train.py --index 0

#etc..
```

This allows us to focus on spending on GPU power and avoid high spending on RAM.

## Next Steps

Next we will add a script on how to use spot machines on AWS to train the model and we will do the embeddings training using 10 shards of the dataset.
