from datasets import DatasetDict
from transformers import DebertaV2TokenizerFast

option_to_index = {option: idx for idx, option in enumerate("ABCDE")}


def map_preprocess(dataset: DatasetDict, tokenizer: DebertaV2TokenizerFast, *, max_length: int) -> DatasetDict:
    return dataset.map(
        preprocess,
        fn_kwargs=dict(tokenizer=tokenizer, max_length=max_length),
        remove_columns=["prompt", "context", "A", "B", "C", "D", "E", "answer"],
    )


def preprocess(example: dict[str, str], *, tokenizer: DebertaV2TokenizerFast, max_length: int):
    first_sentence = ["[CLS] " + example["context"]] * 5
    second_sentences = [" #### " + example["prompt"] + " [SEP] " + example[option] + " [SEP]" for option in "ABCDE"]
    tokenized_example = tokenizer(
        first_sentence, second_sentences, truncation="only_first", max_length=max_length, add_special_tokens=False
    )
    tokenized_example["label"] = option_to_index[example["answer"]]

    return tokenized_example
