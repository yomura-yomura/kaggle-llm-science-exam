import dataclasses

import torch
from datasets import Dataset, DatasetDict
from transformers import DebertaV2TokenizerFast, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy

option_to_index = {option: idx for idx, option in enumerate("ABCDE")}


def map_preprocess(
    dataset: DatasetDict | Dataset,
    tokenizer: DebertaV2TokenizerFast,
    *,
    with_answer: bool,
    max_length: int,
    num_proc: int | None,
) -> DatasetDict | Dataset:
    remove_columns = ["id", "prompt", "context", "A", "B", "C", "D", "E"]
    if with_answer:
        remove_columns.append("answer")

    return dataset.map(
        preprocess,
        fn_kwargs=dict(tokenizer=tokenizer, max_length=max_length, with_answer=with_answer),
        remove_columns=remove_columns,
        num_proc=num_proc,
    )


def preprocess(example: dict[str, str], *, tokenizer: DebertaV2TokenizerFast, max_length: int, with_answer: bool):
    first_sentence = ["[CLS] " + example["context"]] * 5
    second_sentences = [" #### " + example["prompt"] + " [SEP] " + example[option] + " [SEP]" for option in "ABCDE"]
    try:
        tokenized_example = tokenizer(
            first_sentence, second_sentences, truncation="only_first", max_length=max_length, add_special_tokens=False
        )
    except Exception as e:
        print(f"encountered exception: {e}")
        first_sentence = [s[:1000] for s in first_sentence]
        second_sentences = [s[:1000] for s in second_sentences]
        tokenized_example = tokenizer(
            first_sentence, second_sentences, truncation="only_first", max_length=max_length, add_special_tokens=False
        )

    if with_answer:
        tokenized_example["label"] = option_to_index[example["answer"]]

    return tokenized_example


@dataclasses.dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else None
        if label_name is not None:
            labels = [feature.pop(label_name) for feature in features]
        else:
            labels = None

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        try:
            flattened_features = [
                [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
            ]
        except Exception as e:
            print(f"encountered exception: {e}")
            for i, feature in enumerate(features):
                try:
                    _ = [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
                except Exception:
                    print(f"{feature = }")
                    break
            raise e

        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # batch = {k: v.view(batch_size, 1, -1) for k, v in batch.items()}
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
