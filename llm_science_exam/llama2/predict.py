import ctypes
import gc

import numpy as np
import torch
import tqdm
from datasets import Dataset
from transformers import LlamaForCausalLM, LlamaTokenizerFast

from ..score import Perplexity
from ..typing import NDArray
from .prompts import PromptType, get_prompt_type


def get_predicted_labels(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizerFast,
    data: Dataset,
    *,
    model_family: str,
    prompt_id: int,
    upper_limit_to_split_samples: int,
    print_perplexities: bool = False,
    batch_size=8,
) -> NDArray[np.str_]:
    answers = ["A", "B", "C", "D", "E"]

    prompt_type = get_prompt_type(model_family, prompt_id)

    if PromptType.yes_or_no_as_answer in prompt_type:
        cols = ["yes", "no"]
    else:
        match (model_family, prompt_id):
            case ("Platypus2", 2):
                cols = ["1", "2", "3", "4", "5"]
            case ("OpenOrca-Platypus2", 1):
                cols = ["1", "2", "3", "4", "5"]
            case _:
                cols = ["A", "B", "C", "D", "E"]

    if PromptType.yes_or_no_as_answer in prompt_type:
        indices_list = np.arange(len(data)).reshape(-1, len(answers))
    else:
        indices_list = np.array_split(np.arange(len(data)), int(len(data) / batch_size * len(answers)))

    answer_token_ids = [tokens[0] for tokens in tokenizer(cols, add_special_tokens=False)["input_ids"]]
    # answer_token_ids = 4874
    print(f"{answer_token_ids = }")
    preds = []
    for indices in tqdm.tqdm(indices_list, desc="inference"):
        samples = [data[int(i)]["text"] for i in indices]

        assert tokenizer.padding_side == "left"
        inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.forward(**inputs.to(model.device))
            answer_token_logits = outputs["logits"][:, -1, answer_token_ids]

            if answer_token_logits.ndim == 1:
                pred = answer_token_logits.detach().cpu().numpy()
                # pred = np.take(answers, np.argsort(logits, axis=-1), axis=-1)
            elif answer_token_logits.ndim == 2:
                pred = torch.softmax(answer_token_logits, dim=1)[..., cols.index("yes")].detach().cpu().numpy()
            else:
                assert False
            pred = np.take(answers, np.argsort(pred, axis=-1)[..., ::-1], axis=-1)
            del outputs
        preds.append(pred)
    return np.stack(preds, axis=0)

    # indices_list = np.array_split(np.arange(len(data)), int(len(data) / batch_size * len(answers)))
    #
    # perplexity_list = []
    #
    # for indices in tqdm.tqdm(indices_list, desc="inference"):
    #     samples = []
    #     for example in (data[int(i)] for i in indices):
    #         for col in cols:
    #             prompt_type = get_prompt_type(model_family, prompt_id)
    #             if PromptType.prompt_as_answer in prompt_type:
    #                 samples.append(example["text"] + example[col] + f" {tokenizer.eos_token}")
    #             elif PromptType.alphabet_as_answer in prompt_type:
    #                 samples.append(example["text"] + col)
    #             elif PromptType.yes_or_no_as_answer in prompt_type:
    #                 samples.append(example["text"] + col)
    #             else:
    #                 assert False
    #
    #     perplexities = calc_perplexities(
    #         model, tokenizer, samples, upper_limit_to_split_samples=upper_limit_to_split_samples
    #     )
    #     perplexity_list.append(perplexities)
    #
    # perplexities = np.concatenate(perplexity_list, axis=0)
    # # perplexities = perplexities.reshape(-1, len(answers))
    # perplexities = perplexities.reshape(-1, len(answers), 2)[..., cols.index("yes")]
    # if print_perplexities:
    #     print(f"perplexities = {perplexities}")
    #
    # return np.take(answers, np.argsort(perplexities, axis=-1), axis=-1)


def calc_perplexities(
    model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, samples: list[str], *, upper_limit_to_split_samples: int
) -> NDArray[np.float_]:
    perp = Perplexity()

    inputs = tokenizer(samples, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True)

    # token数が大きいものはOOMを防ぐために小分けにして推論させる。
    if inputs["input_ids"].shape[1] > upper_limit_to_split_samples and len(samples) != 1:
        return np.concatenate(
            [
                calc_perplexities(model, tokenizer, [sample], upper_limit_to_split_samples=upper_limit_to_split_samples)
                for sample in samples
            ]
        )

    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach()

        labels = inputs["input_ids"].detach()
        labels.masked_fill_(~inputs["attention_mask"].detach().bool(), -100)

        perplexities = [
            perp(logit.unsqueeze(0), label.unsqueeze(0)).cpu().numpy()
            for logit, label in zip(logits, labels, strict=True)
        ]

        del inputs, outputs, logits, labels
        torch.cuda.empty_cache()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()

    return np.stack(perplexities, axis=0)
