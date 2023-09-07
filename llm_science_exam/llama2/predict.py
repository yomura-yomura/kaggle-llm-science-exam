import gc

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast

from ..score import Perplexity
from ..typing import NDArray
from .prompts import PromptType, get_prompt_type


def get_predicted_labels(
    model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, example: dict, *, model_family: str, prompt_id: int
) -> NDArray[np.str_]:
    answers = ["A", "B", "C", "D", "E"]

    if model_family == "Platypus2" and prompt_id == 2:
        cols = ["1", "2", "3", "4", "5"]
    else:
        cols = ["A", "B", "C", "D", "E"]

    samples = []
    for col in cols:
        match get_prompt_type(model_family, prompt_id):
            case PromptType.prompt_as_answer:
                samples.append(example["text"] + example[col] + f" {tokenizer.eos_token}")
            case PromptType.alphabet_as_answer:
                samples.append(example["text"] + col)
            case _:
                assert False

    perplexities = calc_perplexities(model, tokenizer, samples)
    return np.take(answers, np.argsort(perplexities))


def calc_perplexities(model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, samples: list[str]) -> NDArray[np.float_]:
    perp = Perplexity()

    inputs = tokenizer(samples, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach()

        labels = inputs["input_ids"].detach()
        labels.masked_fill_(~inputs["attention_mask"].detach().bool(), -100)

        perplexities = [
            perp(logit.unsqueeze(0), label.unsqueeze(0)).cpu().numpy() for logit, label in zip(logits, labels)
        ]

        del inputs, outputs, logits, labels
        torch.cuda.empty_cache()
        gc.collect()

    return np.stack(perplexities, axis=0)
