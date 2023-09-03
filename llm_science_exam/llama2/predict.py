import gc

import numpy as np
import torch

from ..score import Perplexity
from ..typing import NDArray


def get_predicted_labels(model, tokenizer, example: dict, prompt_id: int) -> NDArray[np.str_]:
    cols = ["A", "B", "C", "D", "E"]
    samples = []
    for col in cols:
        if prompt_id < 3:
            samples.append(example["text"] + col)
        elif prompt_id in [3, 4]:
            samples.append(example["text"] + example[col])
        else:
            raise NotImplementedError(prompt_id)

    perplexities = calc_perplexities(model, tokenizer, samples)
    return np.take(cols, np.argsort(perplexities))


def calc_perplexities(model, tokenizer, samples: list[str]) -> NDArray[np.float_]:
    perp = Perplexity()

    inputs = tokenizer(samples, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs = model.generate(
        #     **inputs, return_dict_in_generate=True, max_length=len(example["text"]) + 4, output_scores=True
        # )
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
