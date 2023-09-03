import warnings

from datasets import Dataset, DatasetDict

from .prompts import get_prompt_template


def add_prompt_field(dataset: DatasetDict, prompt_id: int, new_field_name: str, *, with_answer: bool) -> DatasetDict:
    return dataset.map(
        map_prompt, fn_kwargs=dict(prompt_id=prompt_id, new_field=new_field_name, with_answer=with_answer)
    )


def map_prompt(example: Dataset, prompt_id: int, new_field: str, with_answer: bool) -> dict[str, str]:
    """fill inputs in prompt for a sample"""

    if prompt_id < 3:
        text = get_prompt_template(prompt_id).format(
            prompt=example["prompt"],
            a=example["A"],
            b=example["B"],
            c=example["C"],
            d=example["D"],
            e=example["E"],
            answer=example["answer"] if with_answer else "",
        )
    elif prompt_id in [3, 4]:
        text = get_prompt_template(prompt_id).format(
            prompt=example["prompt"],
            answer_text=example[example["answer"]] if with_answer else "",
        )
    else:
        raise NotImplementedError(prompt_id)

    if with_answer:
        text += " </s>"

    return {new_field: text}
