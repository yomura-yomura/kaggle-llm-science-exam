from datasets import DatasetDict

from .prompts import PromptType, get_prompt_template, get_prompt_type


def add_prompt_field(
    dataset: DatasetDict, model_family_name: str, prompt_id: int, new_field_name: str, *, with_answer: bool
) -> DatasetDict:
    return dataset.map(
        map_prompt,
        fn_kwargs=dict(
            model_family_name=model_family_name, prompt_id=prompt_id, new_field=new_field_name, with_answer=with_answer
        ),
    )


def map_prompt(
    example: dict[str, str], model_family_name: str, prompt_id: int, new_field: str, with_answer: bool
) -> dict[str, str]:
    """fill inputs in prompt for a sample"""
    match get_prompt_type(model_family_name, prompt_id):
        case PromptType.prompt_as_answer:
            text = get_prompt_template(model_family_name, prompt_id).format(
                prompt=example["prompt"],
                answer_text=example[example["answer"]] if with_answer else "",
            )
        case PromptType.alphabet_as_answer:
            if model_family_name == "Platypus2":
                text = get_prompt_template(model_family_name, prompt_id).format(
                    prompt=example["prompt"],
                    a=example["A"],
                    b=example["B"],
                    c=example["C"],
                    d=example["D"],
                    e=example["E"],
                    answer=["A", "B", "C", "D", "E"].index(example["answer"]) + 1 if with_answer else "",
                )
            else:
                text = get_prompt_template(model_family_name, prompt_id).format(
                    prompt=example["prompt"],
                    a=example["A"],
                    b=example["B"],
                    c=example["C"],
                    d=example["D"],
                    e=example["E"],
                    answer=example["answer"] if with_answer else "",
                )
        case _:
            assert False

    if model_family_name == "Llama2":
        if with_answer:
            text += " </s>"

    return {new_field: text}
