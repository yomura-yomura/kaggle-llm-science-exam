from datasets import Dataset, DatasetDict

from ... import open_book
from .prompts import PromptType, get_prompt_template, get_prompt_type


def add_prompt_field(
    dataset: DatasetDict | Dataset,
    model_family_name: str,
    prompt_id: int,
    new_field_name: str,
    *,
    with_answer: bool,
    context_upper_limit_of_n_words: int,
    n_cpus: int | None = 4,
) -> DatasetDict | Dataset:
    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                k: add_prompt_field(
                    ds,
                    model_family_name,
                    prompt_id,
                    new_field_name,
                    with_answer=with_answer,
                    context_upper_limit_of_n_words=context_upper_limit_of_n_words,
                )
                for k, ds in dataset.items()
            }
        )

    assert isinstance(dataset, Dataset)

    df = dataset.to_pandas()
    prompt_type = get_prompt_type(model_family_name, prompt_id)
    print(f"{prompt_type = }")

    if context_upper_limit_of_n_words > 0:
        print(f"context reduced by context_upper_limit_of_n_words: {context_upper_limit_of_n_words}")
        df = open_book.get_df_with_reduced_context(
            df,
            use_all_answers_in_prompt=PromptType.alphabet_as_answer in get_prompt_type(model_family_name, prompt_id),
            upper_limit_of_n_words=context_upper_limit_of_n_words,
        )

    if PromptType.yes_or_no_as_answer in prompt_type:
        id_vars = ["id", "prompt"]
        cols = ["id", "prompt"]
        if PromptType.with_context in prompt_type:
            id_vars.append("context")
            cols.append("context")
        if "answer" in df.columns:
            id_vars.append("answer")
            cols.append("yes_or_no")

        df = df.melt(
            id_vars=id_vars,
            value_vars=["A", "B", "C", "D", "E"],
            var_name="alphabet",
            value_name="answer_text",
        ).sort_values(["id", "alphabet"])
        if "answer" in id_vars:
            df["yes_or_no"] = (df["alphabet"] == df["answer"]).map({True: "yes", False: "no"})

        df = df[[*cols, "answer_text"]]

    print(f"drop nan: {len(df):,} -> {len(df.dropna()):,}")
    df = df.dropna()

    dataset = Dataset.from_pandas(df)

    return dataset.map(
        map_prompt,
        fn_kwargs=dict(
            model_family_name=model_family_name,
            prompt_id=prompt_id,
            new_field=new_field_name,
            with_answer=with_answer,
        ),
        num_proc=n_cpus
        # remove_columns=["prompt", "context", "A", "B", "C", "D", "E", "answer"],
    )


def map_prompt(
    example: dict[str, str],
    model_family_name: str,
    prompt_id: int,
    new_field: str,
    with_answer: bool,
) -> dict[str, str]:
    """fill inputs in prompt for a sample"""
    prompt_type = get_prompt_type(model_family_name, prompt_id)

    if PromptType.prompt_as_answer in prompt_type:
        kwargs = dict(
            prompt=example["prompt"],
            answer_text=example[example["answer"]] if with_answer else "",
        )
    elif PromptType.alphabet_as_answer in prompt_type:
        if model_family_name in ("Platypus2", "OpenOrca-Platypus2"):
            kwargs = dict(
                prompt=example["prompt"],
                a=example["A"],
                b=example["B"],
                c=example["C"],
                d=example["D"],
                e=example["E"],
                answer=["A", "B", "C", "D", "E"].index(example["answer"]) + 1 if with_answer else "",
            )
        else:
            kwargs = dict(
                prompt=example["prompt"],
                a=example["A"],
                b=example["B"],
                c=example["C"],
                d=example["D"],
                e=example["E"],
                answer=example["answer"] if with_answer else "",
            )
    elif PromptType.yes_or_no_as_answer in prompt_type:
        kwargs = dict(
            prompt=example["prompt"],
            answer_text=example["answer_text"],
            yes_or_no=example["yes_or_no"] if with_answer else "",
        )
    else:
        raise NotImplementedError(f"Unexpected PromptType: {get_prompt_type(model_family_name, prompt_id)}")

    if PromptType.with_context in prompt_type:
        kwargs["context"] = example["context"]

    text = get_prompt_template(model_family_name, prompt_id).format(**kwargs)
    # if model_family_name == "Llama2":
    #     if with_answer:
    #         text += " </s>"

    return {new_field: text}
