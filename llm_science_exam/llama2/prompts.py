import enum
from typing import Literal

from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = {
    #
    # Llama2
    #
    "Llama2": {
        1: """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]

Question: {prompt}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n
E) {e}\n

### Answer: {answer}""",
        2: """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Question: {prompt}
A) {a}
B) {b}
C) {c}
D) {d}
E) {e}
[/INST] {answer}""",
        3: "{prompt}</s>{answer_text}",
        4: """Question: {prompt}

### Answer: {answer_text}""",
        5: """Your answer must be one character from A to E.

Question:
{prompt}
A) {a}
B) {b}
C) {c}
D) {d}
E) {e}

Answer: 
{answer}""",
    },
    #
    # Platypus2
    #
    "Platypus2": {
        1: """ ### Instruction:

{prompt}

### Response: {answer_text}""",
        2: """### Instruction:

{prompt}
1) {a}
2) {b}
3) {c}
4) {d}
5) {e}

### Response: {answer}""",
    },
}


class PromptType(enum.Enum):
    alphabet_as_answer = enum.auto()
    prompt_as_answer = enum.auto()


def get_prompt_type(model_family_name: str, prompt_id: int) -> PromptType:
    if model_family_name == "Llama2":
        if prompt_id < 3 or prompt_id > 4:
            return PromptType.alphabet_as_answer
        elif prompt_id in [3, 4]:
            return PromptType.prompt_as_answer
        else:
            raise NotImplementedError(prompt_id)
    elif model_family_name == "Platypus2":
        if prompt_id == 1:
            return PromptType.prompt_as_answer
        elif prompt_id == 2:
            return PromptType.alphabet_as_answer
        else:
            raise NotImplementedError(prompt_id)
    else:
        raise NotImplementedError(model_family_name)


def get_prompt_template(model_family_name: str, prompt_id: int) -> PromptTemplate:
    match get_prompt_type(model_family_name, prompt_id):
        case PromptType.prompt_as_answer:
            input_variables = ["prompt", "answer_text"]
        case PromptType.alphabet_as_answer:
            input_variables = ["prompt", "a", "b", "c", "d", "e", "answer"]
        case _:
            assert False

    return PromptTemplate(template=PROMPT_TEMPLATE[model_family_name][prompt_id], input_variables=input_variables)
