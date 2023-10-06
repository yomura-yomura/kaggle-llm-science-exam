import enum

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
        6: """Your answer must be one character from A to E.

Context:
{context}

Question:
{prompt}
A) {a}
B) {b}
C) {c}
D) {d}
E) {e}

Answer: 
{answer}""",
        7: """Context:
{context}

Question:
{prompt}

Answer: 
{answer_text}""",
        8: """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be pertinent.

### Input:
Question: {prompt}
Proposed answer: {answer_text}


### Response:
{yes_or_no}""",
        9: """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be pertinent.

### Input:
Context:
{context}

Question:
{prompt}

Proposed answer:
{answer_text}

### Response:
{yes_or_no}""",
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
    #
    # OpenOrca-Platypus2
    #
    "OpenOrca-Platypus2": {
        1: """### Instruction:

{prompt}
1) {a}
2) {b}
3) {c}
4) {d}
5) {e}

### Response: {answer}"""
    },
}


class PromptType(enum.Flag):
    prompt_as_answer = enum.auto()
    alphabet_as_answer = enum.auto()
    yes_or_no_as_answer = enum.auto()

    with_context = enum.auto()


def get_prompt_type(model_family_name: str, prompt_id: int) -> PromptType:
    if model_family_name == "Llama2":
        if prompt_id < 3 or prompt_id == 5:
            return PromptType.alphabet_as_answer
        elif prompt_id in [3, 4]:
            return PromptType.prompt_as_answer
        elif prompt_id == 7:
            return PromptType.prompt_as_answer | PromptType.with_context
        elif prompt_id == 6:
            return PromptType.alphabet_as_answer | PromptType.with_context
        elif prompt_id == 8:
            return PromptType.yes_or_no_as_answer
        elif prompt_id == 9:
            return PromptType.yes_or_no_as_answer | PromptType.with_context
        else:
            raise NotImplementedError(prompt_id)
    elif model_family_name == "Platypus2":
        if prompt_id == 1:
            return PromptType.prompt_as_answer
        elif prompt_id == 2:
            return PromptType.alphabet_as_answer
        else:
            raise NotImplementedError(prompt_id)
    elif model_family_name == "OpenOrca-Platypus2":
        if prompt_id == 1:
            return PromptType.alphabet_as_answer
        else:
            raise NotImplementedError(prompt_id)
    else:
        raise NotImplementedError(model_family_name)


def get_prompt_template(model_family_name: str, prompt_id: int) -> PromptTemplate:
    prompt_type = get_prompt_type(model_family_name, prompt_id)
    if PromptType.prompt_as_answer in prompt_type:
        input_variables = ["prompt", "answer_text"]
    elif PromptType.alphabet_as_answer in prompt_type:
        input_variables = ["prompt", "a", "b", "c", "d", "e", "answer"]
    elif PromptType.yes_or_no_as_answer in prompt_type:
        input_variables = ["prompt", "answer_text", "yes_or_no"]
    else:
        assert False

    if PromptType.with_context in prompt_type:
        input_variables.insert(1, "context")

    return PromptTemplate(template=PROMPT_TEMPLATE[model_family_name][prompt_id], input_variables=input_variables)
