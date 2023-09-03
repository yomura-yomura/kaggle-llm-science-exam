from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = {
    1: """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]
#
# Question: {prompt}\n
# A) {a}\n
# B) {b}\n
# C) {c}\n
# D) {d}\n
# E) {e}\n
#
# ### Answer: {answer}""",
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

### Answer: {answer_text}"""
}


def get_prompt_template(prompt_id: int) -> PromptTemplate:
    if prompt_id < 3:
        input_variables = ["prompt", "a", "b", "c", "d", "e", "answer"]
    elif prompt_id in [3, 4]:
        input_variables = ["prompt", "answer_text"]
    else:
        raise NotImplementedError(prompt_id)

    return PromptTemplate(template=PROMPT_TEMPLATE[prompt_id], input_variables=input_variables)
