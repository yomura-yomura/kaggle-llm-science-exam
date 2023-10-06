from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import DebertaV2ForMultipleChoice
from transformers.modeling_outputs import BaseModelOutput, MultipleChoiceModelOutput

__all__ = ["enable_memory_efficient_forward_method"]


def enable_memory_efficient_forward_method():
    DebertaV2ForMultipleChoice.forward = forward


def forward(
    self: DebertaV2ForMultipleChoice,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    batch: int = 1,
) -> Union[Tuple, MultipleChoiceModelOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
        num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
        `input_ids` above)
    """

    if getattr(self, "never_print_custom_forward_function_msg", True):
        print(f"using custom forward function.")
        self.never_print_custom_forward_function_msg = False

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
    flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
    flat_inputs_embeds = (
        inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
    )

    assert batch == 1

    last_hidden_state = []
    hidden_states = []
    attentions = []
    for idx in range(len(flat_input_ids)):
        outputs = self.deberta(
            flat_input_ids[[idx]] if flat_input_ids is not None else None,
            position_ids=flat_position_ids[[idx]] if flat_position_ids is not None else None,
            token_type_ids=flat_token_type_ids[[idx]] if flat_token_type_ids is not None else None,
            attention_mask=flat_attention_mask[[idx]] if flat_attention_mask is not None else None,
            inputs_embeds=flat_inputs_embeds[[idx]] if flat_inputs_embeds is not None else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state.append(outputs.last_hidden_state)
        if outputs.hidden_states is not None:
            hidden_states.append(outputs.hidden_states)
        if outputs.attentions is not None:
            attentions.append(outputs.attentions)
        del outputs

    last_hidden_state = torch.concat(last_hidden_state, dim=0)
    hidden_states = torch.concat(hidden_states, dim=0) if len(hidden_states) > 0 else None
    attentions = torch.concat(attentions, dim=0) if len(attentions) > 0 else None
    outputs = BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=attentions)

    encoder_layer = outputs[0]
    pooled_output = self.pooler(encoder_layer)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    reshaped_logits = logits.view(-1, num_choices)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

    if not return_dict:
        output = (reshaped_logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return MultipleChoiceModelOutput(
        loss=loss,
        logits=reshaped_logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
