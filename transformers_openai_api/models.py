import logging
import torch

from abc import ABC
from typing import Any, List, Mapping
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Inference device: {device}")

def get_prompts(request: Mapping[str, Any]) -> List[str]:
    prompt = request['prompt']
    if isinstance(prompt, str):
        prompt = [prompt]
    return prompt


def _completions_auto(
        request: Mapping[str, Any],
        tokenizer: Any,
        model: Any,
        generate_config: Mapping[str, Any],
        decode_config: Mapping[str, Any],
        auto_echo: bool):
    generate_args = {}
    generate_args.update(generate_config)
    generate_args.update(request)

    decode_args = {
        "skip_special_tokens": True
    }
    decode_args.update(decode_config)

    if ('top_p' in generate_args or 'top_k' in generate_args or 'temperature' in generate_args) and 'do_sample' not in generate_args:
        generate_args['do_sample'] = True
        if generate_args.get('temperature', 1.0) == 0:
            generate_args.pop('temperature', None)
        elif generate_args.get('top_p', 1.0) == 1.0:
            generate_args.pop('top_p', None)
        if 'top_k' not in generate_args:
            generate_args['top_k'] = 0

    prompts = get_prompts(generate_args)
    echo = generate_args.get('echo', False)
    n = generate_args.get('n', 1)

    generate_args.pop('model', None)
    generate_args.pop('prompt', None)
    generate_args.pop('n', None)

    # TODO
    generate_args.pop('best_of', None)
    generate_args.pop('presence_penalty', None)
    generate_args.pop('frequency_penalty', None)

    inputs = []
    prompt_tokens_count = 0
    for prompt in prompts:
        input = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_tokens_count += input.size(dim=1)
        inputs.append(input)

    choices = []
    completion_tokens_count = 0
    for i in range(0, len(inputs)):
        for _ in range(0, n):
            output = model.generate(inputs[i], **generate_args)[0]
            completion_tokens_count += len(output)
            text = tokenizer.decode(output, **decode_args)
            if echo and not auto_echo:
                text = prompts[i] + text
            choices.append({
                'text': text,
                'index': i,
            })

    return {
        'choices': choices,
        'usage': {
            'prompt_tokens': prompt_tokens_count,
            'completion_tokens': completion_tokens_count,
            'total_tokens': prompt_tokens_count + completion_tokens_count
        }
    }


class Model(ABC):

    def completions(self, request: Mapping[str, Any]):
        pass


class Seq2Seq(Model):
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    generate_config: Mapping[str, Any]
    decode_config: Mapping[str, Any]

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            model_config: Mapping[str, Any],
            tokenizer_config: Mapping[str, Any],
            generate_config: Mapping[str, Any],
            decode_config: Mapping[str, Any]) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path, **model_config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_config)
        self.generate_config = generate_config
        self.decode_config = decode_config

    def completions(self, request) -> List[str]:
        return _completions_auto(request, self.tokenizer, self.model, self.generate_config, self.decode_config, False)


class CausalLM(Model):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generate_config: Mapping[str, Any]
    decode_config: Mapping[str, Any]

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            model_config: Mapping[str, Any],
            tokenizer_config: Mapping[str, Any],
            generate_config: Mapping[str, Any],
            decode_config: Mapping[str, Any]) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_config)
        self.generate_config = generate_config
        self.decode_config = decode_config

    def completions(self, request) -> List[str]:
        return _completions_auto(request, self.tokenizer, self.model, self.generate_config, self.decode_config, False)
