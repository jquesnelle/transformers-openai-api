from typing import Any, Mapping


class Metrics:

    data: Mapping[str, Any]

    def __init__(self) -> None:
        self.data = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_total_tokens': 0,
            'model_metrics': {}
        }

    def update(self, response: Mapping[str, Any]):
        if 'model' in response:
            model = response['model']
            if model not in self.data['model_metrics']:
                self.data['model_metrics'][model] = {
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0,
                    'total_total_tokens': 0,
                    'calls': 0
                }
            
            model_metrics = self.data['model_metrics'][model]
            model_metrics['calls'] += 1

            if 'usage' in response:
                usage = response['usage']
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)

                self.data['total_prompt_tokens'] += prompt_tokens
                self.data['total_completion_tokens'] += completion_tokens
                self.data['total_total_tokens'] += total_tokens

                model_metrics['total_prompt_tokens'] += prompt_tokens
                model_metrics['total_completion_tokens'] += completion_tokens
                model_metrics['total_total_tokens'] += total_tokens

    def get(self) -> Mapping[str, Any]:
        return self.data