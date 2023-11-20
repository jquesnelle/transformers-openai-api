# transformers-openai-api

`transformers-openai-api` is a server for hosting locally running NLP [transformers](https://github.com/huggingface/transformers/) models via the [OpenAI Completions API](https://beta.openai.com/docs/api-reference/completions). In short, you can run `transformers` models and offer them through an API compatible with existing OpenAI tooling such as the [OpenAI Python Client](https://github.com/openai/openai-python) itself or any package that uses it (e.g. [LangChain](https://github.com/hwchase17/langchain)).

## Quickstart

### From pip

```sh
pip install transformers-openai-api
wget https://raw.githubusercontent.com/jquesnelle/transformers-openai-api/master/config.example.json
mv config.example.json config.json
transformers-openai-api
```

### From source

```sh
git clone https://github.com/jquesnelle/transformers-openai-api
cd transformers-openai-api
cp config.example.json config.json
pip install -r requirements.txt
python -m transformers_openai_api
```

## Using with OpenAI Python Client

Simply set the environment variable `OPENAI_API_BASE` to `http://HOST:PORT/v1` before importing the `openai` package. For example, to access a local instance of `transformers-openai-api`, set `OPENAI_API_BASE` to `http://127.0.0.1:5000/v1`. Alternatively, you can set the `api_base` property on the `openai` object:

```python
import openai
openai.api_base = 'http://HOST:PORT/v1'
```

## Configuration

All configuration is managed through `config.json`. By default `transformers-openai-api` looks for this file the in the current working directory, however a different path can be passed as the command-line argument to the program.  See [config.example.json](config.example.json).

### Hosting

By default the API server listens on `127.0.0.1:5000` to change this, add a `HOST` and/or `PORT` entries to the configuration file. For example to serve publicly:
```json
{
    "HOST": "0.0.0.0",
    "PORT": 80
}
```

### Models

The `MODELS` object handles mapping an OpenAI model name to a `transformers` model configuration. The structure of a model configuration is:
| Key | Description |
| - | - |
| `ENABLED` | Boolean value to disable a model |
| `TYPE` | Either "Seq2Seq" or "CausalLM" |
| `MODEL_CONFIG` | Parameters for model creation; passed to `AutoModelForTYPE.from_pretrained` |
| `MODEL_DEVICE` | Convert model to this device; passed to `to` called on the created model (default `cuda`) |
| `TOKENIZER_CONFIG` | Parameters for tokenizer creation; passed to `AutoTokenizer.from_pretrained` |
| `TOKENIZER_DEVICE` | Convert tokens to this device; passed to `to` called on the tokenized input (default `cuda`) |
| `GENERATE_CONFIG` | Parameters for generation; passed to the model's `generate` function |
| `DECODE_CONFIG` | Parameters for decoding; passed to the tokenizer's `decode` function |

#### Using accelerate

To use [accelerate](https://github.com/huggingface/accelerate), set `device_map` on the `MODEL_CONFIG` to `auto` and explicitly set `MODEL_DEVICE` to `null`. The default `text-davinci-003` model in [config.example.json](config.example.json) is an example of this.

#### Using CPU

To switch to CPU inference, set `MODEL_DEVICE` and `TOKENIZER_DEVICE` to `cpu`.

#### Using FP16

To use a model at half-precision, set `torch_dtype` on the `MODEL_CONFIG` to `torch_dtype`. The disabled `text-curie-001` model in [config.example.json](config.example.json) is an example of this.

### Authorization

To limit access to the API (i.e. enforcing `OPENAI_API_KEY`), fill in the `BEARER_TOKENS` object with a list of authorized tokens (e.g. your OpenAI key). If the `BEARER_TOKENS` list does not exist, no authorization will be enforced.
```json
{
    "BEARER_TOKENS": ["sk-..."]
}
```
