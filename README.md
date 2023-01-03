# transformers-openai-api

`transformers-openai-api` is a server for serving locally running [transformers](https://github.com/huggingface/transformers/) models via the [OpenAI Completions API](https://beta.openai.com/docs/api-reference/completions). In short, you can run any `transformers` model and offer it through an API compatible with existing OpenAI tooling such as the [OpenAI Python Client](https://github.com/openai/openai-python) itself or any package that uses it (e.g. [LangChain](https://github.com/hwchase17/langchain)).

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
python transformers_openai_api/
```

## Using with OpenAI Python Client

Simply set the environment variable `OPENAI_API_BASE` to `http://HOST:PORT/v1` before importing the `openai` package. For example, to access a local instance of `transformers-openai-api`, set `OPENAI_API_BASE` to `http://127.0.0.1:5000/v1`. Alternatively, you can set the `api_base` property on the `openai` object

```python
import openai
openai.api_base = 'http://HOST:PORT/v1'
```
