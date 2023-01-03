from setuptools import setup

setup(
    name='transformers-openai-api',
    packages=["transformers_openai_api"],
    version='1.0.0',
    description='An OpenAI Completions API compatible server for locally running transformers models',
    author='Jeffrey Quesnelle <jq@jeffq.com>',
    license='MIT',
    install_requires=[
        'transformers',
        'accelerate',
        'torch',
        'waitress'
    ]
)
