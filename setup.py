from setuptools import find_packages, setup

setup(
    name='transformers-openai-api',
    packages=find_packages(),
    version='1.0.0',
    description='An OpenAI Completions API compatible server for locally running transformers models',
    author='Jeffrey Quesnelle <jq@jeffq.com>',
    license='MIT',
    packages=["transformers_openai_api"],
    install_requires=[
        'transformers',
        'accelerate',
        'torch',
        'waitress'
    ]
)
