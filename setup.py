from setuptools import setup

setup(
    name='transformers-openai-api',
    packages=["transformers_openai_api"],
    version='1.0.0',
    description='An OpenAI Completions API compatible server for NLP transformers models',
    author='Jeffrey Quesnelle <jq@jeffq.com>',
    license='MIT',
    install_requires=[
        'transformers',
        'accelerate',
        'torch',
        'Flask'
    ],
    entry_points={
        'console_scripts': [
            'transformers-openai-api = transformers_openai_api.__main__:main'
        ]
    }
)
