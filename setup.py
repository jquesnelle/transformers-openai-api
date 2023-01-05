from setuptools import setup

setup(
    name='transformers-openai-api',
    packages=["transformers_openai_api"],
    version='1.0.2',
    description='An OpenAI Completions API compatible server for NLP transformers models',
    author='Jeffrey Quesnelle',
    author_email='jq@jeffq.com',
    url='https://github.com/jquesnelle/transformers-openai-api/',
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
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
