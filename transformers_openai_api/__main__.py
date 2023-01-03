import argparse
import os
import sys
from .app import make_transformers_openai_api
from .serve import run_server

def main():
    parser = argparse.ArgumentParser(
        prog='transformers-openai-api',
        description='An OpenAI Completions API compatible server for locally running transformers models')
    parser.add_argument('config', nargs='?', help='Path to config.json',
                        default=os.path.join(os.getcwd(), 'config.json'))
    args = parser.parse_args()

    run_server(make_transformers_openai_api(args.config))

if __name__ == '__main__':
    sys.exit(main())
