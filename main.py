import json
import os
import sys

if 'token_data' not in os.listdir('.'):
    os.mkdir('token_data')

files = os.listdir('./token_data')

# Generate tokens
for tokenizer in [
    'char',
    'vector',
    'bpe',
    'bpe4'
]:
    # Run tokenizer
    if f'vocabulary_{tokenizer}.pkl' not in files:
        print(f'Running {tokenizer} tokenizer')
        os.system(f'venv/bin/python tokenizers/{tokenizer}_tokenizer.py')

with open(sys.argv[1], 'r') as f:
    config = json.load(f)


# Run model
print(f'\nRunning {config["model"]} model with config {sys.argv[1]}')
print('-' * 50)
os.system(f'venv/bin/python models/{config["model"]}.py {sys.argv[1]}')
