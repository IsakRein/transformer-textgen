import json
import os
import sys

# Check that models are uploaded
if 'models' not in os.listdir('.'):
    raise 'No models directory found'

for model in ['rnn', 'lstm', 'transformer']:
    if f'{model}.py' not in os.listdir('./models'):
        raise f'No {model} model found'

# Check that tokenizers are uploaded
if 'tokenizers' not in os.listdir('.'):
    raise 'No tokenizers directory found'

for tokenizer in ['char_tokenizer.py', 'vector_tokenizer.py', 'bpe_tokenizer.py', 'bpe4_tokenizer.py']:
    if tokenizer not in os.listdir('./tokenizers'):
        raise f'No {tokenizer} tokenizer found'

# Create folder
if 'token_data' not in os.listdir('.'):
    os.mkdir('token_data')


files = os.listdir('./token_data')


# Generate tokens
for tokenizer in [
    'char',
    #'vector', Remove vector since it causes memory issues due to the sheer amount of words. 
    'bpe',
    'bpe4'
]:
    # Run tokenizer
    if f'train_vocabulary_{tokenizer}.pkl' not in files:
        print(f'Running {tokenizer} tokenizer')
        #os.system(f'venv/bin/python tokenizers/{tokenizer}_tokenizer.py')
        os.system(f'python3 tokenizers/{tokenizer}_tokenizer.py')

with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# Run model
print(f'\nRunning {config["model"]} model with config {sys.argv[1]}')
print('-' * 50)
os.system(f'python models/{config["model"]}.py {sys.argv[1]}')
