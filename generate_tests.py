# TODO: generera .json-filer för varje testfall
import json

for i in range(1):
    with open(f'tests/test{i}.json', 'w') as f:
        f.write(json.dumps({
            "batch_size": 16,
            "seq_length": 25,
            "max_iters": 2000,
            "eval_interval": 100,
            "learning_rate": 1e-3,
            "eval_iters": 200,
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 4,
            "dropout": 0.1,
            "feed_forward_multiplier": 4,
            "max_new_tokens": 200,
            "d_ff": 256
        }))
