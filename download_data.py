from datasets import load_dataset

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


def save_to_txt(split, filename):
    totalLength = 0
    with open(filename, 'w', encoding='utf-8') as f:
        for item in split:
            totalLength += len(item['text'])
            f.write(item['text'])
    print(f"Saved {totalLength} characters to {filename}")


# Save each split to a separate text file
save_to_txt(dataset['train'], "data/train.txt")
save_to_txt(dataset['validation'], "data/validation.txt")
save_to_txt(dataset['test'], "data/test.txt")
