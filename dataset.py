from datasets import load_dataset

# Load STS-B
dataset = load_dataset("glue", "stsb")

print(dataset)
# -> DatasetDict({train: ..., validation: ..., test: ...})

# View a few examples
print(dataset["train"][0])
