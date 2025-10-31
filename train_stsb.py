from datasets import load_dataset
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

dataset = load_dataset("glue", "stsb")
train_samples = [
    InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["label"]/5.0)
    for row in dataset["train"]
]


model = SentenceTransformer("all-MiniLM-L6-v2")
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model=model)

model.fit(
    [(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    show_progress_bar=True
)

model.save("models/stsb-finetuned")
