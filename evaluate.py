from sentence_transformers import SentenceTransformer, util
from scipy.stats import spearmanr
from datasets import load_dataset
import numpy as np

dataset = load_dataset("glue", "stsb")

def evaluate(model, split="validation"):
    sents1 = dataset[split]["sentence1"]
    sents2 = dataset[split]["sentence2"]
    scores = dataset[split]["label"]

    # Check if labels are available (not all -1)
    if len(set(scores)) == 1 and scores[0] == -1.0:
        return None

    emb1 = model.encode(sents1, convert_to_tensor=True)
    emb2 = model.encode(sents2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()

    rho, _ = spearmanr(cosine_scores, scores)
    return rho

# Load models
print("Loading base model...")
base_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading fine-tuned model...")
finetuned_model = SentenceTransformer("models/stsb-finetuned")

# Evaluate on validation set
print("\n=== Validation Set ===")
base_val_score = evaluate(base_model, "validation")
finetuned_val_score = evaluate(finetuned_model, "validation")
print(f"Base model Spearman correlation: {base_val_score:.4f}")
print(f"Fine-tuned model Spearman correlation: {finetuned_val_score:.4f}")
print(f"Improvement: {(finetuned_val_score - base_val_score):.4f}")

# Evaluate on test set
print("\n=== Test Set ===")
base_test_score = evaluate(base_model, "test")
if base_test_score is None:
    print("Test labels are not publicly available (all labels are -1).")
    print("Evaluation can only be done on the validation set.")
else:
    finetuned_test_score = evaluate(finetuned_model, "test")
    print(f"Base model Spearman correlation: {base_test_score:.4f}")
    print(f"Fine-tuned model Spearman correlation: {finetuned_test_score:.4f}")
    print(f"Improvement: {(finetuned_test_score - base_test_score):.4f}")
