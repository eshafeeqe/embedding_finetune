import time
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.stats import spearmanr
from datasets import load_dataset

# ---------------------------------------------------------
# Load STS-B dataset
# ---------------------------------------------------------
dataset = load_dataset("glue", "stsb")
TIMING_SAMPLES = 512  # use subset for timing

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def model_size_mb(model: SentenceTransformer) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)

def measure_inference_time(model: SentenceTransformer, sentences, batch_size=32, n_runs=3):
    """Return seconds per sentence."""
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # warmup
    _ = model.encode(sentences[:32], batch_size=32, convert_to_tensor=True, show_progress_bar=False)
    total_times = []
    for _ in range(n_runs):
        start = time.time()
        _ = model.encode(sentences, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        total_times.append(time.time() - start)
    avg_time = sum(total_times) / len(total_times)
    return avg_time / len(sentences)

def evaluate_spearman(model: SentenceTransformer, split="validation"):
    sents1 = dataset[split]["sentence1"]
    sents2 = dataset[split]["sentence2"]
    scores = dataset[split]["label"]
    if len(set(scores)) == 1 and scores[0] == -1.0:
        return None
    emb1 = model.encode(sents1, convert_to_tensor=True, show_progress_bar=False)
    emb2 = model.encode(sents2, convert_to_tensor=True, show_progress_bar=False)
    cos_scores = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
    rho, _ = spearmanr(cos_scores, scores)
    return rho

# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
print("Loading base model (all-MiniLM-L6-v2)...")
base_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading fine-tuned model (models/stsb-finetuned)...")
finetuned_model = SentenceTransformer("models/stsb-finetuned")

print("Loading big model (intfloat/e5-base-v2)...")
big_model = SentenceTransformer("intfloat/e5-base-v2")

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
print("\n=== Validation Set ===")
base_val = evaluate_spearman(base_model, "validation")
ft_val = evaluate_spearman(finetuned_model, "validation")
big_val = evaluate_spearman(big_model, "validation")

print(f"Base (MiniLM):     {base_val:.4f}")
print(f"Fine-tuned MiniLM: {ft_val:.4f}")
print(f"Large (E5-base):   {big_val:.4f}")
print(f"Δ fine-tuned vs base: {ft_val - base_val:+.4f}")

# ---------------------------------------------------------
# Inference speed
# ---------------------------------------------------------
print("\n=== Inference Speed (seconds per sentence) ===")
val_sents = dataset["validation"]["sentence1"][:TIMING_SAMPLES]
base_t = measure_inference_time(base_model, val_sents)
ft_t = measure_inference_time(finetuned_model, val_sents)
big_t = measure_inference_time(big_model, val_sents)

print(f"Base MiniLM:       {base_t*1000:.3f} ms/sentence")
print(f"Fine-tuned MiniLM: {ft_t*1000:.3f} ms/sentence")
print(f"Large E5-base:     {big_t*1000:.3f} ms/sentence")

# ---------------------------------------------------------
# Model size
# ---------------------------------------------------------
print("\n=== Model Size (approx, float32) ===")
base_mb = model_size_mb(base_model)
ft_mb = model_size_mb(finetuned_model)
big_mb = model_size_mb(big_model)
print(f"Base MiniLM:       {base_mb:.1f} MB")
print(f"Fine-tuned MiniLM: {ft_mb:.1f} MB")
print(f"Large E5-base:     {big_mb:.1f} MB")

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n=== Summary ===")
print(f"Spearman (val): base={base_val:.4f}, ft={ft_val:.4f}, big={big_val:.4f}")
print(f"Inference time: base={base_t*1000:.2f} ms, ft={ft_t*1000:.2f} ms, big={big_t*1000:.2f} ms")
print(f"Model size:     base={base_mb:.1f} MB, ft={ft_mb:.1f} MB, big={big_mb:.1f} MB")
