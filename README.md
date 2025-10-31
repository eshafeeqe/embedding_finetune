# Domain-Adapted Sentence Embedding (STS Benchmark Fine-Tuning)

## 🧭 Objective

This project fine-tunes a **sentence embedding model** on the **Semantic Textual Similarity Benchmark (STS-B)** dataset to improve how well the model captures **semantic similarity between sentences**.

The goal is to:
- Learn embeddings that better align with human judgments of similarity (scores 0–5).  
- Quantitatively measure improvement over the base pretrained model.  
- Demonstrate fine-tuning, evaluation, and visualization of embedding space.

---

## 🧠 Background

Sentence embedding models convert sentences into numerical vectors.  
Cosine similarity between two vectors represents how semantically related the sentences are.

Generic models like `all-MiniLM-L6-v2` perform well on general text, but fine-tuning them on STS-B helps the model:
- Better align with human-rated semantic similarity,
- Serve as a foundation for retrieval or clustering tasks,
- Demonstrate adaptation of pretrained models to a similarity-scoring objective.

## 💡 Why This Matters

Large Language Models (LLMs) rely on **embedding spaces** to retrieve, compare, or contextualize information — for example, in RAG (Retrieval-Augmented Generation) systems or context-grounded chatbots.

Fine-tuning embedding models on **domain-specific similarity data** ensures:
- The model learns how *that domain* expresses equivalence or relatedness (e.g., legal, clinical, financial, or technical terms).  
- Retrieval systems built on top of these embeddings return **contextually correct** documents.  
- Smaller models, after fine-tuning, can match larger models’ accuracy — allowing **cost-efficient deployment** on constrained environments.

Thus, this experiment demonstrates a key principle of applied AI/ML:
> *Fine-tuning a small model on targeted domain or task data can outperform a larger general model without additional hardware cost.*

## ⚙️ How to Run

### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- CUDA-compatible GPU (recommended for faster training)

### Hardware Used
This project was trained and evaluated on:
- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **CUDA Version**: 12.2

> **Note**: Training can also run on CPU, but will be significantly slower. The model is small enough to train on GPUs with 8GB+ VRAM.

### Setup Steps

1. **Create a virtual environment**

   Using Python's built-in venv:
   ```bash
   python -m venv .venv
   ```

   Or using uv (faster):
   ```bash
   uv venv
   ```

2. **Activate the virtual environment**

   On Linux/macOS:
   ```bash
   source .venv/bin/activate
   ```

   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   Using uv (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Fine-tune the model**
   ```bash
   python train_stsb.py
   ```

5. **Evaluate and benchmark**
   ```bash
   python benchmark_stsb.py
   ```

### Deactivating the Virtual Environment
When you're done, deactivate the virtual environment:
```bash
deactivate
```
## 📊 Results

### **Evaluation Overview**
All models were evaluated on the **STS-Benchmark (validation set)** using **Spearman correlation** between cosine similarity of sentence embeddings and human similarity scores.

| Model | Spearman (Validation) | Δ vs Base | Model Size (MB) | Inference Time (ms/sentence) |
|--------|-----------------------|-----------|------------------|------------------------------|
| **all-MiniLM-L6-v2** (Base) | 0.8672 | — | 86.6 | 0.098 |
| **Fine-tuned MiniLM (STS-B)** | **0.8935** | **+0.0263** | 86.6 | 0.097 |
| **intfloat/e5-base-v2** (Large Model) | 0.8826 | +0.0154 vs Base | 417.6 | 0.227 |

---

### **Key Observations**

1. **Fine-tuning improves alignment with human similarity judgments.**  
   - The fine-tuned MiniLM model achieved **+0.026 Spearman** improvement over the pretrained MiniLM baseline.  
   - This shows that even a small additional STS-B training pass helps the model better capture sentence-level semantics.

2. **Small model matches a larger one.**  
   - Fine-tuned MiniLM (86 MB) slightly **outperforms** the much larger E5-base (418 MB).  
   - This demonstrates that task-specific fine-tuning can close most of the performance gap with larger models.

3. **Inference is significantly faster.**  
   - Fine-tuned MiniLM runs at **~0.10 ms/sentence**, compared to **0.23 ms/sentence** for E5-base.  
   - That’s roughly **2.3× faster** inference, with **5× smaller model size**.

4. **Practical implication.**  
   - A fine-tuned lightweight model can achieve **near–state-of-the-art performance** for semantic similarity while being **deployable on lower-cost hardware (edge GPUs or CPUs)**.

---

### **Summary**
> Fine-tuning a compact model (MiniLM) on STS-B yields a **3% correlation improvement** and reaches **equal or better performance** than a model ~5× larger, while maintaining **~2× faster inference**.  
> This validates that **targeted fine-tuning** can efficiently adapt small language models to high-quality semantic tasks.

---
