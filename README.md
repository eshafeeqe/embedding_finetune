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