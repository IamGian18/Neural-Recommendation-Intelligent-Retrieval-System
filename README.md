# End-to-End E-Commerce Recommendation System (Two-Tower + FAISS + LightGBM)

Sistema de recomendación multietapa (*Retrieval & Ranking*) construido sobre el dataset de comportamiento real de **Retailrocket E-Commerce**. El pipeline resuelve el problema de recomendación a escala combinando recuperación vectorial densa mediante redes neuronales **Two-Tower** indexadas en **FAISS** y re-ranking supervisado mediante **Learning-to-Rank (LightGBM + Optuna)**.

---

## Arquitectura del Pipeline

```text
[Raw Events & Item Properties]
        │
        ▼
1. Data Processing & Sessionization (K-Core filtering, temporal split sin leakage)
        │
        ▼
2. Behavioral Feature Engineering (RFM-like recency, conversion rates, category affinity)
        │
        ├──► 3. Baselines (Popularity, Item-Based CF, TruncatedSVD)
        │
        ▼
4. Candidate Generation (Two-Tower PyTorch Net + In-Batch Contrastive Loss)
        │
        ▼
5. Vector Retrieval (FAISS IndexFlatIP -> Top-100 candidatos en <1 ms)
        │
        ▼
6. Learning-to-Rank (LightGBM LambdaMART + Optuna Tuning + SHAP Explainability)
        │
        ▼
7. Top-10 Personalized Recommendations
