# 📱 Customer Feedback Intelligence

**🔗 Live app:** https://customer-feedback-intelligence-4jbuq6ns55hh8zmugt936v.streamlit.app

Turns 1,940 real ChatGPT iOS app reviews into prioritized, evidence-backed product insights — using modern NLP (sentence embeddings + BERTopic), not keyword counting.

---

## What it does
Analyzes unstructured review text to answer: **what are users complaining about, how much does it hurt, and what should the team fix first?**

## Approach
1. **Embeddings** — each review → a 384-dim meaning-vector (sentence-transformers, MiniLM).
2. **Theme discovery** — KMeans + **BERTopic** (auto-finds themes, labels them, isolates outliers).
3. **Quantify** — each theme's volume + average star rating → a **priority map** (big + angry = fix first).
4. **Trend analysis** — themes over time (normalized by weekly volume).
5. **LLM write-up** — a free LLM drafts recommendations from the computed numbers (analysis stays human-led).
6. **Deployment** — a live Streamlit app with an interactive "classify your review" feature.

## Key findings
- **Account/login problems** are the #1 pain point (169 reviews, **1.53★**).
- Crucially, login complaints **surged to ~40% of weekly reviews in early July** — a likely **regression from a late-June release** (visible only after normalizing for review volume; raw counts hid it).
- A distinct **app-freezing crash** (1.74★) and **overheating** (2.93★) round out the top bugs.
- Most-requested feature: an **iPad-optimized version**.

## ⚠️ Limitations
- Clustering has no ground truth; validated via rating separation (1.66–4.18) + manual review (silhouette is low, which is normal for text).
- "Release regression" is a data-supported hypothesis, not confirmed (no release history available).
- LLM recommendations are a draft; the human-written prioritization is the primary output.

## Tech stack
Python · sentence-transformers · BERTopic · scikit-learn · pandas · matplotlib · Streamlit
