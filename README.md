# 🛰️ AI Truth Checker

Verify factual claims against your own documents using a Retrieval-Augmented pipeline with cross-encoder reranking and Natural Language Inference — not a generative LLM guessing from memory, but a traceable, evidence-based verdict.

Upload your own PDF/TXT documents, type a claim, and get back a **SUPPORTED / CONTRADICTED / UNVERIFIED** verdict with the exact evidence sentence and confidence scores behind it.

---

## ✨ Features

- 📂 **Document management** — upload, index, and delete PDF/TXT documents directly in the app, with embeddings kept in sync (deleting a file removes its vectors too).
- 🔍 **Claim verification** — enter any factual claim and get a verdict grounded in your indexed documents.
- 🔗 **Context Match Score** — a headline percentage showing how closely your claim matches content you've actually uploaded.
- ⚙️ **Full pipeline transparency** — see the retrieved chunks, the selected evidence sentence, and the raw NLI probability breakdown behind every verdict.
- 🧠 **Domain-agnostic decision engine** — no hardcoded subject-matter rules; works the same whether you index space facts, legal text, or financial reports.

---

## 🏗️ Architecture

```
Upload PDF/TXT
      │
      ▼
DocumentLoader → TextChunker → BGE Embeddings → ChromaDB
                                                     │
                                                     ▼
Claim ──► Retriever (semantic search) ──► Cross-Encoder Reranker
                                                     │
                                                     ▼
                                        Evidence Sentence Selection
                                                     │
                                   ┌─────────────────┴─────────────────┐
                                   ▼                                   ▼
                       DeBERTa-v3 NLI (entailment/            Cosine Similarity
                        contradiction/neutral)                 (claim vs evidence)
                                   │                                   │
                                   └─────────────────┬─────────────────┘
                                                      ▼
                                            Decision Engine (rules)
                                                      │
                                                      ▼
                                    Verdict + Confidence + Evidence
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Vector database | [ChromaDB](https://www.trychroma.com) (persistent, HNSW index) |
| Embedding model | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| Reranker | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| NLI model | [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) |
| PDF parsing | [pypdf](https://pypi.org/project/pypdf/) |

---

## 📁 Project Structure

```
app.py                          Streamlit UI — upload, index, verify, delete
requirements.txt
data/
  documents/                    uploaded source files
  chroma_db/                    persistent vector index (created at runtime)
src/
  ingestion/
    document_loader.py          reads .txt / .pdf files into raw text
    text_chunker.py              sentence-based chunking with overlap
    ingestion_pipeline.py       load → chunk → embed → store
  retrieval/
    retriever.py                 embeds queries and searches ChromaDB
  reranking/
    reranker.py                  cross-encoder relevance reranking
  verification/
    evidence_selector.py         picks the single best evidence sentence
    nli_verifier.py               entailment/contradiction/neutral classification
    similarity_scorer.py         cosine similarity between claim and evidence
    decision_engine.py            rule-based verdict logic
    hybrid_verifier.py            orchestrates the full verification pipeline
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- ~2–3 GB free disk space (for model downloads on first run)
- Internet access on first run (models are pulled from Hugging Face and cached locally afterward)

### Installation

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Usage

1. **Upload documents** — go to *Document Management*, upload one or more `.pdf`/`.txt` files, and click **💾 Save Uploaded Files**.
2. **Index them** — click **⚡ Index Documents** to chunk, embed, and store them in ChromaDB.
3. **Verify a claim** — type a factual claim under *Claim Verification* and click **🔍 Verify Claim**.
4. **Review the result** — the Result tab shows the Context Match Score, verdict, and confidence; the Evidence and Pipeline tabs show the exact sentence and retrieved chunks behind the verdict.

---

## ⚠️ Known Limitations

- Decision thresholds (`decision_engine.py`) are reasonable defaults, not empirically calibrated against a labeled test set.
- Sentence splitting is regex/character-based and doesn't handle abbreviations (`Dr.`, `U.S.`, etc.) perfectly.
- Every "Index Documents" click rebuilds the entire vector store from scratch — no incremental indexing yet.
- No authentication — don't deploy publicly without adding access control, since anything indexed becomes queryable by any app user.
- No multi-hop reasoning — verification uses a single best evidence sentence, not combined facts across multiple chunks.

---

## 🗺️ Roadmap / Ideas for Contribution

- [ ] GPU device placement for NLI/reranker models
- [ ] Incremental indexing (only embed new files)
- [ ] Configurable thresholds via the UI (sidebar sliders)
- [ ] Claim history within a session
- [ ] A proper evaluation harness scoring verdict accuracy, not just retrieval recall

---

