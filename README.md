# 🎬 YouTube RAG Pipeline

A full end-to-end **Retrieval-Augmented Generation (RAG)** system that takes a YouTube video link, extracts the transcript, chunks it with precise token tracking, stores embeddings in **Supabase (PostgreSQL + pgvector)**, and answers natural language questions about the video using OpenAI.

---

## 🏗️ Architecture

```
YouTube URL
    │
    ▼
[transcript.py] ──► youtube-transcript-api ──► raw transcript segments
    │
    ▼
[chunker.py] ──► tiktoken sliding-window chunking (500 tok, 50 overlap)
    │
    ▼
[embedder.py] ──► OpenAI text-embedding-3-small (1536-dim)
    │
    ▼
[vector_store.py] ──► Supabase pgvector (video_chunks table)
    │
    ▼
User Query ──► embed ──► similarity_search ──► [qa_engine.py] ──► GPT-4o-mini
```

---

## 📋 Features

| Feature | Details |
|---|---|
| Transcript Extraction | Auto-fetches captions; falls back to manual paste |
| Token-Aware Chunking | Exact token counting via `tiktoken` (cl100k_base) |
| Sliding Window | 500-token chunks, 50-token overlap (configurable) |
| Vector Storage | Supabase PostgreSQL + `pgvector` with IVFFlat index |
| Duplicate Detection | Skips re-indexing if video already in DB |
| Irrelevant Detection | LLM returns "question is irrelevant" if context doesn't match |
| Rich UI | Dark glassmorphism Streamlit app |

---

## 🚀 Setup

### 1. Supabase Schema (run once)

Open your Supabase project → **SQL Editor** → paste and run `schema.sql`:

```sql
-- Enables pgvector, creates video_chunks table + IVFFlat index + match function
```

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key
```

> **Note**: `SUPABASE_DB_URL` is only required if you use `vecs` directly. The app uses the Supabase Python client, so just `SUPABASE_URL` + `SUPABASE_KEY` are sufficient.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
RAG for YTV/
├── app.py                  # Streamlit UI (entry point)
├── schema.sql              # Run once in Supabase SQL Editor
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── transcript.py       # YouTube transcript extraction
    ├── chunker.py          # Token-aware sliding window chunking
    ├── embedder.py         # OpenAI embeddings with retry
    ├── vector_store.py     # Supabase pgvector CRUD + similarity search
    └── qa_engine.py        # RAG Q&A with LLM
```

---

## 💡 Usage

1. **Paste a YouTube URL** in the app and click **Process**
   - The transcript is fetched, chunked, embedded, and stored in Supabase automatically
   - If the video was already processed, it skips re-indexing

2. **Type your question** and click **Ask**
   - The app performs cosine similarity search over the stored chunks
   - The LLM generates an answer grounded strictly in the video transcript
   - If your question is off-topic, it replies: *"The question is irrelevant to this video."*

---

## ⚙️ Configuration

All values can be set in `.env`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE_TOKENS` | `500` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |

You can also adjust `top_k` and chunk size live in the app's sidebar.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| `TranscriptsDisabled` | Video has captions disabled — try another video |
| `NoTranscriptFound` | No English captions — try a video with auto-generated captions |
| `SUPABASE_URL` not set | Copy `.env.example` to `.env` and fill in values |
| Supabase `vector` type error | Run `schema.sql` in your Supabase SQL Editor first |
| `ivfflat` index error | This index needs ~100+ rows; ignore for small datasets |
