# 🎬 YouTube RAG Pipeline

A full end-to-end **Retrieval-Augmented Generation (RAG)** system that takes a YouTube video link, extracts the transcript, chunks it with token tracking, stores embeddings in **Supabase (PostgreSQL + pgvector)**, and answers natural-language questions about the video using OpenAI.

---

## 🏗️ System Architecture

```
YouTube URL
    │
    ▼
[src/transcript.py] ──► youtube-transcript-api v1.x
    │                    Extracts timed caption segments
    ▼
[src/chunker.py] ──► tiktoken (cl100k_base)
    │                Sentence-aware sliding window chunks (≤500 tokens each)
    ▼
[src/embedder.py] ──► OpenAI text-embedding-3-small
    │                  1536-dimensional vectors, batched in groups of 100
    ▼
[src/vector_store.py] ──► Supabase PostgreSQL + pgvector
    │                      video_chunks table, IVFFlat cosine index
    │
    │   (User asks a question)
    │
    ▼
[src/qa_engine.py] ──► embed query → similarity search → GPT-4o-mini
    │                   Returns grounded answer or "irrelevant" response
    ▼
[app.py] ──► Streamlit UI
```

---

## � Project Structure

```
RAG for YTV/
│
├── app.py                        # Streamlit web UI (entry point)
├── schema.sql                    # Run ONCE in Supabase SQL Editor
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables
├── .gitignore                    # Excludes .env, venv, __pycache__, etc.
├── README.md                     # This file
│
├── .streamlit/
│   ├── config.toml               # Dark theme config for Streamlit
│   └── secrets.toml.example      # Template for Streamlit Cloud secrets
│
└── src/
    ├── __init__.py
    ├── transcript.py             # YouTube transcript extraction
    ├── chunker.py                # Sentence-aware token chunking
    ├── embedder.py               # OpenAI embedding generation
    ├── vector_store.py           # Supabase pgvector CRUD
    └── qa_engine.py              # RAG Q&A pipeline
```

---

## 📄 File-by-File Reference

---

### `schema.sql`
**Run this once in your Supabase SQL Editor before launching the app.**

What it does:
- Enables the `pgvector` extension on your Supabase PostgreSQL database
- Creates the `video_chunks` table with all required columns
- Creates an `ivfflat` index for fast cosine-similarity search
- Creates the `match_video_chunks` SQL function (called via `supabase.rpc()`)

**`video_chunks` table columns:**

| Column | Type | Description |
|---|---|---|
| `id` | bigserial | Auto-increment primary key |
| `video_id` | text | YouTube video ID (e.g. `dQw4w9WgXcQ`) |
| `video_title` | text | Title provided by the user |
| `video_url` | text | Full YouTube URL |
| `chunk_index` | int | Chunk sequence number within the video |
| `chunk_text` | text | The actual transcript text of this chunk |
| `token_count` | int | Exact token count (via tiktoken) |
| `start_time_sec` | float | Approximate timestamp of the chunk in the video |
| `embedding` | vector(1536) | OpenAI embedding for cosine similarity search |
| `created_at` | timestamptz | Auto-set insert timestamp |

**`match_video_chunks` SQL function:**  
PostgreSQL function called via `supabase.rpc()`. Takes a query embedding + video_id + match_count, returns the top-k most similar chunks ranked by cosine similarity.

---

### `src/transcript.py`
**Purpose:** Extract timed transcript segments from any YouTube video.

#### Functions

---

**`get_video_id(url: str) → str`**

Parses a YouTube URL and extracts the 11-character video ID.

Supported URL formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://www.youtube.com/shorts/VIDEO_ID`

Raises `ValueError` if no valid video ID is found.

Example:
```python
get_video_id("https://youtu.be/dQw4w9WgXcQ")
# → "dQw4w9WgXcQ"
```

---

**`fetch_transcript(video_id: str, languages: list[str] | None = None) → list[dict]`**

Fetches the transcript for a YouTube video using `youtube-transcript-api` v1.x.

- Tries to fetch in `["en", "en-US", "en-GB"]` first (direct path)
- Falls back to listing all available transcripts and translating to English if English isn't available
- Returns a list of segment dicts:
  ```python
  [{"text": "Hey there", "start": 0.0, "duration": 1.54}, ...]
  ```
- Raises `TranscriptsDisabled` if the video owner has disabled captions
- Raises `ValueError` for any other fetch failure

---

**`segments_to_full_text(segments: list[dict]) → str`**

Joins all segment texts into a single concatenated string. Utility function used for debugging or manual transcript inspection.

---

**`get_video_metadata(video_id: str) → dict`**

Returns a lightweight metadata dict `{"video_id": ..., "video_url": ...}`. Used to build the YouTube URL without making any additional API call.

---

### `src/chunker.py`
**Purpose:** Split transcript segments into sentence-complete, token-bounded chunks. Uses `tiktoken` for exact token counting.

#### Key Constant

| Constant | Value | Meaning |
|---|---|---|
| `ENCODING_NAME` | `cl100k_base` | Same tokenizer used by `text-embedding-3-small` and GPT-4o |
| `OVERLAP_SENTENCES` | `2` | Sentences from end of previous chunk repeated at start of next |

#### Data Class

**`Chunk`** — a dataclass representing a single chunk:

| Field | Type | Description |
|---|---|---|
| `video_id` | str | YouTube video ID |
| `video_title` | str | Human-readable video title |
| `video_url` | str | Full YouTube URL |
| `chunk_index` | int | Position of this chunk in the video |
| `chunk_text` | str | The full text of the chunk (whole sentences only) |
| `token_count` | int | Exact token count via tiktoken |
| `start_time_sec` | float | Approx. video timestamp where this chunk begins |

#### Functions

---

**`count_tokens(text: str) → int`**

Returns the exact token count for any string using the `cl100k_base` tokenizer. Used internally and exposed for UI display (showing how many tokens a chunk contains).

---

**`_split_sentences(text: str) → list[str]`** *(internal)*

Splits text into individual sentences using a regex that detects `.`, `!`, `?` followed by whitespace. No external NLP library required. Returns only non-empty sentence strings.

---

**`chunk_transcript(segments, video_id, video_title, video_url, chunk_size, overlap, overlap_sentences) → list[Chunk]`**

The core chunking function. Implements **sentence-aware sliding window chunking**:

1. Flattens all transcript segments into a list of `(sentence, start_time_sec)` pairs
2. Greedily accumulates whole sentences until the next sentence would exceed `chunk_size` tokens
3. Closes the current chunk and starts a new one at a sentence boundary (no mid-sentence cuts)
4. Carries the last `overlap_sentences` (default: 2) sentences into the start of the next chunk to preserve cross-boundary context

> **Why sentence-aware?**  
> Raw token-slicing cuts text arbitrarily (e.g. `"...attention mecha|nisms..."`).  
> Sentence-aware chunking ensures the LLM always receives complete, coherent sentences — improving retrieval and answer quality.

Parameters:
- `chunk_size` (default `500`) — max tokens per chunk
- `overlap_sentences` (default `2`) — sentences of overlap between consecutive chunks

---

**`chunks_to_dicts(chunks: list[Chunk]) → list[dict]`**

Converts a list of `Chunk` dataclasses into plain Python dicts. Required before inserting into Supabase via the `vector_store` module.

---

### `src/embedder.py`
**Purpose:** Generate embeddings for text using OpenAI's `text-embedding-3-small` model. Handles batching and automatic retry.

#### Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Default OpenAI embedding model |
| `EMBEDDING_DIM` | `1536` | Vector dimension (fixed for this model) |
| `BATCH_SIZE` | `100` | Max texts per single API call |

#### Functions

---

**`embed_texts(texts: list[str], model: str | None = None) → list[list[float]]`**

Generates embeddings for a list of text strings.

- **Batching:** Automatically splits large lists into batches of 100 texts (OpenAI safe limit)
- **Retry:** Uses `tenacity` to retry up to 3 times with exponential back-off (2s → 10s) on any API error
- **Rate-limit courtesy:** Adds a 0.3s delay between batches when processing more than 100 texts
- Returns a list of 1536-dimensional float vectors, **in the same order as the input**

---

**`embed_query(query: str, model: str | None = None) → list[float]`**

Convenience wrapper — generates a single embedding for a user's query string. Returns a single `list[float]` vector (not wrapped in a list).

Used by `qa_engine.py` before performing similarity search.

---

**`_embed_batch(client, texts, model)`** *(internal, retried)*

The actual OpenAI API call for a single batch. Decorated with `@retry` from `tenacity`. Not called directly — used internally by `embed_texts`.

---

### `src/vector_store.py`
**Purpose:** All read/write operations against the Supabase `video_chunks` table with pgvector.

#### Constants

| Constant | Value | Meaning |
|---|---|---|
| `TABLE` | `"video_chunks"` | Supabase table name |
| `MATCH_FN` | `"match_video_chunks"` | SQL function called for similarity search |

#### Functions

---

**`upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) → int`**

Inserts a list of chunks (with their embeddings) into the `video_chunks` table.

- Validates that `chunks` and `embeddings` have the same length
- Builds row dicts pairing each chunk's metadata with its embedding vector
- Inserts in **batches of 200 rows** to stay within Supabase's request limits
- Returns the total number of rows successfully inserted

Called after embedding generation in `app.py`.

---

**`similarity_search(query_embedding: list[float], video_id: str, top_k: int | None = None) → list[dict]`**

Finds the `top_k` most semantically similar chunks for a given video using **cosine similarity**.

- Calls the `match_video_chunks` PostgreSQL function via `supabase.rpc()`
- Filters by `video_id` so results are always from the correct video
- Default `top_k = 5` (configurable via `TOP_K_RETRIEVAL` env var or the sidebar slider)
- Returns a list of dicts, each containing:
  `id, video_id, video_title, chunk_index, chunk_text, token_count, start_time_sec, similarity`

---

**`video_is_indexed(video_id: str) → bool`**

Checks whether a video has already been processed and stored. Used in `app.py` to skip re-ingestion when a video is submitted a second time.

Returns `True` if at least one chunk for the given `video_id` exists in the table.

---

**`delete_video(video_id: str) → int`**

Deletes all chunks associated with a given `video_id` from the `video_chunks` table. Returns the number of rows deleted.

Triggered by the **"Remove Current Video from DB"** button in the sidebar.

---

### `src/qa_engine.py`
**Purpose:** Orchestrates the complete RAG query pipeline — from user question to final answer.

#### Constants

| Constant | Value | Meaning |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Default model for answer generation |
| `IRRELEVANT_RESPONSE` | `"The question is irrelevant..."` | Standard reply when context doesn't match |
| `SYSTEM_PROMPT` | (multi-line string) | Strict instructions for the LLM |

**System Prompt rules enforced:**
1. Only answer from the provided transcript context — no outside knowledge
2. Reply with exactly `"The question is irrelevant to this video."` if context is insufficient
3. Cite the approximate timestamp when possible
4. Never fabricate information

#### Functions

---

**`_build_user_prompt(query: str, chunks: list[dict]) → str`** *(internal)*

Formats the retrieved chunks into a structured prompt string sent to the LLM.

Each chunk is presented as:
```
[Excerpt 1 | Timestamp: ~2m 30s]
<chunk text here>
```

Multiple excerpts are separated by `---`. The user's question is appended at the end.

---

**`answer_query(query: str, video_id: str, video_title: str, top_k: int | None) → dict`**

The main function of the Q&A engine. Executes the full RAG pipeline in 3 steps:

1. **Embed** — calls `embed_query()` to convert the question into a 1536-dim vector
2. **Retrieve** — calls `similarity_search()` to fetch the top-k most relevant transcript chunks from Supabase
3. **Generate** — builds a prompt with the retrieved context and calls GPT-4o-mini at `temperature=0.2`

Returns:
```python
{
    "answer":        str,         # LLM-generated answer (or irrelevant message)
    "chunks_used":   int,         # How many chunks were retrieved
    "chunks":        list[dict],  # The retrieved chunks (for display in UI)
    "is_irrelevant": bool,        # True if "irrelevant" appears in the answer
}
```

If no chunks are retrieved at all (empty search result), immediately returns the `IRRELEVANT_RESPONSE` without calling the LLM.

---

### `app.py`
**Purpose:** Streamlit web application — the user-facing interface for the entire pipeline.

#### Secrets Loading

At startup, the app:
1. Calls `load_dotenv()` — loads `.env` in local development
2. Calls `_load_streamlit_secrets()` — on Streamlit Cloud, copies `st.secrets` keys into `os.environ` so all `src/` modules that use `os.getenv()` work without modification

#### Sidebar

- **Env check** — shows ✅ if all 3 keys are set, or ⚠️ with the missing key names
- **Settings sliders:**
  - `top_k` — how many chunks to retrieve per query (1–10, default 5)
  - `chunk_size` — max tokens per chunk (200–1000, default 500)
  - `overlap` — token overlap between chunks (0–200, default 50)
- **Delete button** — calls `delete_video()` to remove the current video from Supabase

#### Step 1 — Video Processing

1. User pastes a YouTube URL → `get_video_id()` extracts the ID
2. `video_is_indexed()` checks if already stored → skips re-ingestion if so
3. `fetch_transcript()` retrieves timed caption segments
4. `chunk_transcript()` splits into sentence-aware chunks, tracking token counts
5. `embed_texts()` generates embeddings with a live progress bar
6. `upsert_chunks()` stores everything in Supabase

#### Step 2 — Q&A

1. User types a question → `answer_query()` runs the full RAG pipeline
2. Answer displayed in a styled box:
   - **Purple box** — normal grounded answer
   - **Red box** — irrelevant question detected
3. Expandable section shows retrieved transcript excerpts with similarity scores and timestamps

---

## ⚙️ Configuration Reference

All values can be set in `.env` (local) or Streamlit Cloud Secrets (deployed):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `SUPABASE_URL` | *(required)* | Your Supabase project URL |
| `SUPABASE_KEY` | *(required)* | Supabase anon or service-role key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model for answers |
| `CHUNK_SIZE_TOKENS` | `500` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Token overlap (legacy, chunker uses sentence overlap) |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |

---

## 🚀 Setup & Running

### 1. Supabase — Run Schema Once
Open **Supabase → SQL Editor** → paste `schema.sql` → Run.

### 2. Environment Variables
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
python -m streamlit run app.py
```

### 5. Deploy to Streamlit Cloud
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo, `main`, `app.py`
3. **Settings → Secrets** → paste your keys in TOML format (see `.streamlit/secrets.toml.example`)
4. Deploy

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `TranscriptsDisabled` | Video has captions disabled | Try a different video |
| `NoTranscriptFound` | No English captions exist | Try a video with auto-generated captions |
| `Missing env variables` | Keys not set | Fill in `.env` or Streamlit Cloud Secrets |
| `vector` type error on insert | Schema not run | Run `schema.sql` in Supabase SQL Editor |
| Streamlit Cloud: keys missing | Secrets not configured | Add keys in App Settings → Secrets |
