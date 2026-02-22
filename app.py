"""
app.py — Streamlit UI for the YouTube RAG Pipeline
────────────────────────────────────────────────────
Step 1: User pastes a YouTube URL → transcript is fetched, chunked,
        embedded, and stored in Supabase (skipped if already indexed).
Step 2: User types a question → similarity search + LLM answer is shown.
"""

import os
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.transcript import get_video_id, fetch_transcript, get_video_metadata
from src.chunker import chunk_transcript, chunks_to_dicts, count_tokens
from src.embedder import embed_texts
from src.vector_store import upsert_chunks, similarity_search, video_is_indexed, delete_video
from src.qa_engine import answer_query

logging.basicConfig(level=logging.INFO)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Pipeline",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e8e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .rag-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 16px;
        padding: 24px;
        margin-top: 16px;
        line-height: 1.75;
    }

    /* Irrelevant warning */
    .irrelevant-box {
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 16px;
        padding: 20px;
        margin-top: 16px;
    }

    /* Stat chips */
    .stat-chip {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 50px;
        padding: 4px 14px;
        font-size: 13px;
        margin: 4px 4px 4px 0;
        color: #c4b5fd;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #e8e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
    }

    h1, h2, h3 { color: #e8e8f0; }
    .stExpander { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helpers ────────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def check_env() -> list[str]:
    missing = []
    for var in ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]:
        if not os.getenv(var):
            missing.append(var)
    return missing


# ─── Session State ──────────────────────────────────────────────────────────
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "video_url" not in st.session_state:
    st.session_state.video_url = ""


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 YouTube RAG")
    st.markdown("---")
    st.markdown(
        """
        **How it works:**
        1. Paste a YouTube URL  
        2. Click **Process Video**  
        3. Ask any question about the video  
        """
    )
    st.markdown("---")

    missing_vars = check_env()
    if missing_vars:
        st.error(f"⚠️ Missing environment variables:\n`{'`, `'.join(missing_vars)}`")
        st.markdown("Create a `.env` file based on `.env.example`.")
    else:
        st.success("✅ Environment configured")

    st.markdown("---")
    st.markdown("**Settings**")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=5)
    chunk_size = st.slider("Chunk size (tokens)", min_value=200, max_value=1000, value=500, step=50)
    overlap = st.slider("Overlap (tokens)", min_value=0, max_value=200, value=50, step=10)

    st.markdown("---")
    if st.button("🗑️ Remove Current Video from DB", use_container_width=True):
        if st.session_state.video_id:
            with st.spinner("Deleting…"):
                deleted = delete_video(st.session_state.video_id)
            st.success(f"Removed {deleted} chunks.")
            st.session_state.video_id = None
            st.session_state.video_title = ""
            st.session_state.chunk_count = 0
            st.session_state.total_tokens = 0
        else:
            st.warning("No video loaded yet.")


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="text-align:center; background: linear-gradient(135deg,#6366f1,#a855f7,#ec4899);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:2.8rem; font-weight:700; margin-bottom:4px;">
    YouTube RAG Pipeline
    </h1>
    <p style="text-align:center; color:#9ca3af; margin-top:0;">
    Ask any question about a YouTube video — powered by Supabase pgvector + OpenAI
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ─── Step 1: Video Processing ────────────────────────────────────────────────
st.markdown('<div class="rag-card">', unsafe_allow_html=True)
st.markdown("### 📹 Step 1 — Load YouTube Video")

col1, col2 = st.columns([4, 1])
with col1:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
        key="youtube_url_input",
    )
with col2:
    process_btn = st.button("⚡ Process", use_container_width=True)

# Optional: manual title input
video_title_input = st.text_input(
    "Video title (optional — used for display & metadata)",
    placeholder="e.g. How Transformers Work – Explained Simply",
)

if process_btn:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL.")
    elif check_env():
        st.error("Configure your `.env` file before processing.")
    else:
        try:
            with st.spinner("Extracting video ID…"):
                vid_id = get_video_id(youtube_url.strip())
                meta = get_video_metadata(vid_id)
                title = video_title_input.strip() or f"Video {vid_id}"

            already_indexed = video_is_indexed(vid_id)

            if already_indexed:
                st.info(f"✅ Video **{title}** is already indexed in Supabase. Ready to answer questions!")
                st.session_state.video_id = vid_id
                st.session_state.video_title = title
                st.session_state.video_url = meta["video_url"]
            else:
                # Fetch transcript
                with st.spinner("Fetching transcript…"):
                    segments = fetch_transcript(vid_id)

                st.success(f"✅ Transcript fetched — {len(segments)} segments.")

                # Chunk
                with st.spinner("Chunking transcript with token tracking…"):
                    chunks = chunk_transcript(
                        segments,
                        video_id=vid_id,
                        video_title=title,
                        video_url=meta["video_url"],
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    chunk_dicts = chunks_to_dicts(chunks)
                    total_tokens = sum(c["token_count"] for c in chunk_dicts)

                st.success(f"✅ {len(chunks)} chunks | {total_tokens:,} total tokens")

                # Embed
                texts = [c["chunk_text"] for c in chunk_dicts]
                prog = st.progress(0, text="Generating embeddings…")
                all_embeddings = []
                BATCH = 20
                for i in range(0, len(texts), BATCH):
                    batch_texts = texts[i: i + BATCH]
                    batch_embs = embed_texts(batch_texts)
                    all_embeddings.extend(batch_embs)
                    prog.progress(
                        min((i + BATCH) / len(texts), 1.0),
                        text=f"Embedding chunk {min(i + BATCH, len(texts))}/{len(texts)}…",
                    )
                prog.empty()

                # Store in Supabase
                with st.spinner("Storing in Supabase (pgvector)…"):
                    inserted = upsert_chunks(chunk_dicts, all_embeddings)

                st.success(f"✅ {inserted} chunks stored in Supabase!")

                # Persist to session
                st.session_state.video_id = vid_id
                st.session_state.video_title = title
                st.session_state.video_url = meta["video_url"]
                st.session_state.chunk_count = len(chunks)
                st.session_state.total_tokens = total_tokens

        except Exception as exc:
            st.error(f"❌ {exc}")
            st.markdown(
                "💡 **Tip**: If the transcript is unavailable, try a different video or "
                "check that captions are enabled on the video."
            )

# Show current video badge
if st.session_state.video_id:
    st.markdown(
        f"""
        <div style="margin-top:12px">
            <span class="stat-chip">📺 {st.session_state.video_title}</span>
            <span class="stat-chip">🆔 {st.session_state.video_id}</span>
            {"<span class='stat-chip'>📦 " + str(st.session_state.chunk_count) + " chunks</span>" if st.session_state.chunk_count else ""}
            {"<span class='stat-chip'>🪙 " + f"{st.session_state.total_tokens:,}" + " tokens</span>" if st.session_state.total_tokens else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ─── Step 2: Q&A ─────────────────────────────────────────────────────────────
st.markdown('<div class="rag-card">', unsafe_allow_html=True)
st.markdown("### 💬 Step 2 — Ask a Question")

query = st.text_area(
    "Your question",
    placeholder="What does the video say about…?",
    label_visibility="collapsed",
    height=100,
    key="query_input",
)

ask_btn = st.button("🔍 Ask", use_container_width=False)

if ask_btn:
    if not st.session_state.video_id:
        st.warning("Process a video first (Step 1).")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching & generating answer…"):
            result = answer_query(
                query=query.strip(),
                video_id=st.session_state.video_id,
                video_title=st.session_state.video_title,
                top_k=top_k,
            )

        answer = result["answer"]
        is_irrel = result["is_irrelevant"]
        chunks_used = result["chunks_used"]
        retrieved = result["chunks"]

        # Display answer
        if is_irrel:
            st.markdown(
                f'<div class="irrelevant-box">⚠️ <strong>{answer}</strong></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )

        # Stats
        st.markdown(
            f"""
            <div style="margin-top:12px">
                <span class="stat-chip">📎 {chunks_used} chunks used</span>
                <span class="stat-chip">🎯 top-{top_k} retrieval</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Retrieved chunks expander
        if retrieved:
            with st.expander("🔎 View retrieved transcript excerpts"):
                for i, chunk in enumerate(retrieved, start=1):
                    sim = chunk.get("similarity", 0)
                    ts = chunk.get("start_time_sec", 0)
                    st.markdown(
                        f"**Excerpt {i}** — "
                        f"Similarity: `{sim:.3f}` | "
                        f"Timestamp: `{format_timestamp(ts)}` | "
                        f"Tokens: `{chunk.get('token_count', '?')}`"
                    )
                    st.markdown(f"> {chunk['chunk_text'][:500]}{'…' if len(chunk['chunk_text']) > 500 else ''}")
                    st.divider()

st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; margin-top:40px; color:#6b7280; font-size:13px;">
    YouTube RAG Pipeline · Supabase pgvector · OpenAI · Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
