-- ─────────────────────────────────────────────────────────────
--  YouTube RAG Pipeline — Supabase Schema
--  Run this ONCE in your Supabase SQL Editor.
-- ─────────────────────────────────────────────────────────────

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create the main chunks table
create table if not exists video_chunks (
    id              bigserial primary key,
    video_id        text        not null,
    video_title     text,
    video_url       text,
    chunk_index     int         not null,
    chunk_text      text        not null,
    token_count     int         not null,
    start_time_sec  float,
    embedding       vector(1536),           -- text-embedding-3-small dimension
    created_at      timestamptz default now()
);

-- 3. Index for fast lookup by video_id
create index if not exists idx_video_chunks_video_id
    on video_chunks (video_id);

-- 4. IVFFlat index for approximate nearest-neighbour search
--    (Requires at least a few hundred rows to be effective.
--     For small datasets, exact search is used automatically.)
create index if not exists idx_video_chunks_embedding
    on video_chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- 5. Helper function: match chunks by cosine similarity
--    Called from Python via supabase.rpc(...)
create or replace function match_video_chunks(
    query_embedding  vector(1536),
    match_video_id   text,
    match_count      int default 5
)
returns table (
    id              bigint,
    video_id        text,
    video_title     text,
    chunk_index     int,
    chunk_text      text,
    token_count     int,
    start_time_sec  float,
    similarity      float
)
language sql stable
as $$
    select
        vc.id,
        vc.video_id,
        vc.video_title,
        vc.chunk_index,
        vc.chunk_text,
        vc.token_count,
        vc.start_time_sec,
        1 - (vc.embedding <=> query_embedding) as similarity
    from video_chunks vc
    where vc.video_id = match_video_id
    order by vc.embedding <=> query_embedding
    limit match_count;
$$;
