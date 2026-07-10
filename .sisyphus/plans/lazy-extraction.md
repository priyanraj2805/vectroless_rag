# Plan: Lazy Entity Extraction

## Goal
Skip entity extraction during PDF upload. Extract on-demand when user asks a question. Uploads become instant (local operations only).

## Current Flow
```
Upload → Parse PDF → Chunk → Embed → Extract entities (LLM) → Build graph → Ready
Query  → Plan → Execute graph query → Synthesize (LLM)
```

## New Flow
```
Upload → Parse PDF → Chunk → Embed → Ready (no LLM calls)
Query  → Plan → Find relevant chunks → Extract entities from chunks (LLM) → Synthesize (LLM)
```

## Changes

### 1. `app/ingestion/pipeline.py`
- Remove Step 4 (entity extraction) and Step 5 (graph building)
- Document status goes straight to "ready" after embedding
- Keep the rest (parse, chunk, embed, DB insert)

### 2. `app/api/chat.py`
- After `executor.execute(plan)` finds relevant chunks, extract entities from those chunks
- Build per-document groups for the synthesizer
- This is where the LLM extraction now happens

### 3. `app/storage/database.py`
- May need a helper to check if chunks have been extracted
- Or just always extract at query time (simpler)

### 4. `app/query/executor.py`
- No changes needed — it just queries chunks and nodes from DB

## Tradeoffs
- Upload: instant (seconds instead of minutes)
- First query per document: slightly slower (extraction happens on-demand)
- Subsequent queries: same speed (cached extraction results)
- No entity graph built upfront — but query still works via chunk text search

## Files to modify
- `app/ingestion/pipeline.py` — remove extraction steps
- `app/api/chat.py` — add extraction at query time
