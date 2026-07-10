# Plan: Fix PDF Upload Failure

## Problem
PDF uploads fail silently. The user sees "processing" forever with no error message.

## Root Causes
1. **Silent background task failures** (`upload.py:45`) — `executor.submit()` discards the Future, so exceptions vanish
2. **No error handling in background task** (`upload.py:18-28`) — only `finally: db.close()`, no `except` block
3. **Extractor silently returns empty** (`extractor.py:87-89`) — on non-rate-limit errors, returns `{entities: [], relationships: []}` with no status update
4. **No logging anywhere** — errors print to stdout in background threads, invisible to user

## Fix Plan

### 1. `app/api/upload.py` — Add error handling + logging
- Add `except` block in `process_pdf_background` to catch all exceptions
- Log errors with traceback
- Update document status to `"error"` on failure
- Store Future from `executor.submit()` and add error callback

### 2. `app/ingestion/extractor.py` — Better error visibility
- Log extraction errors with more context
- Don't silently swallow JSON decode errors — retry once

### 3. `app/ingestion/pipeline.py` — Let pipeline errors propagate
- Currently pipeline catches exceptions and sets status to "error", then re-raises
- This is correct — just need the background task to catch the re-raised exception

### Files to modify
- `app/api/upload.py` — main fix (error handling + logging)
- `app/ingestion/extractor.py` — improve error logging

### Verification
- Upload a PDF and check terminal output for error messages
- Check document status in GET /api/documents — should show "error" with details if it fails
