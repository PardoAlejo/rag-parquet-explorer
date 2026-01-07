# Retrieval Pool System

## Overview

The retrieval pool system retrieves a **large pool of documents** initially (default 50, up to 100), then ranks them all and shows/uses only the **top-k best** for LLM generation.

This makes re-ranking **significantly more effective** by giving it more candidates to choose from.

## The Problem (Before)

**Old architecture:**
```
top_k = 5
  ↓
Retrieve 5 docs (cosine similarity)
  ↓
Re-rank those 5 docs
  ↓
Still have 5 docs (same ones!)
  ↓
Feed to LLM
```

**Issue:** Re-ranking had no room to improve! It could only re-order the 5 documents already retrieved, but couldn't surface better documents from outside this small pool.

**Re-ranking was almost pointless!**

## The Solution (After)

**New architecture:**
```
pool_size = 50 (configurable 10-100)
top_k = 5 (configurable 1-20)
  ↓
Retrieve 50 docs (cosine similarity - fast)
  ↓
Re-rank ALL 50 docs (cross-encoder - accurate)
  ↓
Pick top 5 best ranked
  ↓
Show those 5 + Feed to LLM
```

**Benefits:**
- Re-ranking picks best 5 from 50 candidates ✅
- Much more likely to surface the truly best documents ✅
- Cosine similarity casts wide net, re-ranking refines ✅

## Configuration

### Config File (rag/config.py)

```python
RETRIEVAL_POOL_SIZE = 50  # Initial retrieval pool
TOP_K_RESULTS = 5         # How many to show/use for LLM
```

### UI Controls

**Advanced Settings (Sidebar):**

1. **Retrieval Pool Size** (10-100, step 10, default 50)
   - How many documents to retrieve initially
   - Larger = better re-ranking effectiveness
   - Slower for very large pools (100+ docs)

2. **Top K for LLM** (1-20, default 5)
   - How many top-ranked documents to show
   - These same documents used for LLM generation
   - Can adjust instantly without re-searching

## How It Works

### Search Flow

**1. Initial Retrieval**
```
User enters query → Retrieve pool_size documents
Status: "Retrieving pool of 50 documents..."
Method: Fast cosine similarity search
Result: 50 candidates cached
```

**2. Ranking (Both Versions)**
```
Original: Keep cosine similarity order
Re-ranked: Re-rank all 50 with cross-encoder
Status: "Re-ranking 50 documents..."
Result: Both pools cached
```

**3. Selection**
```
Slice pools to top_k
Show top 5 documents
Use top 5 for LLM context
```

**4. Display**
```
Header: "Top 5 Documents 🔄"
Caption: "Retrieved 50 docs → Re-ranked → Showing top 5 for LLM"
```

### Caching Strategy

**Cache Key:**
```python
cache_key = f"{query}|{pool_size}|{similarity_threshold}"
```

**Cached Data:**
```python
st.session_state.results_cache = {
    'original_pool': [50 docs],  # Cosine-ranked
    'reranked_pool': [50 docs]   # Cross-encoder ranked
}
```

**Cache Invalidation:**
- Query changes → Clear cache
- Pool size changes → Clear cache
- Threshold changes → Clear cache
- Top-k changes → Keep cache (just slice differently) ✅
- Toggle changes → Keep cache (just switch pools) ✅

## User Experience

### Example Workflow

**Step 1: Initial Search**
```
Query: "What are the consequences of independence movements?"
Pool size: 50
Top K: 5
Re-ranking: OFF

Action: Click "Search"
Result:
  - Retrieved 50 documents from pool
  - Showing top 5 for LLM

Documents shown (cosine similarity order):
  1. Doc A (Score: 0.58)
  2. Doc B (Score: 0.52)
  3. Doc C (Score: 0.48)
  4. Doc D (Score: 0.45)
  5. Doc E (Score: 0.42)
```

**Step 2: Enable Re-ranking**
```
Action: Toggle re-ranking ON
Result: ⚡ Switched to re-ranked results instantly!

Documents shown (re-ranked order):
  1. Doc C (Score: 1.00)  ← Jumped from #3 to #1!
  2. Doc F (Score: 0.85)  ← Surfaced from pool!
  3. Doc A (Score: 0.60)  ← Dropped to #3
  4. Doc H (Score: 0.45)  ← New doc from pool!
  5. Doc B (Score: 0.20)  ← Dropped to #5
```

**Key Insights:**
- Doc C was #3 in cosine, but #1 after re-ranking
- Doc F wasn't in top 5 cosine, but re-ranking found it
- Re-ranking completely changed the order
- Better documents surfaced from the 50-doc pool

**Step 3: Adjust Top K**
```
Action: Slide "Top K for LLM" to 10
Result: Instantly shows top 10 (no re-search needed)

Now showing:
  1-5: Same as before
  6-10: Next best from pool
```

**Step 4: Increase Pool**
```
Action: Slide "Retrieval Pool Size" to 100
Result: Re-searches with larger pool

New flow:
  - Retrieved 100 documents
  - Re-ranked all 100
  - Showing top 10
  - Even better results!
```

## Performance Characteristics

### Retrieval Times

| Pool Size | Retrieval | Re-ranking | Total |
|-----------|-----------|------------|-------|
| 10 docs   | ~0.3s     | ~0.5s      | ~0.8s |
| 50 docs   | ~0.5s     | ~2s        | ~2.5s |
| 100 docs  | ~1s       | ~5s        | ~6s   |

### Operation Times

| Operation | Time | Notes |
|-----------|------|-------|
| Initial search | 2-6s | Depends on pool size |
| Toggle ON/OFF | <100ms | Uses cached pools |
| Change top_k | <100ms | Just slices cache |
| Change pool size | 2-6s | Re-searches |

## Effectiveness Comparison

### Scenario: Looking for Legal Consequences

**Query:** "Legal consequences of independence movements"

**Without Pool System (Old):**
```
Retrieve top 5 docs by cosine:
  1. Doc about economic impact (0.58)
  2. Doc about historical context (0.52)
  3. Doc about legal framework (0.48) ← BEST DOC
  4. Doc about social movements (0.45)
  5. Doc about cultural identity (0.42)

Re-rank those 5:
  1. Doc about legal framework (re-scored)
  2. Doc about economic impact
  3. Doc about historical context
  4. Doc about social movements
  5. Doc about cultural identity

Result: Found the legal doc, but it was already in top 5
```

**With Pool System (New):**
```
Retrieve top 50 docs by cosine:
  1-5: (same as above)
  ...
  23. Doc with legal consequences detail (0.35) ← BETTER DOC!
  ...
  50. Doc about geography (0.12)

Re-rank all 50:
  1. Doc with legal consequences detail (1.00) ← SURFACED!
  2. Doc about legal framework (0.85)
  3. Doc about court rulings (0.75) ← ALSO FOUND!
  4. Doc about constitutional law (0.60) ← NEW!
  5. Doc about economic impact (0.45)

Result: Found multiple highly relevant legal docs that
        weren't in original top 5!
```

**Improvement:** Re-ranking found better documents from deeper in the pool.

## Best Practices

### Recommended Settings

**For General Use:**
- Pool size: 50
- Top K: 5
- Re-ranking: ON

**For Quick Exploration:**
- Pool size: 20
- Top K: 3
- Re-ranking: OFF (faster)

**For Thorough Search:**
- Pool size: 100
- Top K: 10
- Re-ranking: ON (best quality)

### When to Adjust Pool Size

**Increase pool size (50 → 100) when:**
- Re-ranking helps but results still not great
- Large document collection (>10K docs)
- Looking for rare/specific information
- Quality matters more than speed

**Decrease pool size (50 → 20) when:**
- Fast iteration needed
- Small document collection (<1K docs)
- Speed matters more than quality
- Initial results already good

### When to Adjust Top K

**Increase top_k (5 → 10) when:**
- LLM needs more context
- Want to see more options
- Comparing multiple perspectives
- Complex queries need diverse sources

**Decrease top_k (5 → 3) when:**
- Want focused, concise answers
- LLM context window limited
- High-quality results already in top 3
- Faster LLM generation desired

## Technical Implementation

### Pool Retrieval
```python
# Retrieve full pool
pool_original = retriever.search(
    query,
    top_k=pool_size,  # e.g., 50
    use_reranking=False,
    min_similarity=threshold
)

# Cache full pool
st.session_state.results_cache['original_pool'] = pool_original
```

### Re-ranking
```python
# Re-rank full pool
pool_reranked = retriever.search(
    query,
    top_k=pool_size,  # e.g., 50
    use_reranking=True,
    min_similarity=threshold
)

# Cache re-ranked pool
st.session_state.results_cache['reranked_pool'] = pool_reranked
```

### Display Slicing
```python
# Get appropriate pool
if use_reranking:
    pool = st.session_state.results_cache['reranked_pool']
else:
    pool = st.session_state.results_cache['original_pool']

# Slice to top_k for display
results = pool[:top_k]  # e.g., top 5 from 50
```

### LLM Context
```python
# Build context from displayed results only
context_parts = []
for i, result in enumerate(results, 1):  # Only top_k
    context_parts.append(f"[Document {i}]")
    context_parts.append(f"Content: {result['text']}")

context = "\n".join(context_parts)

# Generate answer
answer = llm.generate_with_context(query, context)
```

## Troubleshooting

### "Retrieved 50 docs → Showing top 5 for LLM"
**Meaning:** Working correctly!
- Pool of 50 retrieved
- All 50 ranked
- Showing best 5

### "Retrieved 10 docs → Showing top 10 for LLM"
**Meaning:** Pool smaller than top_k or threshold filtered many
- Only 10 docs passed similarity threshold
- Showing all 10

### Re-ranking doesn't improve results
**Possible causes:**
1. Pool too small (try increasing to 100)
2. No better docs exist in collection
3. Cross-encoder model not suitable for domain

**Solutions:**
- Increase pool size
- Lower similarity threshold
- Check if better docs exist manually

### Slow performance
**Causes:**
- Large pool size (100 docs)
- Re-ranking enabled

**Solutions:**
- Reduce pool size to 30-50
- Disable re-ranking for exploration
- Use re-ranking only for final answers

## Summary

The retrieval pool system makes re-ranking **actually useful** by:

✅ Retrieving large candidate pool (50-100 docs)
✅ Re-ranking ALL candidates (not just top 5)
✅ Surfacing best documents from deep in pool
✅ Showing only top-k for clarity
✅ Using only top-k for LLM (consistency)
✅ Instant top-k adjustment (cached pools)
✅ Effective re-ranking (10x more candidates)

**Result:** Re-ranking can now find documents that cosine similarity ranked #23 and move them to #1!
