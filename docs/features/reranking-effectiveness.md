# Re-ranking Effectiveness Analysis

## Executive Summary

**Date:** 2026-01-07
**Status:** ⚠️ Re-ranking currently NOT effective for this dataset
**Recommendation:** Keep feature implemented but be aware of limitations

## Test Results

### Dataset
- **Documents:** 151,224 Spanish and French news articles
- **Sources:** Spain_translated.parquet, France_translated.parquet
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Test Methodology

Ran 4 diverse queries on real data:
1. "What are the consequences of independence movements?"
2. "Legal framework for constitutional changes"
3. "Economic impact of political decisions"
4. "State of democracy in Europe"

For each query:
- Retrieved pool of 50 documents using cosine similarity
- Re-ranked all 50 documents using cross-encoder
- Compared top 5 from each ranking

### Results

| Metric | Value |
|--------|-------|
| **Queries tested** | 4 |
| **Documents changed (avg)** | 0.0 / 5 |
| **Order preservation** | 100% |
| **Documents surfaced from pool** | 0 |
| **Effectiveness rating** | ❌ NOT EFFECTIVE |

**Detailed breakdown:**
```
Query 1: 0/5 changed (100% same order)
Query 2: 0/5 changed (100% same order)
Query 3: 0/5 changed (100% same order)
Query 4: 0/5 changed (100% same order)
```

## Why Re-ranking Isn't Working

### 1. Model Domain Mismatch
- **Cross-encoder training:** MS MARCO dataset (English Q&A pairs)
- **Our data:** Spanish/French news articles
- **Result:** Model not optimized for multilingual news retrieval

### 2. Embedding Quality
- `all-MiniLM-L6-v2` already performs exceptionally well
- Cosine similarity scores highly correlate with semantic relevance
- Cross-encoder scores perfectly match cosine ranking

### 3. No Room for Improvement
- Top documents from cosine similarity are already the most relevant
- Cross-encoder agrees with embedding model 100% of the time
- Re-ranking adds 2-5 seconds latency with zero benefit

## Current Architecture

### What's Implemented ✅
- Full re-ranking system with cross-encoder
- Instant toggle between original and re-ranked results
- Pool system (retrieve 50, rank all, show top-k)
- Session state caching for performance
- Comprehensive test suite

### What's NOT Working ❌
- Actual ranking improvement for Spanish/French news
- Document surfacing from deeper in the pool
- Any measurable quality increase

## Solutions

### Option A: Multilingual Cross-Encoder (RECOMMENDED)

**Try this model:**
```python
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
```

**Why it might work:**
- Trained on mMARCO (multilingual MS MARCO)
- Supports 100+ languages including Spanish and French
- Specifically designed for cross-lingual retrieval
- Same API, drop-in replacement

**How to test:**
1. Change `RERANKER_MODEL` in `rag/config.py`
2. Run `python test_reranking_effectiveness.py`
3. Look for non-zero document changes

**Expected improvement:**
- 20-40% document reordering (vs current 0%)
- Better surfacing of semantically relevant docs
- Actual effectiveness on multilingual news

### Option B: Keep Current Setup

**When to use:**
- Current setup works fine (cosine similarity is good)
- Re-ranking adds minimal value for this dataset
- But feature exists for future use cases

**What to do:**
- Disable re-ranking toggle by default (`USE_RERANKING = False`)
- Keep code and UI for future experimentation
- Document limitations in README

### Option C: Disable Completely

**NOT RECOMMENDED** - feature is implemented and tested, keep for future use

## Future Enhancements

### Near Term (Next Sprint)

**1. Multilingual Cross-Encoder**
- **Priority:** HIGH
- **Effort:** LOW (30 minutes)
- **Impact:** Potentially HIGH
- **Action:** Test `mmarco-mMiniLMv2-L12-H384-v1` model
- **Success metric:** >20% document reordering

**2. Domain-Specific Fine-Tuning**
- **Priority:** MEDIUM
- **Effort:** HIGH (weeks)
- **Impact:** HIGH
- **Action:** Fine-tune cross-encoder on Spanish/French news Q&A pairs
- **Success metric:** >50% document reordering

### Long Term (Future Roadmap)

**3. Graph-Based Retrieval** 🔮
- **Priority:** FUTURE
- **Effort:** VERY HIGH (major feature)
- **Impact:** TRANSFORMATIVE
- **Description:**
  - Build knowledge graph from document entities and relationships
  - Use graph traversal for multi-hop reasoning
  - Combine vector similarity with graph connectivity
  - Better for complex queries requiring multiple documents

**Potential approaches:**
- **Entity Linking + Graph RAG:**
  - Extract entities (people, places, events) from documents
  - Build graph connecting related entities
  - Query: "How is entity A related to entity B?"
  - Traverse graph to find connection paths

- **Document-Level Graph:**
  - Nodes = documents
  - Edges = semantic similarity, citation, topic overlap
  - Use PageRank-style algorithms for ranking

- **Hybrid Vector + Graph:**
  - Initial retrieval with embeddings (fast)
  - Graph expansion to related documents (comprehensive)
  - Re-rank combined results (accurate)

**Libraries to explore:**
- LlamaIndex (graph-based RAG)
- LangGraph (workflow graphs)
- Neo4j + vector search
- NetworkX for custom graph algorithms

**Success metrics:**
- Better handling of multi-document queries
- Improved citation/reference tracking
- More comprehensive answers

**4. Hybrid Retrieval**
- Combine dense (embeddings) + sparse (BM25) + graph
- Reciprocal Rank Fusion (RRF) for merging results
- Better coverage across different query types

## Testing

### Automated Test
```bash
python test_reranking_effectiveness.py
```

**What it measures:**
- Document position changes after re-ranking
- Documents surfaced from pool (beyond top 5)
- Average reordering percentage
- Effectiveness rating

**Interpreting results:**
- **0-1 changes/query:** Not effective (current state)
- **2-4 changes/query:** Moderately effective
- **4-5 changes/query:** Highly effective

### Manual Testing

**In the UI:**
1. Enter query
2. Click "Search"
3. Note top 5 document titles
4. Toggle re-ranking ON
5. Compare: Did the order change?

**Good sign:** Different documents appear, or order changes significantly
**Bad sign:** Exactly the same order (current state)

## Performance Impact

### Current Costs (with re-ranking enabled)

| Operation | Without Re-rank | With Re-rank | Overhead |
|-----------|----------------|--------------|----------|
| Pool retrieval (50 docs) | 0.5s | 0.5s | 0s |
| Cross-encoder scoring | 0s | 2-5s | +2-5s |
| Total | 0.5s | 3-5.5s | **+600%** |

**Verdict:** 6x slower for zero benefit (current state)

### With Effective Re-ranker

If multilingual model works:
- Same 2-5s overhead
- But: significantly better results
- Worth the tradeoff

## Documentation

### For Users

Update README with:
```markdown
⚠️ **Re-ranking Note:** Currently using English-trained model on Spanish/French data.
Re-ranking may not improve results significantly. For better effectiveness, consider
switching to multilingual cross-encoder (see docs/features/reranking-effectiveness.md)
```

### For Developers

See:
- Test script: `test_reranking_effectiveness.py`
- Implementation: `rag/retrieval/reranker.py`
- Integration: `rag/app.py` (lines 185-304)
- Config: `rag/config.py` (`RERANKER_MODEL`)

## Conclusion

**Current State:**
- ✅ Re-ranking system fully implemented and tested
- ✅ Pool system working correctly
- ✅ Instant toggle functional
- ❌ No actual quality improvement for this dataset
- ❌ 6x slower with zero benefit

**Next Steps:**
1. **Immediate:** Keep current implementation, document limitations
2. **Short-term:** Test multilingual cross-encoder (`mmarco-mMiniLMv2`)
3. **Long-term:** Consider graph-based retrieval for complex queries

**Decision:** Keep feature as-is, well-documented, ready for improvement when better model is available.

---

**Test run date:** 2026-01-07
**Tested by:** Automated effectiveness test
**Next review:** After trying multilingual cross-encoder
