# Documentation Index

This directory contains technical documentation for the RAG system.

## Documentation Structure

```
docs/
├── README.md                    # This file
├── ARCHITECTURE.md              # System architecture and design
├── TESTING.md                   # Testing guide and philosophy
├── SETUP.md                     # Installation and setup
├── TEST_SUMMARY.md              # Test coverage summary
│
├── features/                    # Feature documentation
│   └── FEATURE_SUMMARY.md       # Re-ranker and cache features
│
└── fixes/                       # Bug fix documentation
    ├── reranker-threshold.md    # Critical: Re-ranker threshold bug
    ├── cache-invalidation.md    # High: Cache invalidation bug
    └── ui-dark-mode.md          # Medium: Dark mode visibility
```

## Quick Navigation

### For New Users
Start in the root directory:
1. **START_HERE.md** - 5-minute quick start
2. **README.md** - Project overview
3. **EXAMPLES.md** - Usage examples

### For Developers
1. **ARCHITECTURE.md** - Understand the system design
2. **SETUP.md** - Development environment setup
3. **TESTING.md** - How to write and run tests
4. **CLAUDE.md** (root) - For Claude Code instances

### For Debugging
Check **fixes/** directory for known issues:
- **reranker-threshold.md** - If re-ranking returns no results
- **cache-invalidation.md** - If cache rebuilds unexpectedly
- **ui-dark-mode.md** - If answer box is hard to see

### For Understanding Features
- **features/FEATURE_SUMMARY.md** - Re-ranker and cache system details

## File Descriptions

### Core Documentation

| File | Purpose | Audience |
|------|---------|----------|
| **ARCHITECTURE.md** | System design, data flow, components | Developers |
| **TESTING.md** | Test philosophy, how to run tests | Contributors |
| **SETUP.md** | Installation, dependencies, first run | New developers |
| **TEST_SUMMARY.md** | Test coverage statistics | QA, maintainers |

### Feature Documentation

| File | Purpose |
|------|---------|
| **features/FEATURE_SUMMARY.md** | Overview of re-ranking and caching features, usage, performance |

### Fix Documentation

| File | Bug Severity | Description |
|------|-------------|-------------|
| **fixes/reranker-threshold.md** | CRITICAL | Cross-encoder scores vs cosine similarity scale mismatch |
| **fixes/cache-invalidation.md** | HIGH | mtime changes triggering false cache invalidation |
| **fixes/ui-dark-mode.md** | MEDIUM | Answer box visibility in dark themes |

## Root Documentation Files

These are kept in the root for easy access:

- **README.md** - Main project description
- **START_HERE.md** - Quick start guide
- **EXAMPLES.md** - Detailed usage examples
- **QUICK_REFERENCE.md** - Command cheat sheet
- **CLAUDE.md** - Guide for Claude Code instances

## Documentation Principles

1. **User-facing docs in root** - Easy to find on GitHub
2. **Technical docs in docs/** - Organized by topic
3. **Bug fixes in docs/fixes/** - Historical record with solutions
4. **Features in docs/features/** - Detailed feature documentation

## Contributing to Documentation

When adding new documentation:

1. **Features**: Add to `docs/features/` with clear use cases
2. **Bug fixes**: Add to `docs/fixes/` with problem/solution format
3. **Architecture changes**: Update `docs/ARCHITECTURE.md`
4. **New tests**: Update `docs/TESTING.md` and `docs/TEST_SUMMARY.md`
5. **User-facing changes**: Update root `README.md` or `START_HERE.md`

## Version History

### Jan 2026
- Added re-ranker and cache system documentation
- Created `docs/features/` and `docs/fixes/` directories
- Documented three critical bugs and their fixes
- Reorganized documentation structure

### Jan 2025 (Initial)
- Created core documentation (ARCHITECTURE, TESTING, SETUP)
- Established user-facing docs (README, START_HERE, EXAMPLES)
- Set up comprehensive testing guide

---

**For quick help, see START_HERE.md in the root directory.**
