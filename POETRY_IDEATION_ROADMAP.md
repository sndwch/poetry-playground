# Poetry Ideation Features - Implementation Roadmap

## Overview
Transform the generative poetry library from a generator of complete experimental poems into a comprehensive poetry ideation assistant that helps poets explore themes, discover unexpected connections, and break creative blocks.

## 📊 Implementation Status (Last Updated: 2025-11-05)

### ✅ Completed Features (5)
1. **Line and Phrase Seed Generator** - Generates evocative incomplete phrases and line beginnings
2. **Dynamic Metaphor Generator** - Creates fresh metaphors by mining Project Gutenberg texts
3. **Personal Corpus Analyzer** - Analyzes poet's existing work to understand their voice
4. **Poetry Idea Generator** - Mines classic literature for creative seeds across 10 categories
5. **Resonant Fragment Miner** - Extracts poetic sentence fragments using 26 specialized patterns

### 🚧 In Progress (0)
None currently

### 📋 Planned Features (3)
1. **Theme and Concept Explorer** - Brainstorming assistant for thematic clusters
2. **Prompt and Constraint Generator** - Oulipian writing constraints and creative prompts
3. **Mood and Register Mapping** - Emotional intelligence for word selection

### 💡 Additional Features Implemented (5)
1. **Six Degrees Word Convergence** - Explores unexpected connections between concepts
2. **Ship of Theseus Transformer** - Gradually transforms existing poems
3. **Visual Poetry Generators** - Multiple concrete/visual poetry generation tools
4. **Document Library System** - Centralized Gutenberg text retrieval with anti-repetition
5. **Equidistant Word Finder** - Discovers words equidistant (by Levenshtein distance) from two anchor words, supporting both orthographic and phonetic modes with craft-aware scoring

---

## 🎯 Feedback-Driven Improvements (ChatGPT + Gemini Analysis)

Based on comprehensive external code review, the following improvements have been prioritized by impact and feasibility.

### **Tier 1: Quick Wins** ⚡ - COMPLETE (5/5) ✅

1. **✅ Persistent API Caching** - Implemented diskcache for CMU/Datamuse with exponential backoff. 158x speedup for Datamuse, 30x for CMU. (commit: 882b5de)

2. **✅ Deterministic/Reproducible Outputs** - Added `--seed` flag with thread-through to Python random and numpy. Perfect determinism achieved. (commit: f212708)

3. **✅ Global CLI Flags** - Professional CLI with `--out`, `--format`, `--quiet`, `--verbose`, `--no-color`, `--dry-run`, `--list-fonts`, `--list-procedures`. (commit: 158ae6f)

4. **✅ Automated Model Downloads** - Zero-friction setup with lazy-loading and `--setup` command for NLTK/spaCy models. (commit: 4a38576)

5. **✅ Graceful Error Handling** - Platform-specific error messages with actionable instructions for system dependencies. (commit: 299107c)

### **Tier 2: Foundation** 🏗️ - COMPLETE (6/6) ✅

6. **✅ Modern Configuration System** - Pydantic v2 with multi-source loading (CLI > YAML > pyproject.toml > env > defaults), flexible spaCy model selection. (commit: 1788eb6)

7. **✅ Type Safety & Linting** - Modern ruff + mypy with pre-commit hooks. 82% violation reduction (164 → 29). (commit: ce435fb)

8. **✅ CI/CD Pipeline** - GitHub Actions with linting, type checking, matrix testing (Python 3.9-3.12, Ubuntu/macOS), example generation, Docker builds. (commit: 84c09f4)

9. **✅ Comprehensive Test Suite** - 143 tests covering rhyme, syllables, cache, determinism, forms, metaphors, fragments. All passing with network mocking. (commits: aa57bef, e8232e4)

10. **✅ DRY Refactoring** - Eliminated duplicate code, centralized document retrieval, consolidated shared functionality. (commit: c3326b4)

11. **✅ Enhanced Documentation** - Added example commands, pinned dependency lock file, comprehensive README sections. (commit: c3326b4)

### **Tier 3: Feature Expansion** 🚀 - IN PROGRESS (3/7)

12. **✅ Syllable-Aware Forms** - Haiku/tanka/senryu generation with CMUdict syllable counts, form validation, comprehensive test suite (35 tests). Creates syllable-constrained output. (Item #12 complete, but leads to #12a)

**12a. Grammatical Templates for Forms** - HIGH PRIORITY (Phases 1-4 COMPLETE ✅)

**Problem**: Syllable-aware forms produce grammatically incoherent "word soup" (e.g., "tomorrow did had going").

**Solution**: POS-based grammatical templates that enforce syntactic structure while maintaining syllable constraints.

**Progress**:
- ✅ **Phase 1**: POS-tagged word bank (~10,000+ words organized by POS/syllables, disk-cached). File: `pos_vocabulary.py` (350+ lines, 24 tests)
- ✅ **Phase 2**: Grammatical template system with 40+ POS patterns for haiku/tanka lines. File: `grammatical_forms.py` (400+ lines, comprehensive tests)
- ✅ **Phase 3**: Semantic coherence scoring using spaCy similarity, vocabulary diversity metrics, imagery richness. Filters low-quality outputs.
- ✅ **Phase 4.1-4.2**: Template expansion via analysis of 600 classic haiku (Basho, Issa, Buson), identified 25 common structural patterns
- ✅ **Phase 4.4**: Metaphor template integration - 50+ nature/imagery templates using curated vocabulary (600+ evocative verbs, 400+ atmospheric nouns). File: `metaphor_templates.py` (commit: 7a7afcc)
- ⏭️ **Phase 4.3**: LLM-assisted template expansion (SKIPPED - manual curation proved sufficient)

**Remaining Work**:
- Phase 5: Multi-template selection (variety across batches)
- Phase 6: Interactive template refinement
- Phase 7: User-defined custom templates

**Current Status**: Generates grammatically correct, semantically coherent haiku/tanka with rich imagery. Quality dramatically improved from "word soup" to readable poetry.

13. **❌ Corpus Export Formats** (NOT STARTED) - JSON/CSV export for poetry analysis, batch generation workflows. Benefit: Integration with other tools.

14. **❌ Better Logging & Debugging** (NOT STARTED) - `--log-level DEBUG/INFO/WARN` flag, structured logging, detailed generation traces. Benefit: Troubleshooting and development.

15. **❌ Plugin System** (NOT STARTED) - Modular architecture for custom generators, extensible framework. Benefit: Community contributions, experimentation.

16. **❌ Smart Layout Engine** (NOT STARTED) - Visual poem layout algorithms with whitespace optimization, non-overlapping placement. Benefit: More readable visual poems.

17. **❌ Corpus Analyzer Dashboard** (NOT STARTED) - Per-book signatures with bigrams, sentiment, POS rhythms. HTML export for academic use. Benefit: Shows value, attracts users.

### **Tier 4: Advanced Capabilities** 🌟 (Long-term)

18. **Evolutionary Ship of Theseus++** - Treat poem as genome with fitness functions (meter, rhyme, semantic distance). Automated multi-step transformation with quality control.

19. **Six Degrees → Story Scaffolds** - Generate 3-beat or 5-beat narrative arcs (L1 image, L2 turn, L3 implication). Bridges ideation to structure.

20. **Theme and Concept Explorer** - ConceptNet API integration, Wikipedia cultural/historical references, NRC emotion associations. Completes original roadmap vision.

21. **Web Interface/API** - Flask/FastAPI with `/generate`, `/analyze` endpoints. Simple web UI for all tools. 10x accessibility improvement.

22. **Transformer Model Integration** - GPT-2/DistilGPT-2 for coherent stanza generation with `--model` flag. State-of-the-art generation quality.

23. **Interactive & Multimodal Features** - Visual editor (terminal/web), text-on-image backgrounds, audio generation (`--speak`), custom corpus uploads. Creative empowerment.

---

## Feature 1: Theme and Concept Explorer ⏳ PLANNED

### Purpose
Create a brainstorming assistant that generates conceptual clusters and word constellations from seed words, presenting unexpected thematic directions.

### Core Functionality
- **Theme expansion**: Start from seed words, expand outward through semantic relatives, historical/cultural associations, contrasting concepts, sensory associations
- **Constellation visualization**: ASCII art or networkx graphs showing relationship strengths and unexpected bridges
- **Integration points**: ConceptNet API, Wikipedia API, NRC Word-Emotion Association

### Implementation Approach
```python
class ThemeExplorer:
    def explore_theme(seed_words, depth=2, branches=5):
        # Expand theme through multiple dimensions
        # Return ThemeCluster with scored relationships

    def generate_constellation(theme_cluster):
        # Visualize as graph with connection strengths
```

---

## Feature 2: Dynamic Metaphor Generator ✅ COMPLETED

**Status**: Fully implemented with adaptive scaling system.

### Implementation
- Mines Project Gutenberg texts for metaphor patterns
- Adaptive document scaling (maintains quality by increasing variety)
- 50+ metaphor templates with nature/imagery focus
- Curated vocabulary: 600+ evocative verbs, 400+ atmospheric nouns
- Quality scoring with spaCy similarity
- File: `generativepoetry/metaphor_generator.py`

---

## Feature 3: Poetry Idea Generator ✅ COMPLETED

**Status**: Fully implemented with 10-category mining system.

### Implementation
- Extracts creative seeds from classic literature
- 10 categories: emotional moments, character situations, vivid imagery, setting descriptions, philosophical fragments, dialogue sparks, sensory details, conflict scenarios, opening lines, metaphysical concepts
- Anti-repetition document tracking
- Adaptive scaling for quality maintenance
- File: `generativepoetry/idea_generator.py`

---

## Feature 4: Personal Corpus Analyzer ✅ COMPLETED

**Status**: Fully implemented.

### Implementation
- Upload personal poetry collections
- Analysis: themes, word frequency, emotional patterns, stylistic trends
- Interactive CLI integration
- File: `generativepoetry/corpus_analyzer.py`

---

## Feature 5: Resonant Fragment Miner ✅ COMPLETED

**Status**: Fully implemented with 26 specialized patterns.

### Implementation
- Extracts poetic sentence fragments from classic texts
- 26 patterns across 5 categories: causality, temporal, universal, singular, modal
- Quality scoring (threshold: 0.65+)
- Adaptive document retrieval
- Request 10-200 fragments
- File: `generativepoetry/causal_poetry.py`

---

## Feature 6: Line and Phrase Seed Generator ✅ COMPLETED

**Status**: Fully implemented.

### Implementation
- Generates opening lines, pivotal fragments, closing thoughts
- Tailored to input words using phonetic and semantic relationships
- Interactive CLI integration
- File: `generativepoetry/line_seeds.py`

---

## Feature 7: Six Degrees Word Convergence ✅ COMPLETED

**Status**: Fully implemented.

### Implementation
- Explores unexpected pathways between any two concepts
- Semantic relationship traversal
- Discovers surprising connections for poetic inspiration
- File: `generativepoetry/six_degrees.py`

---

## Feature 8: Ship of Theseus Transformer ✅ COMPLETED

**Status**: Fully implemented.

### Implementation
- Gradually transforms existing poems
- Systematic word replacement while maintaining structure
- Explores meaning shifts with word choice variations
- File: `generativepoetry/ship_of_theseus.py`

---

## Core Principles

1. **"Never lower standards, always scale documents"** - Maintain quality by increasing variety, not compromising filters
2. **Constraint-based creativity** - Use systematic limitations to spark unexpected discoveries
3. **Iterative exploration** - Tools designed for repeated use as you develop your poetic practice
4. **Computational augmentation** - Technology as creative partner, not replacement

---

## Technical Architecture Notes

### Adaptive Scaling System
All text processing modules follow the principle: retrieve additional diverse documents when yield is low, never compromise quality thresholds.

- Anti-repetition tracking ensures document diversity
- Quality thresholds: 0.65+ for fragments, strict validation for ideas
- Intelligent batch sizing based on remaining needs
- Centralized document library: `generativepoetry/document_library.py`

### Word Quality Filtering
Enhanced validation filters out proper nouns, non-English words, very rare/archaic words (<1 in 1M frequency), abbreviations, acronyms, technical jargon.

Customize in: `generativepoetry/word_validator.py`, `generativepoetry/vocabulary.py`

### Configuration System
Multi-source priority: CLI flags > YAML config > pyproject.toml > environment variables > defaults

Type-safe with Pydantic v2, flexible spaCy model selection (sm/md/lg)

---

## Development Workflow

### Pre-commit Hooks
Automated checks with ruff (linting + formatting) and mypy (type checking).

Install: `pre-commit install`

### Testing Strategy
143 comprehensive tests covering:
- Rhyme detection, syllable counting (with CMUdict fallback)
- API caching (diskcache validation)
- Deterministic outputs (seed reproducibility)
- Form generation (haiku/tanka/senryu validation)
- Metaphor/fragment quality scoring
- Network dependency mocking (pytest.skip decorators)

Run: `pytest tests/`

### CI/CD Pipeline
GitHub Actions workflow:
1. Lint job (ruff)
2. Type check job (mypy, non-blocking)
3. Test matrix (Python 3.9-3.12, Ubuntu/macOS)
4. Example generation (fixed seeds, artifact uploads)
5. Docker build validation

All checks run on push to master and pull requests.

---

## Future Directions

### Short-term (Next 2-4 weeks)
- Complete Tier 3 items #13-17 (export formats, logging, plugin system, layout, dashboard)
- Finalize Grammatical Templates Phases 5-7 (multi-template selection, interactive refinement, custom templates)

### Medium-term (2-4 months)
- Begin Tier 4 advanced capabilities
- Focus on Theme Explorer and Web Interface for accessibility

### Long-term (4+ months)
- Transformer model integration for state-of-the-art generation
- Interactive multimodal features
- Community plugin ecosystem
