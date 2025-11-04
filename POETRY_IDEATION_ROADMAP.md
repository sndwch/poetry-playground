# Poetry Ideation Features - Implementation Roadmap

## Overview
Transform the generative poetry library from a generator of complete experimental poems into a comprehensive poetry ideation assistant that helps poets explore themes, discover unexpected connections, and break creative blocks.

## 📊 Implementation Status (Last Updated: 2025-11-04)

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

### 💡 Additional Features Implemented (4)
1. **Six Degrees Word Convergence** - Explores unexpected connections between concepts
2. **Ship of Theseus Transformer** - Gradually transforms existing poems
3. **Visual Poetry Generators** - Multiple concrete/visual poetry generation tools
4. **Document Library System** - Centralized Gutenberg text retrieval with anti-repetition

---

## 🎯 Feedback-Driven Improvements (ChatGPT + Gemini Analysis)

Based on comprehensive external code review, the following improvements have been prioritized by impact and feasibility. These integrate with existing roadmap features and address infrastructure, UX, and advanced capabilities.

### **Tier 1: Quick Wins** ⚡ (1-2 weeks)
**High Impact, Low-Medium Effort - Immediate Priority**

**Progress: 5/5 completed - ALL TIER 1 ITEMS COMPLETE! ✅**

1. **✅ Persistent API Caching** 🔧 - COMPLETED (2025-11-04)
   - ✅ Implemented diskcache for CMU/Datamuse lookups
   - ✅ Cache keying by (endpoint, params, version) with TTL
   - ✅ Exponential backoff with jitter + retry logic (max 3 attempts)
   - ✅ Offline mode support
   - **Results**: 158x speedup for Datamuse, 30x for CMU pronouncing
   - **Files**: `generativepoetry/cache.py`, `generativepoetry/lexigen.py`, `requirements.txt`
   - **Commit**: 882b5de

2. **✅ Deterministic/Reproducible Outputs** 🎲 - COMPLETED (2025-11-04)
   - ✅ Added `--seed INT` flag to CLI
   - ✅ Thread through Python random and numpy.random
   - ✅ Echo seed at start and end of session
   - ✅ Environment variable support (GP_SEED)
   - **Results**: Perfect determinism - same seed = same output
   - **Files**: `generativepoetry/seed_manager.py` (new), `generativepoetry/config.py`, `generativepoetry/cli.py`
   - **Commit**: f212708

3. **✅ Global CLI Flags** 🎨 - COMPLETED (2025-11-04)
   - ✅ `--out PATH` / `-o` for output directory control
   - ✅ `--format FORMAT` / `-f` for png/pdf/svg/txt output
   - ✅ `--quiet` / `-q` for suppressing non-essential output
   - ✅ `--verbose` / `-v` for detailed debug information
   - ✅ `--no-color` for better terminal compatibility
   - ✅ `--dry-run` for previewing without file generation
   - ✅ `--list-fonts` shows available PDF fonts (14 standard + custom)
   - ✅ `--list-procedures` shows all 13 generation methods
   - **Results**: Professional CLI with proper logging control and discovery features
   - **Files**: `generativepoetry/cli.py`, `generativepoetry/config.py`, `generativepoetry/logger.py`
   - **Commit**: 158ae6f

4. **✅ Automated Model Downloads** 📦 - COMPLETED (2025-11-04)
   - ✅ Created `generativepoetry/setup_models.py` with comprehensive setup functions
   - ✅ Added `setup()` function for complete automated installation
   - ✅ Implemented lazy-loading functions for on-demand model downloads
   - ✅ Added `--setup` CLI command for manual model installation
   - ✅ Fixed critical bug in decomposer.py (module-level imports causing crashes)
   - ✅ Fixed missing stopwords download in pdf.py
   - ✅ Centralized all model download logic from scattered locations
   - **Results**: Zero-friction first-run experience with automatic downloads
   - **Models managed**: NLTK (punkt, words, brown, wordnet, stopwords), spaCy (en_core_web_sm)
   - **Files**: `generativepoetry/setup_models.py` (new), `generativepoetry/cli.py`, `generativepoetry/decomposer.py`, `generativepoetry/pdf.py`
   - **Commit**: 4a38576

5. **✅ Graceful Error Handling** ⚠️ - COMPLETED (2025-11-04)
   - ✅ Enhanced check_system_dependencies() with beautiful formatted output
   - ✅ Added get_hunspell_install_instructions() with platform-specific instructions
   - ✅ Improved check_hunspell_installed() to verify both package and system libraries
   - ✅ Enhanced PDF PNG conversion error handling with specific exception handling
   - ✅ Added detailed error messages for PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
   - ✅ Improved hunspell setup to gracefully handle Windows (instead of raising exception)
   - ✅ Platform-specific instructions for macOS (brew), Ubuntu/Debian (apt-get), Fedora/RHEL (yum), Windows
   - **Results**: Clear, actionable error messages that guide users to resolve issues
   - **Platforms supported**: macOS, Linux (Ubuntu/Debian/Fedora/RHEL), Windows
   - **Files**: `generativepoetry/system_utils.py`, `generativepoetry/pdf.py`, `generativepoetry/utils.py`
   - **Commit**: 299107c

### **Tier 2: Foundation** 🏗️ (2-4 weeks) - 9/17 Complete
**Medium-High Impact, Medium Effort - Core Infrastructure**

6. **✅ Modern Configuration System** ⚙️ - COMPLETED (2025-11-04)
   - ✅ Migrated from simple dataclasses to Pydantic v2 BaseModel with validation
   - ✅ Added `[tool.generativepoetry]` section to pyproject.toml
   - ✅ Implemented multi-source config loading with proper priority chain
   - ✅ Added `--config CONFIG.yml` CLI flag for YAML config files
   - ✅ Implemented environment variable support with GP_* prefix
   - ✅ Added `--spacy-model {sm|md|lg}` CLI flag for model selection
   - ✅ Created comprehensive validation (min/max constraints, mutually exclusive flags)
   - ✅ Fixed cache.py to use lazy config loading (get_config())
   - ✅ Updated README.md with complete configuration documentation
   - **Results**: Config priority verified: CLI > YAML > pyproject.toml > env > defaults
   - **Benefit**: Professional packaging, type-safe configuration, flexible customization, validated settings
   - **Files**: `generativepoetry/config.py` (complete rewrite, 381 lines), `generativepoetry/cli.py`, `generativepoetry/cache.py`, `pyproject.toml`, `README.md`
   - **Commit**: 1788eb6

7. **✅ Type Safety & Linting** 🔍 - COMPLETED (2025-11-04)
   - ✅ Installed and configured ruff + mypy
   - ✅ Replaced black/isort/flake8 with modern ruff (10-100x faster)
   - ✅ Fixed 9 bare except clauses (E722) - now catch Exception explicitly
   - ✅ Fixed 11 mutable argument defaults (B006) - use None with initialization
   - ✅ Eliminated ALL star imports (F403/F405) - 81 violations fixed
   - ✅ Applied 40 code quality improvements (generators, boolean returns, etc.)
   - ✅ Set up pre-commit hooks with ruff linter + formatter
   - **Result**: Reduced violations from 164 → 29 (82% reduction, 135 fixes total)
   - **Remaining**: 29 non-critical violations (B007, SIM103, SIM102, RUF012, etc.)
   - **Benefit**: Catches bugs early, better IDE support, faster linting
   - **Files**: `pyproject.toml`, all `.py` files, `.pre-commit-config.yaml`
   - **Commits**: 5578ecd (critical fixes), 5cc7a40 (star imports), f30e730 (unsafe fixes), 7f69e68 (pre-commit)

8. **✅ CI/CD Pipeline** ✅ - COMPLETED (2025-11-04)
   - ✅ Created comprehensive GitHub Actions workflow (.github/workflows/ci.yml)
   - ✅ Lint job: Runs ruff linter and formatter checks
   - ✅ Type-check job: Runs mypy for type safety (non-blocking)
   - ✅ Test job: Matrix testing on Python 3.9, 3.10, 3.11, 3.12 across Ubuntu and macOS
   - ✅ Caching strategy: Python deps, NLTK data, spaCy models
   - ✅ Coverage reporting with Codecov integration
   - ✅ Build examples job: Generates example poems with fixed seeds (master only)
   - ✅ Docker build job: Validates image builds and CLI commands work
   - ✅ Created Dockerfile with pre-cached NLTK data and spaCy models
   - ✅ Added .dockerignore for optimized builds
   - ✅ System dependencies included (poppler-utils, git)
   - ✅ Updated README with comprehensive CI/CD and Docker documentation
   - **Results**: Full CI/CD pipeline ready, Docker image ~500MB with all dependencies
   - **Benefit**: Professional dev workflow, prevents regressions, reproducible environments
   - **Files**: `.github/workflows/ci.yml`, `Dockerfile`, `.dockerignore`, `README.md`
   - **Note**: PyPI publishing removed per user request (keeping project private)
   - **Commit**: `84c09f4` - Implement CI/CD Pipeline with GitHub Actions and Docker

9. **🔧 Comprehensive Test Suite** 🧪 - IN PROGRESS (2025-11-04)
   - ✅ Unit tests for rhyme/syllable pickers (18 tests in test_rhyme_syllable.py) - 4 failures to fix
   - ✅ Deterministic layout with seed (18 tests in test_deterministic.py) - all passing
   - ✅ Cache hit validation (9 tests in test_cache.py) - 6 failures to fix
   - ✅ Snapshot tests for PDF/SVG structure - not pixels (28 tests in test_pdf_structure.py) - all passing
   - ✅ Tests for config module (20 tests in test_config.py) - all passing
   - ✅ Tests for word_validator module (27 tests in test_word_validator.py) - all passing
   - ✅ Tests for corpus_analyzer module (25 tests in test_corpus_analyzer.py) - all passing
   - 🔧 Fixing pre-existing test failures (test_generativepoetry.py, test_line_seeds.py, test_improved_markov.py)
   - **Current Status**: 145 new tests created, 118 passing locally (81%), fixing remaining failures
   - **Progress**: test_config, test_corpus_analyzer, test_deterministic, test_pdf_structure, test_word_validator all 100% passing
   - **Remaining Work**: Fix cache API tests, network mocking issues, rhyme/syllable edge cases
   - **Coverage**: Tests for all major modules including new Tier 1 & 2 features
   - **Benefit**: Confident refactoring, catches regressions, validates deterministic behavior
   - **Files**: `tests/test_rhyme_syllable.py`, `tests/test_cache.py`, `tests/test_deterministic.py`, `tests/test_pdf_structure.py`, `tests/test_config.py`, `tests/test_word_validator.py`, `tests/test_corpus_analyzer.py`, existing `tests/test_generativepoetry.py`, `tests/test_markov.py`, `tests/test_line_seeds.py`, `tests/test_improved_markov.py`
   - **Commits**: `5a0513b`, `aa57bef`, `9c7b1b9`, `91c64d9`, `dd33490` (test fixes in progress)

10. **Observability & Profiling** 📊
    - Structured logging: procedure name, seed, timing, cache_hits, API_calls
    - `--profile` flag for per-stage timing table
    - **Benefit**: Performance optimization, debugging
    - **Files**: `generativepoetry/logger.py`, all modules

11. **Documentation Cleanup** 📝
    - Create `CONTRIBUTING.md` with one-command dev setup
    - Consolidate to single README (remove duplicates)
    - Add copyable example commands in `example_images/`
    - Pin deps with "known good" lock snapshot
    - **Benefit**: Easier onboarding, more contributors
    - **Files**: `CONTRIBUTING.md`, `README.md`, `requirements.lock`

### **Tier 3: Feature Expansion** 🚀 (4-8 weeks)
**High Impact, Medium-High Effort - User-Facing Features**

12. **Syllable-Aware Forms** 🎋
    - Haiku/senryu (5-7-5), tanka enforcement
    - CMUdict syllable counts with graceful fallbacks
    - Kireji-like "cut" detection (caesura heuristics)
    - `--form haiku` switch with constraint failure reports
    - **Benefit**: Publishable micro-forms, expands use cases
    - **Files**: `generativepoetry/forms.py` (new), CLI integration

13. **Constraint Solver Engine** 🧩
    - Small ILP/backtracking for: acrostics, pangrams, end-rhyme schemes, letter budgets
    - Accept `--constraints file.yml`
    - **Benefit**: Enables roadmap Feature 3 (Oulipian constraints)!
    - **Files**: `generativepoetry/constraint_solver.py` (new)

14. **Enhanced Metaphor Miner** 🔮
    - Deduplication across sessions
    - Cliché filters (penalize high web frequency)
    - Novelty scoring with conceptual distance
    - Formalize `--scale` parameter for distance control
    - **Benefit**: Higher quality metaphors, less repetition
    - **Files**: `generativepoetry/metaphor_generator.py`

15. **SVG-First Rendering** 🖼️
    - SVG as core render target (text remains selectable/editable)
    - Rasterize to PNG on request
    - Layout schema: `PoemDoc{page, frames[], items[]}`
    - Serialize to JSON alongside image for post-editing
    - **Benefit**: Better quality, post-editable outputs
    - **Files**: `generativepoetry/rendering.py` (new), visual poem generators

16. **Visual Intelligibility Helpers** 👁️
    - Collision detection for overlapping text
    - Minimal energy layout pass (force-directed)
    - Optional "reading rails" overlays (guides for traversal)
    - **Benefit**: More readable visual poems
    - **Files**: `generativepoetry/layout.py` (new)

17. **Corpus Analyzer Dashboard** 📊
    - Per-book/year signatures: bigrams, sentiment, POS rhythms
    - Average line length, "turn" frequency
    - Export small HTML report per corpus
    - Ship one public example
    - **Benefit**: Shows value, attracts academic users
    - **Files**: `generativepoetry/corpus_analyzer.py`, HTML templates

### **Tier 4: Advanced Capabilities** 🌟 (8+ weeks)
**High Impact, High Effort - Transformative Features**

18. **Evolutionary Ship of Theseus++** 🧬
    - Treat poem as genome with fitness function
    - Fitness on: meter/syllables, rhyme density, semantic distance to prompt
    - Iteratively mutate and select best N variants
    - Explain scoring for top 5 variants
    - **Benefit**: Automated multi-step transformation with quality control
    - **Files**: `generativepoetry/evolutionary.py` (new)

19. **Six Degrees → Story Scaffolds** 📖
    - 3-beat or 5-beat outlines (L1 image, L2 turn, L3 implication)
    - Integrate with Line Seeds Generator
    - Generate narrative arcs from concept graphs
    - **Benefit**: Bridges ideation to structure
    - **Files**: `generativepoetry/six_degrees.py`, `line_seeds.py`

20. **Theme and Concept Explorer** 🌐
    - *Integrate with roadmap Feature 1*
    - Use ConceptNet API for deeper semantic relationships
    - Wikipedia API for cultural/historical references
    - NRC Word-Emotion Association for emotional palette
    - **Benefit**: Completes original roadmap vision
    - **Files**: `generativepoetry/theme_explorer.py` (new)

21. **Web Interface/API** 🌍
    - Flask/FastAPI with `/generate`, `/analyze` endpoints
    - Simple web UI for all 13+ tools
    - uvicorn deployment-ready
    - **Benefit**: 10x accessibility, wider audience
    - **Files**: `generativepoetry/api/` (new), `web/` frontend

22. **Transformer Model Integration** 🤖
    - GPT-2/DistilGPT-2 for coherent stanza generation
    - Prompt-based ideation (theme → continuation)
    - `--model [classic|gpt2|distilgpt2]` flag
    - **Benefit**: State-of-the-art generation quality
    - **Files**: `generativepoetry/transformers.py` (new)

23. **Interactive & Multimodal Features** 🎭
    - Interactive visual editor (move/rotate/resize words in terminal or web)
    - Text-on-image backgrounds (blend typography with art)
    - Audio generation: `--speak` flag with gTTS/pyttsx3
    - Custom corpus support (upload .txt files, URLs, paste text)
    - **Benefit**: Creative empowerment, multimodal outputs
    - **Files**: `generativepoetry/interactive.py`, `audio.py`, `custom_corpus.py` (new)

---

## Feature 1: Theme and Concept Explorer ⏳ PLANNED
**Branch name:** `feature/theme-explorer`
**Status:** Not yet implemented

### Purpose
Create a brainstorming assistant that generates conceptual clusters and word constellations from seed words, presenting unexpected thematic directions and connections.

### Implementation Details

#### 1.1 Core Module: `generativepoetry/theme_explorer.py`
```python
class ThemeExplorer:
    def __init__(self):
        - Initialize with enhanced word relationship mappings
        - Load cultural/historical reference database
        - Set up metaphor pattern recognition

    def explore_theme(seed_words, depth=2, branches=5):
        - Start from seed words
        - For each depth level, expand outward finding:
          * Semantic relatives
          * Historical/cultural associations
          * Contrasting concepts (antonyms, tensions)
          * Sensory associations (colors, textures, sounds)
          * Emotional valences
        - Return ThemeCluster object

    def generate_constellation(theme_cluster):
        - Visualize relationships as ASCII art or networkx graph
        - Show connection strengths
        - Highlight unexpected bridges
```

#### 1.2 Data Structures
```python
class ThemeCluster:
    - seed_words: List[str]
    - primary_themes: Dict[str, float]  # theme -> relevance score
    - metaphor_suggestions: List[MetaphorPair]
    - cultural_references: List[Reference]
    - emotional_palette: Dict[str, float]
    - sensory_associations: Dict[str, List[str]]
    - contrasts_tensions: List[Tuple[str, str]]

class MetaphorPair:
    - source_domain: str
    - target_domain: str
    - connecting_attributes: List[str]
```

#### 1.3 New Data Sources
- Add Wikipedia API integration for cultural references
- Enhance with ConceptNet for deeper semantic relationships
- Add emotional lexicon (NRC Word-Emotion Association)
- Integrate sensory word mappings

#### 1.4 CLI Integration
Add menu option: "Explore Themes and Concepts"
- Prompt for seed words
- Ask for exploration depth (shallow/medium/deep)
- Display results in organized categories
- Option to save exploration results to file

#### 1.5 Example Output
```
Seed words: ocean, memory

Primary Themes Discovered:
- Depth/Surface (0.89)
- Erosion/Persistence (0.76)
- Tides/Cycles (0.71)
- Navigation/Lost (0.68)

Metaphor Suggestions:
- Memory as ocean floor (connecting: hidden, layered, pressure)
- Thoughts as tides (connecting: rhythmic, pulling, returning)

Cultural/Literary References:
- Proust's madeleine (memory trigger)
- Poseidon/Neptune (ocean deity, emotional turbulence)
- The Odyssey (sea journey as life journey)

Sensory Palette:
- Textures: smooth, gritty, wet, crystalline
- Sounds: whoosh, whisper, echo, crash
- Colors: cerulean, grey-green, pearl, midnight

Emotional Tensions:
- Vast ←→ Intimate
- Turbulent ←→ Calm
- Clear ←→ Murky
```

---

## Feature 2: Dynamic Metaphor Generator ✅ COMPLETED
**File:** `generativepoetry/metaphor_generator.py`
**Status:** Fully implemented and integrated into CLI
**Completed:** 2024

### Purpose
Generate fresh, unexpected metaphors by mining Project Gutenberg texts for patterns and cross-domain connections, providing poets with numerous metaphorical possibilities from a single prompt.

### Implementation Notes
- Fully implemented with adaptive scaling system
- Mines multiple Gutenberg texts for diverse metaphor patterns
- Supports 9 metaphor types: simile, direct, implied, possessive, appositive, compound, extended, conceptual, synesthetic
- CLI integration complete with interactive menu
- Quality scoring based on novelty, coherence, vividness, and semantic distance

### Implementation Details

#### 2.1 Core Module: `generativepoetry/metaphor_generator.py`
```python
class MetaphorGenerator:
    def __init__(self):
        - Initialize Gutenberg text cache
        - Load metaphor pattern templates
        - Set up domain classifiers
        - Initialize quality scoring

    def extract_metaphor_patterns(text_id=None):
        - Scan Gutenberg texts for metaphorical structures
        - Extract "X is/was like Y", "X as Y" patterns
        - Identify implicit metaphors through verb usage
        - Store patterns with context

    def generate_metaphor_batch(source_domain, target_domain, count=10):
        - Cross domains to create unexpected connections
        - Use multiple formulation styles
        - Score for novelty and coherence
        - Return sorted by quality

    def mine_descriptive_context(concept):
        - Find concept in Gutenberg texts
        - Extract surrounding descriptive words
        - Build associative word clouds
        - Return context-rich descriptions

    def create_extended_metaphor(base_metaphor):
        - Develop single metaphor into extended form
        - Add supporting imagery
        - Maintain consistency
        - Return multi-line exploration
```

#### 2.2 Metaphor Types
```python
METAPHOR_TYPES = {
    'simile': "X is like Y",
    'direct': "X is Y",
    'implied': "X [verb typically associated with Y]",
    'possessive': "Y's X" / "X of Y",
    'appositive': "X, that Y",
    'compound': "X-Y",
    'extended': Multi-sentence development,
    'conceptual': Abstract mapping (TIME IS MONEY),
    'synesthetic': Cross-sensory metaphors
}
```

#### 2.3 Domain Categories (from Gutenberg mining)
```python
DOMAINS = {
    'nature': ['ocean', 'forest', 'storm', 'garden', ...],
    'architecture': ['cathedral', 'bridge', 'tower', 'ruins', ...],
    'time': ['clock', 'season', 'dawn', 'century', ...],
    'body': ['heart', 'bones', 'blood', 'breath', ...],
    'cosmos': ['stars', 'void', 'orbit', 'constellation', ...],
    'technology': ['engine', 'wire', 'signal', 'machine', ...],
    'textiles': ['thread', 'weave', 'fray', 'pattern', ...],
    'music': ['symphony', 'discord', 'rhythm', 'silence', ...]
}
```

#### 2.4 Pattern Extraction from Gutenberg
```python
def extract_from_gutenberg():
    - Download random selection of texts
    - Use regex to find metaphorical structures
    - Parse with spaCy for syntax understanding
    - Extract: tenor, vehicle, grounds (connecting attributes)
    - Store successful patterns for reuse
    - Build frequency maps of domain crossings
```

#### 2.5 Quality Scoring
```python
def score_metaphor(metaphor):
    - Novelty: How unexpected is the connection?
    - Coherence: Do the domains share attributes?
    - Vividness: How concrete/imageable?
    - Emotional resonance: What feeling does it evoke?
    - Multiplicity: How many valid interpretations?
    - Literary precedent: Similar to known good metaphors?
```

#### 2.6 Example Output
```
=== Metaphor Generator ===
Source: memory
Target: ocean
Mining Gutenberg for patterns...

GENERATED METAPHORS:

Simple Forms:
- "Memory is an ocean floor, littered with shipwrecks"
- "Memories drift like kelp in tidal pools"
- "The ocean of recollection"

Extended Form:
"Memory is an ocean. Some days calm,
reflective as glass. Other days, storms
churn the sediment, and what was buried
rises, unbidden, to break upon the present's shore."

Implied Metaphors:
- "Memories ebb and flow"
- "Drowning in remembrance"
- "The tide of recollection"

Compound/Possessive:
- "Memory-waves"
- "Ocean's memory"
- "The sea-change of remembering"

From Gutenberg (Moby Dick):
"Memory, that archipelago of moments"
(Adapted from: "The archipelago of islands")

From Gutenberg (Proust):
"Memory, that vast ocean floor where time accumulates like sediment"
(Inspired by passages on time and memory)
```

---

## Feature 3: Prompt and Constraint Generator ⏳ PLANNED
**Branch name:** `feature/constraint-generator`
**Status:** Not yet implemented

### Purpose
Generate Oulipian writing constraints and creative prompts based on user interests, helping poets break patterns through structured challenges.

### Implementation Details

#### 2.1 Core Module: `generativepoetry/constraint_generator.py`
```python
class ConstraintGenerator:
    def __init__(self):
        - Load constraint templates
        - Initialize difficulty levels
        - Set up combination rules

    def generate_formal_constraint(seed_words, difficulty='medium'):
        - Select appropriate constraint type
        - Customize based on seed words
        - Return Constraint object with clear rules

    def generate_thematic_prompt(mood, style_preferences):
        - Combine unexpected elements
        - Add specific requirements
        - Include optional bonus challenges

    def generate_hybrid_prompt():
        - Combine formal and thematic constraints
        - Ensure they're compatible
        - Provide escape hatches for too-difficult combinations
```

#### 2.2 Constraint Types
```python
CONSTRAINT_TYPES = {
    'phonetic': [
        'hidden_rhyme',  # Each line contains internal rhyme with seed
        'echo_form',      # Last word of line becomes first of next
        'sound_gradient', # Gradual shift from one sound to another
    ],
    'semantic': [
        'antonym_bridge', # Connect opposites in each stanza
        'metaphor_chain', # Each metaphor builds on previous
        'register_shift', # Alternate formal/informal each line
    ],
    'structural': [
        'fibonacci_syllables', # Lines follow Fibonacci sequence
        'mirror_poem',        # Second half mirrors first
        'erasure_ready',     # Must work as poem and erasure source
    ],
    'lexical': [
        'vocabulary_exile',   # Forbidden common words
        'etymology_family',   # All words from same root
        'temporal_words',     # Only words from specific era
    ]
}
```

#### 2.3 Prompt Templates
```python
PROMPT_TEMPLATES = [
    "Write about {theme1} using only words that could describe {theme2}",
    "Create a poem where {constraint} but never mention {avoided_word}",
    "Use the rhythm of {sound_pattern} to explore {emotion}",
    "Write as if {perspective} observing {scene}",
    "Translate the feeling of {sensory1} into {sensory2} imagery"
]
```

#### 2.4 Difficulty Scaling
- **Easy**: Single constraint, familiar vocabulary
- **Medium**: 2-3 compatible constraints, some unusual words
- **Hard**: Multiple interlocking constraints, rare words required
- **Experimental**: Contradictory constraints requiring creative resolution

#### 2.5 Example Output
```
=== Your Poetry Constraint ===
Difficulty: Medium

Formal Constraint: "Echo Chamber"
- The last word of each line must rhyme with a word in the next line
- That rhyming word cannot be at the line's end
- Each stanza must return to a sound from the first stanza

Thematic Prompt:
Write about "digital loneliness" using only words that existed before 1950

Vocabulary Seeds (optional starting points):
- telegraph, distance, echo, wire, pulse, hollow, static

Bonus Challenge:
Include at least three words that have changed meaning since 1950

Example opening:
"The wire carries hollow songs at night,
Songs that fight against the closing door..."
```

---

## Feature 4: Line and Phrase Seed Generator ✅ COMPLETED
**File:** `generativepoetry/line_seeds.py`
**Status:** Fully implemented and integrated into CLI
**Completed:** 2024

### Purpose
Generate evocative incomplete phrases and line beginnings that serve as creative catalysts rather than finished products.

### Implementation Notes
- Fully implemented with 7 generation strategies
- Supports 7 seed types: opening, pivot, image, emotional, sonic, closing, fragment
- Quality evaluation with cliché detection
- CLI integration complete with interactive menu
- Uses centralized vocabulary from shared vocabulary module

### Implementation Details

#### 3.1 Core Module: `generativepoetry/line_seeds.py`
```python
class LineSeedGenerator:
    def __init__(self):
        - Load phrase patterns
        - Initialize unusual word combinations
        - Set up rhythm patterns

    def generate_opening_line(mood, style):
        - Create strong opening with forward momentum
        - Leave semantic space for development
        - Ensure rhythmic interest

    def generate_fragment(position='any'):
        - Create evocative incomplete thoughts
        - Use unusual but meaningful word pairings
        - Maintain multiple interpretation possibilities

    def generate_ending_approach():
        - Suggest final line strategies
        - Provide tonal options
        - Include circular return possibilities
```

#### 3.2 Generation Strategies
```python
GENERATION_STRATEGIES = {
    'juxtaposition': combine_distant_concepts,
    'synesthesia': cross_sensory_description,
    'incomplete_metaphor': start_comparison_without_resolution,
    'rhythmic_break': establish_then_disrupt_pattern,
    'question_implied': statement_that_suggests_question,
    'temporal_shift': mix_time_indicators,
    'perspective_blur': ambiguous_speaker_position
}
```

#### 3.3 Fragment Types
- **Opening Hooks**: Strong first lines with momentum
- **Pivot Points**: Lines that can change poem direction
- **Image Seeds**: Vivid but incomplete sensory descriptions
- **Emotional Triggers**: Phrases that evoke without stating
- **Sound Patterns**: Rhythmic/sonic templates to build from
- **Closure Approaches**: Various ways to end (question, image, statement, echo)

#### 3.4 Quality Control
```python
def evaluate_seed_quality(seed):
    - Check for clichés
    - Ensure semantic openness
    - Verify rhythmic interest
    - Measure evocative potential
    - Return quality score and improvement suggestions
```

#### 3.5 Example Output
```
=== Line Seeds for "nostalgia + technology" ===

Opening Lines:
1. "The modem's song was lullaby before..."
2. "In pixel-light, my grandmother's face..."
3. "We saved our voices on magnetic tape..."

Pivotal Fragments:
- "...through static, something almost like..."
- "...the blue screen's prayer..."
- "...buffering what we meant to say..."

Image Seeds:
- "dial tones in empty rooms"
- "screenshots of extinct websites"
- "the weight of unplugged phones"

Sonic Patterns to Build On:
- "click-whir-pause" (mechanical rhythm)
- "soft/stop/sought" (consonant progression)

Ending Approaches:
- Return to dial tone image but transformed
- Question about what replaces absence
- Single concrete detail that holds everything
```

---

## Feature 5: Personal Corpus Analyzer ✅ COMPLETED
**File:** `generativepoetry/corpus_analyzer.py`
**Status:** Fully implemented and integrated into CLI
**Completed:** 2024

### Purpose
Analyze a poet's existing work to understand their voice, then suggest expansions that maintain authenticity while pushing boundaries.

### Implementation Notes
- Fully implemented with comprehensive style fingerprinting
- Analyzes vocabulary, themes, emotional register, compositional patterns
- Generates vocabulary expansions using Datamuse API
- Creates inspired stanzas with systematic word substitutions
- CLI integration complete with detailed reporting
- Privacy-focused: local storage only, no external API calls with personal work

### Implementation Details

#### 4.1 Core Module: `generativepoetry/personal_analyzer.py`
```python
class PersonalCorpusAnalyzer:
    def __init__(self, corpus_path):
        - Load poet's existing works
        - Initialize style fingerprinting
        - Set up vocabulary analysis

    def analyze_style():
        - Identify recurring themes
        - Map vocabulary preferences
        - Detect rhythm patterns
        - Find characteristic phrases
        - Identify avoided areas

    def suggest_expansions():
        - Find adjacent vocabulary
        - Propose new theme combinations
        - Suggest form experiments
        - Identify productive constraints

    def generate_in_style():
        - Create seeds matching voice
        - Maintain characteristic rhythms
        - Use familiar + one unexpected element
```

#### 4.2 Analysis Dimensions
```python
StyleFingerprint:
    - vocabulary_complexity: float
    - avg_line_length: float
    - enjambment_frequency: float
    - metaphor_density: float
    - abstraction_level: float
    - emotional_range: Dict[str, float]
    - preferred_sounds: List[str]
    - avoided_words: Set[str]
    - characteristic_phrases: List[str]
    - theme_clusters: List[ThemeCluster]
```

#### 4.3 Expansion Strategies
- **Adjacent Vocabulary**: Words one step away from current usage
- **Theme Bridging**: Connect two existing themes not yet linked
- **Form Stretching**: Slightly modify familiar forms
- **Sound Exploration**: Emphasize underused sonic qualities
- **Register Mixing**: Blend formal levels not yet combined

#### 4.4 Privacy and Storage
```python
class SecureCorpus:
    - Local storage only
    - Optional encryption
    - No external API calls with personal work
    - Clear data deletion method
```

#### 4.5 Example Output
```
=== Your Poetry Analysis ===

Style Fingerprint:
- You prefer concrete imagery (78% concrete vs abstract)
- Average line length: 8-10 syllables
- High enjambment rate (65%)
- Recurring themes: memory, landscape, family
- Characteristic sounds: soft consonants, long vowels

Vocabulary Insights:
- Most used: water, light, stone, remember
- Never used but adjacent: cascade, luminous, granite, recollect
- Emotional palette: melancholy (45%), wonder (30%), longing (25%)

Suggested Expansions:
1. Theme Bridge: "landscape + technology" (unexplored combination)
2. Vocabulary: Add geological terms (stratified, erosion, sediment)
3. Form: Try prose poetry (you excel at enjambment)
4. Sound: Explore harsh consonants for contrast

Personalized Seed:
"Where water meets granite, the light..." (uses your vocabulary + one new element)
```

---

## Feature 6: Mood and Register Mapping ⏳ PLANNED
**Branch name:** `feature/mood-register`
**Status:** Not yet implemented

### Purpose
Add emotional intelligence to word selection, helping poets find words that carry the right emotional weight and register for their intended effect.

### Implementation Details

#### 5.1 Core Module: `generativepoetry/mood_mapper.py`
```python
class MoodMapper:
    def __init__(self):
        - Load emotional lexicons (NRC, VADER)
        - Initialize register classifications
        - Set up nuance detection

    def analyze_emotional_valence(word):
        - Return emotional dimensions
        - Identify primary and secondary affects
        - Measure intensity

    def find_register_alternatives(word, target_register):
        - Find synonyms at different formality levels
        - Maintain emotional consistency
        - Preserve sonic qualities where possible

    def suggest_mood_progression(start_mood, end_mood, steps):
        - Create emotional arc
        - Suggest transitional words
        - Identify pivot points
```

#### 5.2 Emotional Dimensions
```python
EMOTIONAL_AXES = {
    'valence': (-1.0, 1.0),  # negative to positive
    'arousal': (0.0, 1.0),   # calm to excited
    'dominance': (0.0, 1.0), # submissive to dominant
}

EMOTIONAL_CATEGORIES = [
    'joy', 'sadness', 'anger', 'fear', 'surprise',
    'disgust', 'trust', 'anticipation', 'nostalgia',
    'melancholy', 'yearning', 'serenity', 'unease'
]
```

#### 5.3 Register Levels
```python
REGISTER_LEVELS = {
    'formal': {'utilize', 'commence', 'endeavor'},
    'neutral': {'use', 'start', 'try'},
    'informal': {'grab', 'kick off', 'give it a shot'},
    'poetic': {'employ', 'dawn', 'essay'},
    'archaic': {'wield', 'betake', 'assay'}
}
```

#### 5.4 Nuance Detection
```python
def detect_nuance(word):
    - Identify subtle connotations
    - Check cultural associations
    - Note potential triggers
    - Measure cliché risk
    - Return NuanceProfile
```

#### 5.5 Example Output
```
=== Mood Mapping for "grief" ===

Current Analysis:
- Valence: -0.8 (strongly negative)
- Arousal: 0.3 (relatively calm)
- Dominant emotions: sadness (0.9), loss (0.8)

Register Alternatives:
- More formal: "bereavement", "lamentation"
- Same level: "sorrow", "mourning"
- More informal: "heartache", "pain"
- Poetic: "dolour", "keen"

Lighter Alternatives (if needed):
- "melancholy" (-0.5 valence, more wistful)
- "wistfulness" (-0.3 valence, includes yearning)
- "pensiveness" (-0.2 valence, thoughtful sadness)

Words to Avoid (cliché risk):
- "tears" (overused with grief)
- "broken" (overused metaphor)
- "heavy heart" (phrase is clichéd)

Suggested Progression to Hope:
grief → sorrow → melancholy → quietude → acceptance → peace → hope
```

---

## Additional Implemented Features

### Poetry Idea Generator ✅ COMPLETED
**File:** `generativepoetry/idea_generator.py`
**Status:** Fully implemented and integrated into CLI
**Completed:** 2024

**Purpose:** Mine classic literature from Project Gutenberg to extract creative seeds across 10 distinct categories.

**Categories:**
1. Emotional Moments
2. Vivid Imagery
3. Character Situations
4. Philosophical Fragments
5. Setting Descriptions
6. Dialogue Sparks
7. Opening Lines
8. Sensory Details
9. Conflict Scenarios
10. Metaphysical Concepts

**Key Features:**
- Uses regex patterns to identify meaningful fragments from classic texts
- Adaptive scaling system maintains quality while increasing yield
- Anti-repetition tracking ensures diverse source texts
- Configurable targets (10-200 ideas per session)
- Quality filtering and deduplication

---

### Resonant Fragment Miner ✅ COMPLETED
**File:** `generativepoetry/causal_poetry.py`
**Status:** Fully implemented and integrated into CLI
**Completed:** 2024

**Purpose:** Extract poetic sentence fragments from Project Gutenberg texts using 26 specialized patterns across 5 categories.

**Fragment Categories:**
1. **Causality** (8 patterns) - Because/therefore/thus relationships
2. **Temporal** (6 patterns) - Time-based transitions
3. **Universal** (4 patterns) - Always/never/all statements
4. **Singular** (4 patterns) - Only/alone/single focus
5. **Modal** (4 patterns) - Might/could/would possibilities

**Key Features:**
- 26 specialized regex patterns for identifying resonant fragments
- Quality scoring based on length, punctuation, capitalization
- Configurable targets (10-200 fragments per session)
- Adaptive scaling maintains 0.65+ quality threshold
- CLI integration with detailed progress reporting

---

### Six Degrees Word Convergence ✅ COMPLETED
**File:** `generativepoetry/six_degrees.py`
**Status:** Fully implemented and integrated into CLI
**Completed:** 2024

**Purpose:** Explore unexpected pathways between any two concepts using semantic relationships.

**Key Features:**
- Breadth-first search through word relationship networks
- Uses Datamuse API for semantic connections
- Finds shortest paths between disparate concepts
- Reveals surprising conceptual bridges
- CLI integration for interactive exploration

---

### Ship of Theseus Transformer ✅ COMPLETED
**File:** `generativepoetry/poem_transformer.py`
**Status:** Fully implemented and integrated into CLI
**Completed:** 2024

**Purpose:** Gradually transform existing poems by systematically replacing words while maintaining structure.

**Key Features:**
- Multiple transformation modes (semantic, phonetic, contextual)
- Preserves poem structure and line breaks
- Progressive transformation with configurable percentage
- Shows original and transformed versions side-by-side
- CLI integration for interactive transformation

---

### Document Library System ✅ COMPLETED
**File:** `generativepoetry/document_library.py`
**Status:** Core infrastructure component
**Completed:** 2024

**Purpose:** Centralized Gutenberg text retrieval with anti-repetition tracking.

**Key Features:**
- Anti-repetition tracking ensures diverse text sources
- Text signature system prevents duplicate documents
- Configurable minimum length requirements per use case
- Shared across all text-mining modules
- Performance optimization with caching

---

## Implementation Order & Strategy

### ✅ Phase 1: Foundation - COMPLETED
1. ✅ Created base classes and data structures
2. ✅ Added centralized configuration system (`config.py`)
3. ✅ Implemented document library with anti-repetition tracking
4. ✅ Set up adaptive scaling system across all modules
5. ✅ Centralized vocabulary system

### ✅ Phase 2: Core Feature Development - COMPLETED
**Implemented in this order:**
1. ✅ **Line Seeds Generator** - Simplest, immediate value
2. ✅ **Metaphor Generator** - Mining Gutenberg for metaphorical patterns
3. ✅ **Personal Corpus Analyzer** - Complex style fingerprinting
4. ✅ **Poetry Idea Generator** - 10 categories of creative seeds
5. ✅ **Resonant Fragment Miner** - 26 specialized patterns

### ✅ Phase 3: Integration & Enhancement - COMPLETED
1. ✅ Updated CLI with all features (13 interactive tools)
2. ✅ Added comprehensive README documentation
3. ✅ Performance optimization with caching
4. ✅ Quality thresholds and filtering systems
5. ✅ Export/save functionality for all tools

### 🚧 Phase 4: Remaining Features - IN PLANNING
**Next priorities:**
1. ⏳ **Theme and Concept Explorer** - Brainstorming assistant for thematic clusters
2. ⏳ **Constraint Generator** - Oulipian writing constraints
3. ⏳ **Mood Mapper** - Emotional intelligence for word selection

### 📊 Phase 5: Future Enhancement - ONGOING
1. User feedback integration (review ChatGPT/Gemini feedback)
2. Additional data sources (ConceptNet, NRC Emotion Lexicon)
3. Web interface (optional)
4. Collaborative features (prompt sharing, constraint challenges)

---

## Testing Strategy

### Unit Tests
- Test each word relationship function
- Verify emotional mappings
- Check constraint generation logic
- Validate style analysis

### Integration Tests
- Full pipeline from input to output
- CLI interaction flows
- Data persistence
- API rate limiting

### User Testing
- Poet feedback sessions
- A/B testing different approaches
- Usability studies
- Output quality assessment

---

## Data Requirements

### New Data Sources Needed
1. **NRC Emotion Lexicon** (free for research)
2. **ConceptNet** API (for semantic relationships)
3. **Wikipedia** API (for cultural references)
4. **Historical word frequency** data
5. **Sensory word** mappings
6. **Register classification** database

### Storage Considerations
- Local caching of API responses
- User preference storage
- Personal corpus encryption
- Session history

---

## Success Metrics

### Quantitative
- Time to generate useful idea
- Number of seeds used in actual poems
- User retention rate
- Feature usage statistics

### Qualitative
- User satisfaction surveys
- Quality of generated ideas
- Diversity of outputs
- Success breaking writer's block

---

## Future Possibilities

### Advanced Features
1. **Collaborative mode** - Multiple poets building on each other
2. **Historical style** - Generate in style of different eras
3. **Cross-cultural** - Incorporate non-English poetic traditions
4. **Music integration** - Generate to rhythm/meter of songs
5. **Visual poetry** - Add concrete/visual poetry tools
6. **Performance scripts** - Generate for spoken word
7. **Translation bridge** - Use translation as generation method
8. **Dream logic** - Surrealist automatic writing simulation

### Machine Learning Enhancements
1. Fine-tuned language models for poetry
2. Style transfer between poets
3. Emotion trajectory prediction
4. Metaphor quality scoring
5. Cliché detection ML model

### Community Features
1. Prompt sharing marketplace
2. Constraint challenges
3. Collaborative anthologies
4. Style fingerprint sharing
5. Workshop integration tools

---

## 📋 Summary & Next Steps

### Current State Assessment
**Strong Foundation Established:**
- ✅ 5 core ideation features completed (Line Seeds, Metaphors, Corpus Analyzer, Ideas, Fragments)
- ✅ 4 additional features implemented (Six Degrees, Ship of Theseus, Visual Generators, Document Library)
- ✅ Adaptive scaling system maintains quality across all text-mining tools
- ✅ Centralized configuration and vocabulary systems
- ✅ 13 interactive CLI tools available

**Areas for Improvement (from external feedback):**
- Infrastructure: Caching, determinism, configuration management
- UX: CLI ergonomics, error handling, documentation
- Testing: Unit tests, CI/CD, type safety
- Features: Constraint solver, syllable-aware forms, enhanced metaphors

### Recommended Implementation Order

**Phase A: Infrastructure Sprint** (Weeks 1-2)
Focus on Tier 1 quick wins to improve developer experience and user satisfaction:
1. Persistent API caching → immediate CLI speed boost
2. `--seed` flag → reproducibility for debugging/demos
3. Global CLI flags → professional UX
4. Automated model downloads → reduce friction
5. Better error messages → happier users

**Phase B: Foundation Strengthening** (Weeks 3-6)
Build professional infrastructure for long-term maintainability:
1. Modern config system (pyproject.toml + Pydantic)
2. Type hints + linting (ruff + mypy)
3. CI/CD pipeline (GitHub Actions)
4. Comprehensive test suite
5. Documentation cleanup (CONTRIBUTING.md)

**Phase C: Feature Expansion** (Weeks 7-14)
Implement user-facing features that expand capabilities:
1. Syllable-aware forms (haiku, tanka) → new use cases
2. Constraint solver engine → enables roadmap Feature 3!
3. Enhanced metaphor miner → quality improvements
4. SVG-first rendering → better visual output
5. Corpus analyzer dashboard → showcase value

**Phase D: Advanced Capabilities** (Weeks 15+)
Transformative features for maximum impact:
1. Theme Explorer (roadmap Feature 1) with ConceptNet/Wikipedia
2. Web interface/API → 10x accessibility
3. Transformer integration → modern ML capabilities
4. Evolutionary Ship of Theseus++ → advanced transformations

### Integration with Existing Roadmap

**Feedback addresses 3 planned features:**
- **Feature 1 (Theme Explorer)**: Tier 4, item #20 - ConceptNet + Wikipedia integration
- **Feature 3 (Constraint Generator)**: Tier 3, item #13 - Constraint solver engine
- **Feature 6 (Mood Mapper)**: Can integrate with NRC Emotion Lexicon (mentioned in Tier 4, item #20)

**New capabilities from feedback:**
- Syllable-aware forms enable formal poetry structures
- SVG rendering improves visual poem quality
- Web API democratizes access
- Transformer integration provides state-of-the-art generation

### Success Metrics for Next Phase

**Engineering Metrics:**
- Cache hit rate > 80% for API calls
- Test coverage > 70%
- Type hint coverage > 90%
- CI passing on all Python versions (3.9-3.12)

**User Experience Metrics:**
- Installation success rate (automated model downloads)
- Time to first successful generation < 2 minutes
- CLI error clarity (user-reported confusion rate)

**Feature Adoption:**
- % of users trying new constraint solver
- Haiku generation attempts per week
- Web API request volume (once deployed)

### Action Items for Next Commit

1. ✅ Update POETRY_IDEATION_ROADMAP.md with feedback synthesis (COMPLETED)
2. ⏳ Implement Tier 1, item #1: Persistent API caching
3. ⏳ Implement Tier 1, item #2: `--seed` flag
4. ⏳ Update README.md to mention reproducibility features
5. ⏳ Create GitHub issue templates for bugs/features
6. ⏳ Begin CONTRIBUTING.md draft

### Maintenance Policy

**Roadmap Updates Required:**
- When starting any Tier 1-4 item → Move to "In Progress"
- When completing any feature → Update status, add implementation notes
- Monthly review → Reassess priorities based on user feedback
- Use `/update-roadmap` slash command to enforce consistency

**Commit Message Format:**
```
[category] Brief description

- Detailed change 1
- Detailed change 2
- Update POETRY_IDEATION_ROADMAP.md if feature-related

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Categories:** `[infra]`, `[feature]`, `[fix]`, `[docs]`, `[test]`, `[ci]`

---

## Appendix: External Review Sources

**ChatGPT Analysis** (2025-11-04)
- Focus: Engineering best practices, caching, determinism, CI/CD
- Key insight: "Datamuse & CMU lookups are perfect for tiny persistent cache"
- Recommendation: SVG-first rendering with JSON serialization

**Gemini Analysis** (2025-11-04)
- Focus: User experience, modern ML integration, web accessibility
- Key insight: "Automate model downloads to reduce friction"
- Recommendation: Flask/FastAPI web interface for wider audience

**Integration:** Combined feedback provides balanced view of infrastructure needs (ChatGPT) and feature expansion opportunities (Gemini), with significant overlap on caching, configuration, and testing priorities.
