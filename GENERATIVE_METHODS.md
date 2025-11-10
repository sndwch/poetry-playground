# Poetry Playground: Generative Methods Reference

Complete catalog of all 21 generative methods, their inputs, outputs, implementation details, and quality characteristics. This document serves as both a user reference and an AI ideation resource for future method development.

## Overview

Poetry Playground provides computational poetry tools organized into categories:

**Core Philosophy:**
- Quality over quantity: All methods use unified quality scoring
- Avoid clichés: 119 clichéd phrases + 217 overused words filtered
- Balance frequency: Not too common (boring), not too rare (obscure)
- Preserve creativity: Favor novelty, imagery, and poetic potential

**Recent Improvements:**
1. **Strategy Engine (2025-11-09)**: Multi-generator orchestration system
   - Hybrid execution: Parallel for independent generators, sequential for dependent
   - Universal normalization: All outputs to PoeticBuildingBlock format
   - Cross-type ranking: Quality-based ranking across generator types
   - First strategy: "Bridge Two Concepts" with 6 generators
2. **Equidistant Finder Quality Fix (2025-11-10)**: Now uses Quality Scorer instead of raw frequency
   - Before: Returned stop words ("the", "and") as top results
   - After: Returns creatively interesting words ("threshold", "vestige")
3. **TUI Stdout Pollution Fix**: Added `verbose` parameter to suppress progress messages in TUI
4. **Complete TUI Integration**: All 21 procedures now available in both CLI and TUI
5. **Rich Format Rendering**: ANSI code stripping for proper TUI display

**Method Categories:**

**Ideation Tools** (7 methods):
- Line Seeds Generator: Opening/pivot/ending suggestions
- Metaphor Generator: Extract patterns from literature
- Poetry Ideas: 10 categories from classic texts
- Resonant Fragments: Poetic sentence fragments
- Corpus Analyzer: Analyze your existing work
- Ship of Theseus: Transform poems gradually
- Six Degrees: Find convergence between concepts

**Semantic/Lexical Tools** (4 methods):
- Conceptual Cloud: 6-dimensional word associations
- Semantic Path: Transitional paths through meaning-space
- Equidistant Finder: Words between two anchors (quality-ranked)
- Related Rare Words: "Strange orbit" vocabulary

**Form Generators** (3 methods):
- Haiku: 5-7-5 syllable forms
- Tanka: 5-7-5-7-7 extended haiku
- Senryu: 5-7-5 human nature focus

**Visual/Experimental** (4 methods):
- Futurist Poem: Mathematical operators + wordplay
- Markov Generator: Joyce-like sequences
- Visual Puzzle: Terminal-based visual poetry
- PDF Generators (3): Chaotic/CharSoup/WordSoup (PDF-only)

**System Tools** (2 methods):
- Dependency Checker: Verify installation
- Template Generators: POS-based line generation

**Orchestration Tools** (1 method):
- Strategy Engine: Multi-generator orchestration system
  - Bridge Two Concepts: Semantic pathfinding with hybrid execution

---

## Table of Contents

1. [Metaphor Generation](#metaphor-generation)
2. [Line Seeds & Fragments](#line-seeds--fragments)
3. [Poetry Ideas](#poetry-ideas)
4. [Conceptual Cloud](#conceptual-cloud)
5. [Lexical Generation](#lexical-generation)
6. [Syllabic Form Generators](#syllabic-form-generators)
7. [Corpus Analysis](#corpus-analysis)
8. [Transformation & Path Finding](#transformation--path-finding)
9. [Visual Poetry (PDF)](#visual-poetry-pdf)
10. [Word Finding & Distance Tools](#word-finding--distance-tools)
11. [Markov Chain Generation](#markov-chain-generation)
12. [Template-Based Generation](#template-based-generation)
13. [Semantic & Phonetic Tools](#semantic--phonetic-tools)
14. [System Tools](#system-tools)
15. [Strategy Engine (Orchestration)](#strategy-engine-orchestration)

---

## Metaphor Generation

**Module:** `poetryplayground/metaphor_generator.py`

### `MetaphorGenerator.extract_metaphor_patterns()`

**Description:** Extracts metaphorical patterns from Project Gutenberg texts using regex patterns.

**How it works:**
- Retrieves diverse classic literature texts
- Scans for patterns like "X is like Y", "X as Y", "the X of Y"
- Validates word pairs for quality and English validity
- Applies comprehensive quality scoring and cliché filtering
- Sorts by quality and returns unique pairs

**Inputs:**
- `num_texts` (int, default=3): Number of different texts to sample from
- `verbose` (bool, default=True): Print progress messages (False for TUI, True for CLI)

**Expected outputs:**
- List of tuples: `[(source, target, context_sentence), ...]`
- Example: `[("silence", "amber", "The silence was like amber, preserving..."), ...]`
- Each metaphor has quality score ≥0.5

**Quality features:**
- Filters out clichéd phrases ("life is journey", "heart is fire")
- Prefers abstract→concrete mappings for vivid imagery
- Requires minimum word quality (avg ≥0.4)

---

### `MetaphorGenerator.generate_metaphor()`

**Description:** Generates a single fresh metaphor with connecting attributes.

**How it works:**
- Crosses semantic domains to find unexpected combinations
- Uses shared attributes to connect source and target
- Scores each metaphor for novelty, coherence, and imagery quality
- Returns high-quality metaphors with connecting grounds

**Inputs:**
- `source_domain` (str, optional): Starting semantic domain
- `target_domain` (str, optional): Target semantic domain

**Expected outputs:**
- `Metaphor` object with:
  - `text`: Full metaphor text
  - `source`: Tenor (what's being described)
  - `target`: Vehicle (what it's compared to)
  - `metaphor_type`: Type (simile, direct, etc.)
  - `quality_score`: 0-1 quality rating
  - `grounds`: List of connecting attributes

**Example:**
```python
Metaphor(
    text="silence is amber",
    source="silence",
    target="amber",
    metaphor_type=MetaphorType.DIRECT,
    quality_score=0.88,
    grounds=["golden", "preserved", "ancient"]
)
```

---

## Line Seeds & Fragments

**Module:** `poetryplayground/line_seeds.py`

### `LineSeedGenerator.generate_opening_line()`

**Description:** Generates a strong opening line with forward momentum.

**How it works:**
- Selects generation strategy (juxtaposition, temporal shift, incomplete metaphor)
- Uses seed words to build evocative opening
- Evaluates quality across 4 dimensions (word quality, novelty, imagery, structure)

**Inputs:**
- `seed_words` (List[str]): Words to base generation on
- `mood` (str, optional): Mood/tone for the line

**Expected outputs:**
- `LineSeed` object with:
  - `text`: The opening line
  - `seed_type`: SeedType.OPENING
  - `strategy`: Generation strategy used
  - `momentum`: 0.7-0.95 (forward movement)
  - `openness`: 0.6-0.9 (possible directions)
  - `quality_score`: 0-1 overall quality

**Example:**
```python
LineSeed(
    text="Between silence and vestige, the threshold...",
    seed_type=SeedType.OPENING,
    strategy=GenerationStrategy.JUXTAPOSITION,
    momentum=0.86,
    openness=0.82,
    quality_score=0.88
)
```

---

### `LineSeedGenerator.generate_fragment()`

**Description:** Generates evocative incomplete fragments.

**How it works:**
- Uses template-based generation if available, falls back to patterns
- Creates 2-5 syllable fragments
- Expands seed words with similar meanings and phonetic relatives
- Filters by quality (removes clichés, prefers concrete imagery)

**Inputs:**
- `seed_words` (List[str]): Words to base generation on
- `position` (str, default="any"): Position in poem ('opening', 'middle', 'closing', 'any')

**Expected outputs:**
- `LineSeed` with fragment text
- Momentum: 0.4-0.7
- Openness: 0.7-1.0

**Example:**
```
"...beside strange silent..."
"...through granite, ash..."
```

---

### `LineSeedGenerator.generate_image_seed()`

**Description:** Generates vivid sensory imagery fragments.

**How it works:**
- Selects sensory mode (sight, sound, touch, smell, taste)
- Combines sensory vocabulary with contextual words
- Prioritizes concrete, vivid words (concreteness 0.6-0.9)

**Inputs:**
- `seed_words` (List[str]): Words to base generation on

**Expected outputs:**
- `LineSeed` with sensory imagery
- Quality threshold: Higher for imagery (≥0.6 preferred)

**Example:**
```
"the tender of sounds"
"shimmer, rust, granite"
"smoke intensity"
```

---

### `LineSeedGenerator.generate_pivot_line()`

**Description:** Generates a line that can change poem direction.

**How it works:**
- Uses contrasting or questioning templates
- Creates tension or shifts perspective
- High openness to allow direction change

**Inputs:**
- `seed_words` (List[str]): Words to base generation on

**Expected outputs:**
- `LineSeed` with pivot line
- Momentum: 0.6-0.9
- Openness: 0.8-1.0

**Example:**
```
"But silence breaks distant..."
"Or was it threshold that whispers..."
"Until the vestige turns..."
```

---

### `LineSeedGenerator.generate_sonic_pattern()`

**Description:** Generates rhythmic/sonic template using phonetic patterns.

**How it works:**
- Gets phonetically related words (alliteration, assonance)
- Creates sound-based patterns
- Includes rhythm suggestions

**Inputs:**
- `seed_words` (List[str]): Words to base generation on

**Expected outputs:**
- `LineSeed` with sonic pattern
- Includes notes about sound patterns

**Example:**
```
"science, silent, silence"
"threshold-pause-vestige"
```

---

### `LineSeedGenerator.generate_ending_approach()`

**Description:** Suggests approaches for ending the poem.

**How it works:**
- Selects closing strategy (return to opening, sudden concrete, open question, etc.)
- Can echo/transform opening line if provided
- Creates resolution or leaves openness

**Inputs:**
- `seed_words` (List[str]): Words to base generation on
- `opening_line` (str, optional): Opening to echo/transform

**Expected outputs:**
- `LineSeed` with ending suggestion
- Momentum: 0.1-0.3 (low, winding down)
- Openness: 0.2-0.5
- Includes notes about closing approach

**Example:**
```
"All silence, no vestige"
"Just threshold."
"Again, Between silence and vestige..."
```

---

### `LineSeedGenerator.generate_seed_collection()`

**Description:** Generates a diverse collection of varied line seeds.

**How it works:**
- Generates one of each type (opening, pivot, image, sonic, ending)
- Fills remaining with random mix
- Sorts by quality score

**Inputs:**
- `seed_words` (List[str]): Words to base generation on
- `num_seeds` (int, default=10): Number of seeds to generate

**Expected outputs:**
- List of `LineSeed` objects, quality-sorted
- Guaranteed variety across seed types

---

## Personalized Line Seeds

**Module:** `poetryplayground/personalized_seeds.py`

### `PersonalizedLineSeedGenerator.generate_personalized_collection()`

**Description:** Generates line seeds that match your personal stylistic fingerprint by analyzing your existing poetry corpus.

**How it works:**
- Uses StyleFingerprint from personal corpus analysis to guide generation
- Creates **hybrid vocabulary pool**: blends fingerprint words (strictness%) with generic words
- Generates candidates using standard LineSeedGenerator strategies
- Scores each candidate on **two dimensions**:
  1. **Universal quality** (70%): Uses standard quality scorer
  2. **Style fit** (30%): Matches 4 style dimensions from fingerprint
- Returns top-scoring seeds that are both high-quality AND personally authentic

**Style matching dimensions (25% each):**
1. **Line length patterns**: Matches typical syllable counts from your poems
2. **POS sequence patterns**: Matches syntactic structures you naturally use
3. **Concreteness ratio**: Balances abstract vs concrete language like your work
4. **Phonetic patterns**: Matches alliteration/rhyme frequency in your style

**Inputs:**
- `fingerprint` (StyleFingerprint): From PersonalCorpusAnalyzer
- `strictness` (float, 0.0-1.0, default=0.7): Vocabulary bias
  - 0.0 = fully generic vocabulary (ignores fingerprint words)
  - 1.0 = only fingerprint vocabulary (maximum personalization)
  - 0.7 = recommended balance (70% your words, 30% generic)
- `count` (int, default=20): Number of seeds to generate
- `min_quality` (float, default=0.5): Minimum universal quality threshold
- `min_style_fit` (float, default=0.4): Minimum style match threshold

**Expected outputs:**
- List of `LineSeed` objects with added style metadata:
  - `quality_score`: Universal quality (0-1)
  - `style_fit_score`: How well it matches your style (0-1)
  - `style_components`: Dict with breakdown:
    - `line_length`: Syllable count match (0-1)
    - `pos_pattern`: Syntactic structure match (0-1)
    - `concreteness`: Abstract/concrete balance match (0-1)
    - `phonetic`: Sound pattern match (0-1)
  - Sorted by combined score: 0.7 × quality + 0.3 × style_fit

**Example:**
```python
from poetryplayground.corpus_analyzer import PersonalCorpusAnalyzer
from poetryplayground.personalized_seeds import PersonalizedLineSeedGenerator

# Step 1: Analyze your poetry corpus
analyzer = PersonalCorpusAnalyzer()
fingerprint = analyzer.analyze_directory("~/my_poems")

# Step 2: Create personalized generator
generator = PersonalizedLineSeedGenerator(
    fingerprint=fingerprint,
    strictness=0.7  # 70% your vocabulary, 30% generic
)

# Step 3: Generate personalized line seeds
seeds = generator.generate_personalized_collection(
    count=20,
    min_quality=0.5,
    min_style_fit=0.4
)

# Step 4: Review results
for seed in seeds[:5]:  # Top 5 seeds
    quality = seed.quality_score
    style = seed.style_fit_score
    combined = 0.7 * quality + 0.3 * style

    print(f"{seed.text}")
    print(f"  Quality: {quality:.2f} | Style: {style:.2f} | Combined: {combined:.2f}")
    print(f"  Style breakdown: {seed.style_components}")
```

**Example output:**
```
Between silence and vestige, the threshold...
  Quality: 0.88 | Style: 0.82 | Combined: 0.86
  Style breakdown: {
    'line_length': 1.0,    # Perfect match to your typical line length
    'pos_pattern': 0.87,   # Strong syntactic similarity
    'concreteness': 0.72,  # Good abstract/concrete balance
    'phonetic': 0.68       # Some sound pattern alignment
  }

Stone whispers through amber light
  Quality: 0.82 | Style: 0.78 | Combined: 0.81
  Style breakdown: {
    'line_length': 0.8,
    'pos_pattern': 0.92,
    'concreteness': 0.85,
    'phonetic': 0.55
  }
```

**Quality features:**
- **Dual scoring**: Ensures seeds are BOTH high-quality AND authentically "you"
- **Strictness control**: Balance between personal voice and creative exploration
- **Transparent scoring**: See exactly why each seed fits your style
- **Adaptive vocabulary**: Uses YOUR words while maintaining craft standards
- **Style awareness**: Matches not just words but HOW you arrange them

**Use cases:**
- Generate opening lines that "sound like you"
- Find fragments that fit your natural rhythm
- Maintain authentic voice while exploring new directions
- Break through writer's block with personally-resonant material
- Study what makes your style distinctive

**Comparison to standard LineSeedGenerator:**

| Feature | LineSeedGenerator | PersonalizedLineSeedGenerator |
|---------|-------------------|------------------------------|
| **Vocabulary** | Generic 100K word corpus | Your words + generic (blended) |
| **Quality scoring** | Universal only | Universal + style fit |
| **Line length** | Random | Matches your typical length |
| **Syntax** | Generic templates | Matches your POS patterns |
| **Output feel** | High-quality but generic | High-quality AND personally authentic |

**Workflow integration:**
```python
# Typical workflow:
# 1. Analyze corpus once (save fingerprint)
# 2. Generate personalized seeds for each new poem
# 3. Adjust strictness based on how "experimental" vs "authentic" you want

# For familiar voice:
seeds = gen.generate_personalized_collection(strictness=0.9)  # Very personal

# For creative exploration:
seeds = gen.generate_personalized_collection(strictness=0.3)  # More generic

# For balanced approach:
seeds = gen.generate_personalized_collection(strictness=0.7)  # Recommended
```

**Performance notes:**
- First run: ~5-10s (builds vocabulary pool, loads spaCy)
- Subsequent runs: ~2-3s (cached vocabularies)
- Strictness only affects vocabulary selection, not generation time
- Generating 20 seeds: ~3x longer than standard generator (due to dual scoring)

**Quality thresholds:**
- `min_quality`: Reject universally poor seeds (0.5 recommended)
- `min_style_fit`: Reject seeds that don't match your style (0.4 recommended)
- Combined threshold: Implicitly 0.7×0.5 + 0.3×0.4 = 0.47 minimum combined score
- Excellent seeds: Combined score ≥0.80

**Limitations & future enhancements:**
- Current concreteness scoring: Limited to simple heuristic (needs classifier)
- Phonetic matching: Basic alliteration/rhyme counting (could use advanced phonetics)
- No emotional tone matching yet (planned enhancement)
- No meter/rhythm analysis yet (planned enhancement)

---

## Poetry Ideas

**Module:** `poetryplayground/idea_generator.py`

### `PoetryIdeaGenerator.generate_ideas()`

**Description:** Mines classic literature for creative seeds across 10 categories.

**How it works:**
- Retrieves diverse Project Gutenberg documents
- Extracts ideas using category-specific regex patterns
- Applies type-specific quality thresholds
- Adaptive scaling: retrieves more documents if needed
- Deduplicates across all categories

**Inputs:**
- `num_ideas` (int, default=20): Target number of ideas
- `preferred_types` (List[IdeaType], optional): Filter to specific types

**Expected outputs:**
- `IdeaCollection` containing 10 categories:
  - `emotional_moments`: Feelings, realizations (threshold ≥0.55)
  - `vivid_imagery`: Visual scenes, landscapes (threshold ≥0.60)
  - `character_situations`: People in scenarios (threshold ≥0.55)
  - `philosophical_fragments`: Big questions, concepts (threshold ≥0.50)
  - `setting_descriptions`: Places, times (threshold ≥0.55)
  - `dialogue_sparks`: Quoted speech (threshold ≥0.55)
  - `opening_lines`: First sentences (threshold ≥0.55)
  - `sensory_details`: Smell, taste, touch (threshold ≥0.60)
  - `conflict_scenarios`: Tensions, contrasts (threshold ≥0.55)
  - `metaphysical_concepts`: Soul, spirit, transcendence (threshold ≥0.50)

**Example:**
```python
IdeaCollection(
    vivid_imagery=[
        PoetryIdea(
            text="The crimson light fell across the ancient stones.",
            idea_type=IdeaType.VIVID_IMAGERY,
            source_preview="From 'The Scarlet Letter'...",
            creative_prompt="Use this image as the central metaphor for a poem",
            keywords=["crimson", "light", "ancient", "stones"],
            quality_score=0.75
        )
    ],
    ...
)
```

---

### `IdeaCollection.get_random_mixed_selection()`

**Description:** Get a random mix of ideas across all types.

**How it works:**
- Optionally weights selection toward higher quality ideas
- Takes from top 60% when prefer_quality=True

**Inputs:**
- `count` (int): Number of ideas to return
- `prefer_quality` (bool, default=True): Weight toward higher quality

**Expected outputs:**
- List of `PoetryIdea` objects

---

## Conceptual Cloud

**Module:** `poetryplayground/conceptual_cloud.py`

### `generate_conceptual_cloud()`

**Description:** Generates six types of word clusters around a center word ("poet's radar").

**How it works:**
- For each cluster type, retrieves related words
- Scores each word for quality
- Combines word quality with type-specific factors
- Returns top k words per cluster

**Inputs:**
- `center_word` (str): The center word
- `k_per_cluster` (int, default=10): Words per cluster
- `total_limit` (int, default=50): Maximum total words
- `sections` (List[str], optional): Which clusters to include
- `include_scores` (bool, default=True): Include quality scores
- `cache_results` (bool, default=True): Cache API calls
- `min_score` (float, default=0.0): Minimum quality threshold
- `allow_rare` (bool, default=True): Allow rare words

**Expected outputs:**
- `ConceptualCloud` object containing 6 cluster types:
  1. **Semantic**: Near-meaning words (spaCy + Datamuse)
  2. **Contextual**: Collocations, words that appear together
  3. **Opposite**: Antonyms
  4. **Phonetic**: Rhymes, similar sounds
  5. **Imagery**: Concrete nouns with sensory qualities
  6. **Rare**: Unusual but related words

Each cluster contains `CloudTerm` objects with:
- `term`: The word
- `cluster_type`: Which cluster it belongs to
- `score`: Quality score (0-1)
- `freq_bucket`: 'common', 'mid', or 'rare'
- `metadata`: Additional info (POS, concreteness, etc.)

**Example:**
```python
ConceptualCloud(
    center_word="silence",
    clusters={
        ClusterType.SEMANTIC: [
            CloudTerm("quiet", ClusterType.SEMANTIC, 0.88, "mid"),
            CloudTerm("stillness", ClusterType.SEMANTIC, 0.85, "mid"),
        ],
        ClusterType.IMAGERY: [
            CloudTerm("stone", ClusterType.IMAGERY, 0.92, "mid",
                     metadata={"concreteness": 0.96}),
        ],
        ...
    },
    total_terms=42
)
```

**Quality features:**
- Semantic: Quality-ranked by word quality
- Contextual: Penalizes clichéd phrases (e.g., "heart breaks")
- Opposite: Combines Datamuse relevance + word quality
- Phonetic: Quality-ranked rhymes
- Imagery: 50% word quality + 50% concreteness
- Rare: Quality + rarity bonus

---

## Lexical Generation

**Module:** `poetryplayground/lexigen.py`

### `similar_meaning_words()`

**Description:** Finds words with similar meanings using spaCy embeddings and Datamuse.

**How it works:**
- Uses spaCy word vectors for semantic similarity
- Falls back to Datamuse means-like endpoint
- Deduplicates and validates

**Inputs:**
- `word` (str): Target word
- `sample_size` (int, default=10): Number of results

**Expected outputs:**
- List of similar words: `["quiet", "stillness", "hush", ...]`

---

### `contextually_linked_words()`

**Description:** Finds words that appear in similar contexts.

**How it works:**
- Uses Datamuse rel_trg (triggers) endpoint
- Finds words that appear near the target word

**Inputs:**
- `word` (str): Target word
- `sample_size` (int, default=10): Number of results
- `datamuse_api_max` (int, default=20): Max API results

**Expected outputs:**
- List of contextually linked words

---

### `frequently_following_words()`

**Description:** Finds words that frequently follow the target word.

**How it works:**
- Uses Datamuse left-context endpoint
- Returns words commonly appearing after target

**Inputs:**
- `word` (str): Target word
- `sample_size` (int, default=10): Number of results
- `datamuse_api_max` (int, default=20): Max API results

**Expected outputs:**
- List of following words

---

### `phonetically_related_words()`

**Description:** Finds words with similar sounds (rhymes, near-rhymes).

**How it works:**
- Uses CMU Pronouncing Dictionary via `pronouncing` library
- Gets perfect rhymes, slant rhymes
- Falls back to Datamuse sounds-like endpoint

**Inputs:**
- `word` (str): Target word
- `sample_size` (int, default=10): Number of results
- `include_slant` (bool, default=True): Include near-rhymes

**Expected outputs:**
- List of phonetically related words

---

### `related_rare_words()`

**Description:** Finds unusual but related words ("strange orbit").

**How it works:**
- Gets semantic neighbors
- Filters for low-frequency words (Zipf < 3.5)
- Validates quality to avoid just obscure junk

**Inputs:**
- `word` (str): Target word
- `sample_size` (int, default=10): Number of results
- `max_zipf` (float, default=3.5): Maximum frequency

**Expected outputs:**
- List of rare but related words

---

## Markov Chain Generation

**Module:** `poetryplayground/markov_generator.py`

### `MarkovGenerator.generate_line()`

**Description:** Generates lines using Markov chains trained on input text.

**How it works:**
- Builds n-gram model from training text
- Walks chain to generate sequences
- Optionally constrains by syllable count
- Can use custom start words

**Inputs:**
- `start_word` (str, optional): Word to start with
- `max_words` (int, default=15): Maximum words in line
- `target_syllables` (int, optional): Target syllable count

**Expected outputs:**
- Generated line as string

**Example:**
```
"The silence of ancient stones whispers"
```

---

### `MarkovGenerator.generate_poem()`

**Description:** Generates a multi-line poem using Markov chains.

**How it works:**
- Generates multiple lines using generate_line()
- Can respect stanza structure
- Optionally maintains syllable patterns

**Inputs:**
- `num_lines` (int, default=4): Number of lines
- `lines_per_stanza` (int, optional): Stanza structure
- `syllable_pattern` (List[int], optional): Syllables per line

**Expected outputs:**
- Multi-line poem as string

---

## Template-Based Generation

**Module:** `poetryplayground/grammatical_templates.py`

### `TemplateGenerator.generate_line()`

**Description:** Generates grammatically correct lines from POS templates.

**How it works:**
- Selects random POS template (e.g., "ADJ NOUN VERB")
- Fills each slot with appropriate word
- Validates syllable count
- Checks grammatical agreement

**Inputs:**
- `target_syllables` (int): Target syllable count
- `max_attempts` (int, default=100): Generation attempts

**Expected outputs:**
- `(line, template)` tuple
- Line conforms to syllable constraint

**Example:**
```python
("lonely commuter waits", "ADJ NOUN VERB")
("granite whispers softly", "NOUN VERB ADV")
```

---

### `TemplateGenerator.generate_phrase()`

**Description:** Generates short grammatical phrases (2-4 syllables).

**How it works:**
- Uses simpler templates for fragments
- Common patterns: "ADJ NOUN", "NOUN VERB", "ADV VERB"

**Inputs:**
- `target_syllables` (int, default=3): Target syllables

**Expected outputs:**
- Short phrase string

---

## Semantic & Phonetic Tools

**Module:** `poetryplayground/semantic_geodesic.py`

### `SemanticSpace.find_nearest()`

**Description:** Finds nearest neighbors in semantic vector space.

**How it works:**
- Uses pre-built k-NN index of word vectors
- Fast approximate nearest neighbor search
- Excludes specified words

**Inputs:**
- `vector` (ndarray): Query vector
- `k` (int, default=10): Number of neighbors
- `exclude` (Set[str], optional): Words to exclude

**Expected outputs:**
- List of `(word, similarity_score)` tuples

---

### `SemanticSpace.find_path()`

**Description:** Finds semantic path between two words.

**How it works:**
- Interpolates between word vectors
- Finds words at intermediate points
- Creates "semantic journey" from start to end

**Inputs:**
- `start_word` (str): Starting word
- `end_word` (str): Ending word
- `num_steps` (int, default=5): Interpolation steps

**Expected outputs:**
- List of words forming semantic path

**Example:**
```python
find_path("silence", "thunder", num_steps=5)
# ["silence", "quiet", "murmur", "rumble", "thunder"]
```

---

## Syllabic Form Generators

**Module:** `poetryplayground/forms.py`

### `FormGenerator.generate_haiku()`

**Description:** Generates 5-7-5 syllable haiku with grammatical templates.

**How it works:**
- Uses POS-based templates for natural grammar
- Validates syllable counts strictly (5-7-5)
- Can incorporate seed words into generation
- Uses quality-scored vocabulary

**Inputs:**
- `seed_words` (List[str], optional): Words to include/inspire
- `strict` (bool, default=True): Enforce strict 5-7-5 syllable count

**Expected outputs:**
- Tuple of `(lines, validation_result)`
- `lines`: List of 3 strings (one per line)
- `validation_result`: SyllableValidation object with:
  - `valid`: Boolean
  - `syllable_pattern`: Actual pattern (e.g., "5-7-5")
  - `target_pattern`: Expected pattern
  - Report with details

**Example:**
```python
lines, validation = gen.generate_haiku(seed_words=["silence", "stone"], strict=True)
# lines = [
#     "Ancient silence waits",  # 5
#     "Stone whispers through the threshold",  # 7
#     "Vestige disappears"  # 5
# ]
# validation.valid = True
```

---

### `FormGenerator.generate_tanka()`

**Description:** Generates 5-7-5-7-7 syllable tanka (extended haiku form).

**How it works:**
- Same template-based approach as haiku
- Two additional 7-syllable lines create turn/resolution
- Often more narrative or emotional than haiku

**Inputs:**
- `seed_words` (List[str], optional): Words to include/inspire
- `strict` (bool, default=True): Enforce strict syllable count

**Expected outputs:**
- Tuple of `(lines, validation_result)`
- `lines`: List of 5 strings
- Pattern: 5-7-5-7-7

**Example:**
```python
lines, validation = gen.generate_tanka(seed_words=["winter"])
# 5 lines following 5-7-5-7-7 pattern
```

---

### `FormGenerator.generate_senryu()`

**Description:** Generates 5-7-5 syllable senryu (human nature focus, vs. haiku's nature focus).

**How it works:**
- Same syllable structure as haiku (5-7-5)
- Focuses on human nature, humor, irony
- Uses different vocabulary domains (human behavior vs. nature)

**Inputs:**
- `seed_words` (List[str], optional): Words to include/inspire
- `strict` (bool, default=True): Enforce strict syllable count

**Expected outputs:**
- Same as haiku (3 lines, 5-7-5)
- Thematic focus on human nature

**Example:**
```python
lines, validation = gen.generate_senryu()
# Focus on human situations, emotions, ironies
```

---

## Corpus Analysis

**Module:** `poetryplayground/corpus_analyzer.py`

### `PersonalCorpusAnalyzer.analyze_directory()`

**Description:** Analyzes your existing poetry collection for style insights and patterns.

**How it works:**
- Scans directory for text files
- Extracts linguistic patterns:
  - Vocabulary preferences
  - POS distributions
  - Line length patterns
  - Rhyme schemes
  - Imagery types
  - Recurring themes
- Generates "fingerprint" of your style

**Inputs:**
- `directory_path` (str): Path to poetry collection
- File formats: .txt, .md, .poem

**Expected outputs:**
- `CorpusFingerprint` object with:
  - `vocabulary_profile`: Frequency analysis
  - `pos_distribution`: Part-of-speech patterns
  - `line_metrics`: Average line length, stanza structure
  - `imagery_profile`: Concrete vs. abstract ratio
  - `phonetic_patterns`: Rhyme/assonance frequency
  - `thematic_clusters`: Topic modeling results

**Reports generated:**
1. **Style Report**: What makes your poetry distinctive
2. **Inspiration Report**: Suggestions based on your patterns

**Example:**
```python
analyzer = PersonalCorpusAnalyzer()
fingerprint = analyzer.analyze_directory("/path/to/poems")

# Generate insights
style_report = analyzer.generate_style_report(fingerprint)
inspiration = analyzer.generate_inspiration_report(fingerprint)
```

**Key insights provided:**
- Vocabulary diversity (type/token ratio)
- Dominant imagery types (visual, auditory, tactile)
- Syntactic complexity
- Tonal patterns
- Gaps/opportunities for expansion

---

## Transformation & Path Finding

**Module:** `poetryplayground/ship_of_theseus.py`

### `ShipOfTheseusTransformer.gradual_transform()`

**Description:** Gradually transforms a poem while maintaining structure (inspired by Ship of Theseus paradox).

**How it works:**
- Replaces words one at a time with semantically similar alternatives
- Preserves structural constraints:
  - Part-of-speech (optional)
  - Syllable counts (optional)
  - Line breaks and formatting
- Each step creates intermediate version
- Returns complete transformation sequence

**Inputs:**
- `original` (str): Original poem text
- `steps` (int, default=5): Number of transformation steps
- `preserve_pos` (bool, default=True): Keep same parts of speech
- `preserve_syllables` (bool, default=True): Maintain syllable counts

**Expected outputs:**
- List of `TransformationResult` objects, one per step:
  - `transformed`: Text at this step
  - `replacement_ratio`: Proportion of words replaced (0.0-1.0)
  - `num_replacements`: Total word count changed
  - `preserved_structure`: Boolean

**Example:**
```python
transformer = ShipOfTheseusTransformer()
results = transformer.gradual_transform(
    original="The ancient silence waits beyond the threshold.",
    steps=3,
    preserve_pos=True,
    preserve_syllables=True
)

# Step 0: "The ancient silence waits beyond the threshold."  (0% replaced)
# Step 1: "The timeless silence waits beyond the doorway."   (40% replaced)
# Step 2: "The eternal stillness waits across the entrance." (60% replaced)
# Step 3: "The perpetual quiet remains across the portal."   (100% replaced)
```

**Use cases:**
- Explore variations while maintaining "feel"
- Generate alternative versions for selection
- Study how meaning shifts with word substitutions
- Create "parallel universe" versions of poems

---

**Module:** `poetryplayground/six_degrees.py`

### `SixDegrees.find_convergence()`

**Description:** Finds where two semantic paths converge ("six degrees of separation" for words).

**How it works:**
- Starts from both words simultaneously
- Expands semantic neighborhoods at each step
- Searches for overlap in the semantic graphs
- Returns convergence point and paths

**Inputs:**
- `word_a` (str): First word
- `word_b` (str): Second word
- `max_depth` (int, default=6): Maximum search depth

**Expected outputs:**
- `ConvergenceResult` object with:
  - `convergence_word`: Where paths meet
  - `path_a`: Steps from word_a to convergence
  - `path_b`: Steps from word_b to convergence
  - `total_steps`: Combined path length
  - `similarity_scores`: Semantic similarity at each step

**Example:**
```python
sd = SixDegrees()
result = sd.find_convergence("silence", "thunder")

# Possible result:
# path_a: silence → quiet → sound
# path_b: thunder → noise → sound
# convergence_word: "sound"
# total_steps: 4
```

**Use cases:**
- Explore conceptual bridges between ideas
- Generate transitional vocabulary
- Find unexpected connections
- Create thematic arcs

---

**Module:** `poetryplayground/semantic_geodesic.py`

### `find_semantic_path()`

**Description:** Finds transitional paths through meaning-space with multiple path methods.

**How it works:**
- Creates semantic "geodesic" (shortest/smoothest path) between words
- Multiple pathfinding methods:
  - **Linear**: Straight interpolation through vector space
  - **Bezier**: Smooth curved path with control points
  - **Shortest**: A* search for minimum semantic distance
- Returns primary path plus alternatives at each step
- Calculates path quality metrics

**Inputs:**
- `start_word` (str): Starting word
- `end_word` (str): Ending word
- `steps` (int, default=5): Number of intermediate steps (min 3)
- `k` (int, default=3): Alternative words per step
- `method` (str, default="linear"): Path method (linear/bezier/shortest)
- `semantic_space` (SemanticSpace, optional): Pre-loaded space

**Expected outputs:**
- `SemanticPath` object with:
  - `bridges`: List of BridgeWord lists (alternatives per step)
  - `smoothness_score`: 0-1 (higher = smoother transitions)
  - `deviation_score`: How much path curves
  - `diversity_score`: Variety in bridge words

**Example:**
```python
path = find_semantic_path(
    "silence", "thunder",
    steps=5,
    k=3,
    method="bezier"
)

# Primary path: silence → quiet → murmur → rumble → thunder
# Alternatives at step 2: [murmur, whisper, hum]
# Smoothness: 0.87 ★★★★☆
```

**Use cases:**
- Create thematic transitions in poems
- Explore semantic relationships
- Generate word sequences for development
- Find elegant conceptual bridges

**Path method differences:**
- **Linear**: Direct, predictable, fast
- **Bezier**: Smooth, curved, aesthetic
- **Shortest**: Optimized, minimal semantic distance

---

## Visual Poetry (PDF)

**Modules:** `poetryplayground/chaotic_generator.py`, `poetryplayground/charsoup_generator.py`, `poetryplayground/wordsoup_generator.py`

### `ChaoticConcretePoemPDFGenerator`

**Description:** Creates PDF visual poetry with abstract spatial arrangements.

**How it works:**
- Places words at random positions
- Uses mathematical chaos functions for placement
- Creates visual patterns through typography
- PDF-only (cannot display in terminal)

**Inputs:**
- `input_words` (List[str]): Words to arrange
- `page_size` (tuple, optional): PDF dimensions
- `font_variations` (bool, default=True): Vary fonts/sizes

**Expected outputs:**
- PDF file with visual arrangement
- No text return (visual medium only)

**Example use:**
```python
gen = ChaoticConcretePoemPDFGenerator()
gen.generate_pdf(input_words=["silence", "vestige", "threshold"])
# Creates visually arranged PDF
```

---

### `CharSoupPoemPDFGenerator`

**Description:** Creates PDF poems with character-level visual chaos.

**How it works:**
- Breaks words into individual characters
- Scatters characters spatially
- Creates typographic experiments
- Emphasizes visual over semantic reading

**Inputs:**
- `input_words` (List[str]): Words to deconstruct
- `density` (float, optional): Character density

**Expected outputs:**
- PDF file with character arrangements
- Abstract visual poetry

---

### `StopWordSoupPDFGenerator`

**Description:** Creates minimalist PDF poems using only stop words ("the", "and", "of").

**How it works:**
- Extracts stop words from input
- Arranges in visual patterns
- Explores grammatical function as aesthetic
- Minimalist approach

**Inputs:**
- `input_words` (List[str]): Words to filter
- `layout_style` (str, optional): Arrangement pattern

**Expected outputs:**
- PDF file with stop word arrangements
- Minimalist visual poetry

**Note:** All three PDF generators are CLI/API only - not available in TUI (visual output incompatible with terminal display).

---

### `PoemGenerator.poem_from_word_list()` (Visual Puzzle Poem)

**Description:** Creates terminal-based interactive visual puzzle poems from word lists.

**How it works:**
- Takes word list input
- Creates visual arrangement for terminal display
- Uses ASCII art and positioning
- Interactive/playful presentation
- Terminal-compatible (unlike PDF generators)

**Inputs:**
- `input_words` (List[str]): Words to arrange

**Expected outputs:**
- Terminal-displayable visual poem
- Text-based visual arrangement
- Interactive puzzle elements

**Example:**
```python
from poetryplayground.poemgen import PoemGenerator

pgen = PoemGenerator()
puzzle = pgen.poem_from_word_list(["silence", "threshold", "vestige"])
# Creates terminal-displayable visual arrangement
```

**Note:** This is the terminal-friendly visual poetry option (vs. PDF generators).

---

## Word Finding & Distance Tools

**Module:** `poetryplayground/finders.py`

### `find_equidistant()`

**Description:** Finds words equidistant (by Levenshtein distance) from two anchor words.

**How it works:**
- Calculates Levenshtein distance between anchors
- Searches vocabulary for words at equal distance from both
- **QUALITY INTEGRATION (NEW)**: Ranks by creative quality, not frequency
  - Uses universal Quality Scorer
  - Filters out stop words and boring common words
  - Rewards novelty, balanced frequency, non-clichéd words
- Supports both orthographic (spelling) and phonetic distance
- Window parameter allows near-equidistant matches

**Inputs:**
- `a` (str): First anchor word
- `b` (str): Second anchor word
- `mode` (str, default="orth"): Distance mode
  - "orth": Orthographic (spelling) distance
  - "phono": Phonetic (sound) distance
- `window` (int, default=0): Distance tolerance (0=exact, 1=d±1, etc.)
- `min_zipf` (float, default=3.0): Minimum frequency (filters very rare)
- `pos_filter` (str, optional): Part of speech filter ("NOUN", "VERB", etc.)
- `syllable_filter` (Tuple[int, int], optional): (min, max) syllables

**Expected outputs:**
- List of `EquidistantHit` objects containing:
  - `word`: The equidistant word
  - `target_distance`: Distance between anchors
  - `dist_a`: Distance from anchor A
  - `dist_b`: Distance from anchor B
  - `mode`: "orth" or "phono"
  - `zipf_frequency`: Word frequency
  - `syllables`: Syllable count
  - `pos`: Part of speech
  - `score`: **Quality score** (novelty + balance, NOT raw frequency)

**Quality scoring (IMPROVED 2025-11-09):**
- **Before**: Scored by raw frequency (common words = high score)
  - Problem: Returned "the", "and", "in" as top results
- **After**: Uses Quality Scorer
  - Combines: novelty (0-1) + frequency balance (0-1) + exactness bonus
  - Stop words get LOW scores (e.g., "the" → 0.12)
  - Creative words get HIGH scores (e.g., "threshold" → 0.88)
  - Results: Creatively interesting, poetically useful words

**Example:**
```python
hits = find_equidistant("dog", "cat", mode="orth", window=0)

# OLD RESULTS (before quality fix):
# 1. the (score=5.93) ← boring stop word
# 2. and (score=5.87) ← boring stop word
# 3. in (score=5.21) ← boring stop word

# NEW RESULTS (after quality fix):
# 1. fog (score=4.82) ← creative, interesting
# 2. cod (score=4.65) ← less common, evocative
# 3. dot (score=4.58) ← simple but visual

# All mathematically correct (d=3/3) but NOW quality-ranked!
```

**Use cases:**
- Find creative bridge words between concepts
- Discover unexpected connections
- Generate word ladders
- Explore assonance/consonance patterns

**Scoring components:**
- Exactness: +2.0 for each exact distance match
- **Quality**: +5.0 × overall_quality (0-1) [NEW!]
- Length penalty: -0.1 × |word_len - avg_anchor_len|
- Rime bonus: +0.5 if shares rime with either anchor

---

**Module:** `poetryplayground/causal_poetry.py`

### `ResonantFragmentMiner.mine_fragments()`

**Description:** Extracts poetic sentence fragments from literature.

**How it works:**
- Scans Project Gutenberg texts
- Identifies sentence fragments with poetic qualities:
  - Incomplete syntax (deliberate fragments)
  - Evocative imagery
  - Rhythmic qualities
  - Emotional resonance
- Scores fragments for poetic quality
- Returns diverse selection

**Inputs:**
- `target_count` (int, default=50): Number of fragments desired
- `num_texts` (int, default=5): Texts to mine from
- `min_quality` (float, default=0.5): Minimum quality threshold

**Expected outputs:**
- `FragmentCollection` with:
  - Fragments organized by pattern type
  - Each fragment has quality score
  - Source attribution included

**Fragment patterns detected:**
- Prepositional phrases ("beyond the threshold...")
- Participial phrases ("whispered through silence...")
- Appositives ("the vestige, ancient and worn...")
- Absolute phrases ("silence settled, the threshold crossed...")
- Noun phrases with modifiers ("the distant, amber light...")

**Example:**
```python
miner = ResonantFragmentMiner()
collection = miner.mine_fragments(target_count=50, num_texts=5)

# Sample fragments:
# "beyond the ancient threshold..."
# "where silence meets vestige..."
# "through amber, through ash..."
```

**Use cases:**
- Find evocative openings
- Discover transitional phrases
- Collect imagery fragments
- Study poetic syntax patterns

---

## Quality Scoring (Universal)

**Module:** `poetryplayground/quality_scorer.py`

All generative methods now use the unified quality scoring system:

### Key Quality Dimensions

1. **Frequency** (0-1): Balance between too rare and too common
   - Ideal range: 5e-06 to 5e-05
   - Too common (>1e-03): Boring, overused
   - Too rare (<1e-07): Obscure, inaccessible

2. **Novelty** (0-1): Avoidance of clichés
   - Checks against 119 clichéd phrases
   - Checks against 217 overused poetry words
   - Phrase-level and word-level detection

3. **Coherence** (0-1): Semantic fit with context
   - Currently based on domain fit
   - Future: semantic embedding similarity

4. **Register** (0-1): Formality/tone appropriateness
   - Currently neutral default (0.7)
   - Future: corpus-based register classification

5. **Imagery** (0-1): Concreteness matching
   - Based on 232 Brysbaert concreteness ratings
   - 0.0 = completely abstract (e.g., "truth" = 0.48)
   - 1.0 = completely concrete (e.g., "stone" = 0.96)
   - Scores distance from target concreteness

### Quality Score Output

All scored items return `QualityScore` with:
- `overall`: Weighted combination (0-1)
- `frequency`: Frequency score
- `novelty`: Cliché avoidance score
- `coherence`: Contextual fit score
- `register`: Formality score
- `imagery`: Concreteness score
- `component_scores`: Dict of all components

Grading scale:
- A+ (≥0.90): Exceptional
- A (≥0.80): Excellent
- B (≥0.70): Good
- C (≥0.60): Acceptable
- D (≥0.50): Marginal
- F (<0.50): Poor

---

## Usage Patterns

### Example: Generate high-quality metaphors
```python
from poetryplayground.metaphor_generator import MetaphorGenerator

gen = MetaphorGenerator()
patterns = gen.extract_metaphor_patterns(num_texts=5)

# Only high-quality metaphors (quality ≥ 0.5)
quality_metaphors = [(s, t) for s, t, _ in patterns]
```

### Example: Create a conceptual cloud
```python
from poetryplayground.conceptual_cloud import generate_conceptual_cloud

cloud = generate_conceptual_cloud(
    center_word="silence",
    k_per_cluster=10,
    sections=["semantic", "imagery", "rare"]
)

# Access high-quality imagery words
imagery_terms = cloud.get_cluster(ClusterType.IMAGERY)
best_images = [t for t in imagery_terms if t.score > 0.8]
```

### Example: Generate quality-filtered ideas
```python
from poetryplayground.idea_generator import PoetryIdeaGenerator, IdeaType

gen = PoetryIdeaGenerator()
ideas = gen.generate_ideas(
    num_ideas=30,
    preferred_types=[IdeaType.VIVID_IMAGERY, IdeaType.SENSORY_DETAIL]
)

# Get high-quality sensory ideas
sensory = ideas.get_ideas_by_type(IdeaType.SENSORY_DETAIL)
best = [idea for idea in sensory if idea.quality_score > 0.7]
```

### Example: Generate line seed collection
```python
from poetryplayground.line_seeds import LineSeedGenerator

gen = LineSeedGenerator()
seeds = gen.generate_seed_collection(
    seed_words=["silence", "threshold", "vestige"],
    num_seeds=15
)

# Seeds are already quality-sorted (best first)
best_seeds = seeds[:5]  # Top 5

# Filter by quality
high_quality = [s for s in seeds if s.quality_score > 0.8]
```

---

## Quality Thresholds by Type

### Metaphors
- Minimum quality: 0.5
- Cliché detection: threshold 0.6 for phrases
- Word quality: avg ≥ 0.4

### Line Seeds
- Fresh imagery/fragments: ≥ 0.75 preferred
- Clichéd but acceptable: 0.5-0.75
- Rejected: < 0.5

### Poetry Ideas
- Vivid imagery/sensory: ≥ 0.60
- Philosophical/metaphysical: ≥ 0.50
- Other types: ≥ 0.55

### Conceptual Cloud
- All scores normalized 0-1
- No hard threshold (sorted by quality)
- Typical range: 0.65-0.95

---

## Performance Notes

- **Caching**: Most API calls (Datamuse) are cached for 24 hours
- **Semantic space**: k-NN index loads once, then fast (<10ms per query)
- **Quality scoring**: ~1ms per word, ~5ms per phrase
- **Document retrieval**: Adaptive scaling prevents over-fetching
- **Template generation**: Fast (uses pre-built POS vocabulary)

---

## Future Enhancements

1. **Register scoring**: Corpus-based formality classification
2. **Coherence scoring**: Semantic embedding similarity
3. **Emotional tone**: Sentiment-aware generation
4. **Domain awareness**: Context-specific quality thresholds
5. **User preferences**: Personalized quality weights
6. **Quality learning**: Feedback-based score adjustment

---

## System Tools

**Module:** `poetryplayground/system_utils.py`

### `check_system_dependencies()`

**Description:** Verifies installation status of all required dependencies (spaCy, NLTK, semantic space, etc.).

**How it works:**
- Checks for spaCy and language models
- Verifies NLTK data packages
- Tests semantic space availability
- Checks Datamuse API connectivity
- Validates cache directories
- Reports version information

**Inputs:**
- None (system introspection)

**Expected outputs:**
- Formatted report string with:
  - ✓ Installed/working components
  - ✗ Missing/broken components
  - Version numbers
  - Installation instructions for missing items

**Example output:**
```
===========================================
SYSTEM DEPENDENCY CHECK
===========================================

✓ Python 3.12.9
✓ spaCy 3.7.2
✓ spaCy model: en_core_web_md (3.7.0)
✓ NLTK 3.8.1
✓ NLTK data: brown corpus
✓ NLTK data: cmudict
✓ Semantic space index: loaded (100,000 words)
✓ Datamuse API: responding
✓ Cache directory: /Users/name/.cache/poetryplayground

System ready for poetry generation!
```

**Use cases:**
- Diagnose installation issues
- Verify setup after install
- Check before running intensive operations
- Troubleshoot missing dependencies

**TUI Integration:**
Available as "Check System Dependencies" in TUI procedure list.

---

## Strategy Engine (Orchestration)

**Module:** `poetryplayground/strategy_engine.py`

### Overview

The **Strategy Engine** is a multi-generator orchestration system that acts as a "conductor" for complex creative workflows. Unlike single-purpose generators, strategies combine multiple generators (ConceptualCloud, LineSeedGenerator, MetaphorGenerator, SemanticPath) into cohesive "creative recipes" that fulfill high-level poetic briefs.

**Key Concepts:**
- **Strategy**: A creative recipe that orchestrates multiple generators
- **Hybrid Execution**: Parallel execution for independent generators, sequential for dependent ones
- **Universal Normalization**: All outputs converted to `PoeticBuildingBlock` format
- **Cross-Type Ranking**: Quality scores (0-1 scale) enable ranking across generator types
- **Plugin Architecture**: New strategies can be registered at runtime

**Design Philosophy:**
- Respect generator dependencies (parallel → sequential pipeline)
- Preserve original objects for rich display
- Filter by quality threshold (default: 0.5)
- Rank all results by unified quality score

### `StrategyEngine.execute()`

**Description:** Execute a registered strategy with given parameters.

**How it works:**
1. Validates strategy exists and params are valid
2. Instantiates strategy class
3. Runs strategy orchestration (parallel + sequential)
4. Normalizes disparate outputs to unified format
5. Ranks by quality score
6. Returns `StrategyResult` with metadata

**Inputs:**
- `strategy_name` (str): Name of registered strategy (e.g., "bridge_two_concepts")
- `params` (Dict[str, Any]): Strategy-specific parameters

**Expected outputs:**
- `StrategyResult` object containing:
  - `building_blocks`: List of `PoeticBuildingBlock` (ranked by quality)
  - `execution_time`: Time taken in seconds
  - `generators_used`: List of generator names invoked
  - `metadata`: Strategy-specific metadata
  - `params`: Original parameters

**Example:**
```python
from poetryplayground.strategy_engine import get_strategy_engine
from poetryplayground.strategies.bridge_two_concepts import BridgeTwoConceptsStrategy

# Get singleton engine
engine = get_strategy_engine()

# Register strategy
engine.register_strategy("bridge_two_concepts", BridgeTwoConceptsStrategy)

# Execute
result = engine.execute("bridge_two_concepts", {
    "start_word": "rust",
    "end_word": "forgiveness",
    "seed_words": ["severe", "quiet"]
})

# Access ranked results
for block in result.building_blocks[:10]:
    print(f"{block.text} ({block.quality_score:.2f}) - {block.source_method}")
```

**Output format:**
```
ash (0.92) - ConceptualCloud
But rust never forgives (0.87) - LineSeedGenerator
forgiveness is a slow erosion (0.85) - MetaphorGenerator
threshold (0.83) - SemanticPath
```

---

### Strategy: Bridge Two Concepts

**Module:** `poetryplayground/strategies/bridge_two_concepts.py`

**Description:** Bridges two words using semantic paths, conceptual clouds, metaphors, and line seeds. Demonstrates hybrid execution with parallel and sequential generator batches.

**How it works:**

**Phase 1 - Parallel Batch (Independent generators)**:
1. Semantic path finding (start → end)
2. Conceptual cloud for start word
3. Conceptual cloud for end word
4. Metaphor generation (start → end)

**Phase 2 - Sequential Batch (Dependent on Phase 1)**:
5. Opening line (using start + bridge words from path)
6. Pivot line (using all discovered vocabulary)

The strategy uses `ThreadPoolExecutor` for parallel execution, then extracts vocabulary from Phase 1 results to seed Phase 2 line generators.

**Inputs:**
- `start_word` (str, required): Starting concept
- `end_word` (str, required): Ending concept
- `seed_words` (List[str], optional): Additional tone/context words
- `num_steps` (int, default=5): Steps in semantic path (minimum 3)
- `k_per_cluster` (int, default=5): Terms per cloud cluster

**Expected outputs:**
- Ranked list of `PoeticBuildingBlock` objects from 6 generators
- Typical yield: 30-60 building blocks
- Quality score range: 0.5-1.0 (filtered by threshold)

**Building block types:**
- **BridgeWord**: Transitional vocabulary from semantic path
- **Imagery**: Concrete, visual words from conceptual clouds
- **Rare**: Uncommon words from conceptual clouds
- **Metaphor**: Creative connections between concepts
- **LineSeed-opening**: Opening lines with forward momentum
- **LineSeed-pivot**: Pivot lines using discovered vocabulary

**Example:**
```python
from poetryplayground.strategies.bridge_two_concepts import BridgeTwoConceptsStrategy

strategy = BridgeTwoConceptsStrategy(
    max_workers=4,              # Parallel threads
    min_quality_threshold=0.5   # Filter threshold
)

result = strategy.run({
    "start_word": "rust",
    "end_word": "forgiveness",
    "seed_words": ["severe", "quiet"],
    "num_steps": 5,
    "k_per_cluster": 5
})

# Examine results by type
for block in result.building_blocks:
    print(f"{block.block_type:20} | {block.text:40} | {block.quality_score:.2f}")
```

**Sample output:**
```
Block Type          | Text                                     | Score
--------------------+------------------------------------------+-------
imagery             | ash                                      | 0.92
LineSeed-opening    | But rust never forgives                  | 0.87
Metaphor            | forgiveness is a slow erosion            | 0.85
BridgeWord          | threshold                                | 0.83
imagery             | amber                                    | 0.81
rare                | vestige                                  | 0.79
LineSeed-pivot      | Severe rust at the threshold of amber    | 0.78
BridgeWord          | corrosion                                | 0.76
```

**Quality characteristics:**
- All blocks use unified QualityScorer (same 0-1 scale)
- Cross-type ranking is meaningful (LineSeed vs CloudTerm vs Metaphor)
- Metadata preserves source details (cluster_type, momentum, grounds, etc.)
- Original objects available for rich display

**CLI Integration:**
```bash
# Via CLI menu
poetry run python -m poetryplayground.cli
# Select: "⚙️ Strategy Engine: Bridge Two Concepts"

# Or via code
from poetryplayground.cli import strategy_engine_action
strategy_engine_action()
```

**Export options:**
- **JSON**: Full metadata, timestamped filename
- **Markdown**: Table format with quality scores
- **Display**: Grouped by block type, star ratings

**Performance:**
- Typical execution: 2-4 seconds
- Parallel speedup: ~2-3x vs sequential
- Generators with errors return empty results (non-blocking)

---

### Data Structures

#### `PoeticBuildingBlock`

Universal container for cross-generator results.

**Attributes:**
- `text` (str): The poetic content ("ash", "But rust never forgives...")
- `source_method` (str): Generator name ("ConceptualCloud", "LineSeedGenerator", etc.)
- `block_type` (str): Category ("BridgeWord", "Imagery", "Metaphor", "LineSeed-opening")
- `quality_score` (float): Universal quality score (0.0-1.0)
- `metadata` (Dict[str, Any]): Additional context (cluster_type, momentum, grounds, etc.)
- `original_object` (Any): Original generator output for rich display

**Example:**
```python
PoeticBuildingBlock(
    text="ash",
    source_method="ConceptualCloud",
    block_type="imagery",
    quality_score=0.92,
    metadata={
        "cluster_type": "IMAGERY",
        "freq_bucket": "rare",
        "concreteness": 0.96
    },
    original_object=CloudTerm(...)
)
```

#### `StrategyResult`

Output from strategy execution.

**Attributes:**
- `strategy_name` (str): Name of executed strategy
- `building_blocks` (List[PoeticBuildingBlock]): Ranked results
- `execution_time` (float): Seconds taken
- `generators_used` (List[str]): Generator names invoked
- `metadata` (Dict[str, Any]): Strategy-specific info
- `params` (Dict[str, Any]): Original parameters

---

### Future Strategies

The Strategy Engine is designed for extensibility. Planned strategies include:

**`explore_single_concept`**: Deep exploration of one word
- Semantic neighbors at multiple distances
- Conceptual cloud (all 6 dimensions)
- Metaphors using word as source and target
- Line seeds emphasizing the concept

**`emotional_arc`**: Create vocabulary for emotional journey
- Semantic path through emotional states
- Imagery for each stage of arc
- Metaphors for emotional transitions
- Opening/pivot/ending lines for narrative flow

**`concrete_to_abstract`**: Bridge physical to conceptual
- Semantic path from concrete to abstract
- Imagery cluster for concrete word
- Rare words for abstract concept
- Metaphors connecting domains

**`thematic_expansion`**: Expand a theme with complementary material
- Synonyms and semantic relatives
- Contextual collocations
- Opposite concepts for tension
- Metaphorical expressions

---

### Creating Custom Strategies

To create a new strategy:

1. **Inherit from `BaseStrategy`**:
```python
from poetryplayground.strategy_engine import BaseStrategy, StrategyResult

class MyStrategy(BaseStrategy):
    def validate_params(self, params):
        # Validate required parameters
        if "my_param" not in params:
            return False, "Missing my_param"
        return True, ""

    def run(self, params):
        # Orchestrate generators
        # Return StrategyResult
```

2. **Implement orchestration logic**:
```python
def run(self, params):
    # Phase 1: Run independent generators in parallel
    parallel_results = self._run_parallel_batch(...)

    # Phase 2: Run dependent generators sequentially
    sequential_results = self._run_sequential_batch(..., parallel_results)

    # Normalize and rank
    blocks = self.normalize_results({**parallel_results, **sequential_results})
    blocks.sort(key=lambda b: b.quality_score, reverse=True)

    return StrategyResult(
        strategy_name="my_strategy",
        building_blocks=blocks,
        execution_time=...,
        generators_used=[...],
        metadata={...}
    )
```

3. **Register with engine**:
```python
from poetryplayground.strategy_engine import get_strategy_engine

engine = get_strategy_engine()
engine.register_strategy("my_strategy", MyStrategy)
```

---

## Additional Generative Methods

### Futurist Poem Generator

**Module:** `poetryplayground/poemgen.py`

### `PoemGenerator.poem_line_from_word_list()`

**Description:** Creates Marinetti-inspired futurist poetry with mathematical connectors.

**How it works:**
- Expands word list with phonetic relatives
- Uses mathematical operators as connectors ("+", "-", "*", "%", "=", "!=", "::")
- Creates non-linear, fragmented aesthetic
- Emphasizes sound and visual impact over semantic meaning

**Inputs:**
- `word_list` (List[str]): Base vocabulary
- `connectors` (List[str], optional): Mathematical/symbolic connectors
- `max_line_length` (int, default=40): Maximum characters per line

**Expected outputs:**
- Single futurist-style line with mathematical connectors
- Example: "silence + threshold != vestige :: amber"

**Full poem generation:**
```python
pgen = PoemGenerator()
for _ in range(25):  # 25 lines
    line = pgen.poem_line_from_word_list(
        word_list=["silence", "threshold", "vestige"],
        connectors=[" + ", " - ", " * ", " % ", " = ", " != ", " :: "],
        max_line_length=40
    )
    print(line)
```

---

### Markov Poem Generator

**Module:** `poetryplayground/poemgen.py`

### `PoemGenerator.poem_from_markov()`

**Description:** Generates Joyce-like wordplay using Markov chains with rhyme schemes.

**How it works:**
- Builds Markov model from input words + related vocabulary
- Generates lines with probabilistic word sequences
- Incorporates rhyme patterns
- Balances coherence and surrealism

**Inputs:**
- `input_words` (List[str]): Seed vocabulary
- `num_lines` (int, default=10): Number of lines to generate

**Expected outputs:**
- Multi-line poem with Markov-generated sequences
- Maintains some semantic coherence while being playful
- May include internal rhymes and sound patterns

**Example:**
```python
pgen = PoemGenerator()
poem = pgen.poem_from_markov(
    input_words=["silence", "whisper", "echo"],
    num_lines=10
)
```

---

## AI Ideation Guide: Patterns & Opportunities

This section helps AI assistants identify patterns and gaps for proposing new generative methods.

### Successful Pattern Analysis

**What makes a good generative method:**
1. **Clear creative goal**: Not just "find words" but "find words that bridge two concepts creatively"
2. **Quality integration**: Uses Quality Scorer to filter/rank results
3. **Balanced complexity**: Not too simple (just an API call) nor too complex (50 parameters)
4. **Actionable output**: Gives poet usable material, not just data
5. **Complements existing tools**: Fills a gap rather than duplicating

**Strong existing patterns:**
- **Semantic exploration**: Conceptual Cloud (6 dimensions), Semantic Path (A→B trajectories)
- **Literature mining**: Metaphors, Ideas (10 categories), Fragments
- **Transformation**: Ship of Theseus (gradual), Markov (probabilistic)
- **Constraint-based**: Syllabic forms (haiku/tanka/senryu), Equidistant (distance-based)
- **Visual/spatial**: PDF generators, Puzzle poems

### Identified Gaps & Opportunities

**Unexplored semantic dimensions:**
- Temporal paths (past→present→future vocabulary)
- Emotional arcs (joy→ambivalence→sorrow)
- Abstract→concrete gradients (beyond just imagery cluster)
- Synesthetic mappings (sound words → color words)

**Underutilized constraints:**
- Meter/rhythm generators (iambic pentameter, etc.)
- Alliteration chains (maximum phonetic similarity sequences)
- Vowel progression poems (a→e→i→o→u orchestration)
- Stress pattern templates (beyond syllable count)

**Interactive/adaptive methods:**
- Collaborative completion (AI proposes next line based on poet's previous)
- Style mimicry (analyze poet's style, generate in that style)
- Constraint relaxation (start strict, gradually loosen)
- Quality threshold adaption (learn from poet's selections)

**Cross-method combinations:**
- Semantic Path + Syllabic Form = "5-7-5 path from A to B"
- Conceptual Cloud + Equidistant = "Cloud around midpoint of two words"
- Ship of Theseus + Quality Scorer = "Transform toward higher quality"
- Corpus Analysis + Line Seeds = "Generate seeds in your style"

**Novel data sources:**
- Song lyrics (modern language, rhythm patterns)
- Scientific abstracts (precise, technical metaphors)
- Historical documents (archaic vocabulary, formal syntax)
- Multilingual cognates (cross-language sound patterns)

**Unexplored outputs:**
- Revision suggestions (improve existing poem)
- Constraint detection (analyze poem, identify forms)
- Completion candidates (multiple ending options)
- Variation generator (N versions of same theme)

### Quality Integration Checklist

When proposing new methods, ensure:
- [ ] Uses `get_quality_scorer()` for word/phrase ranking
- [ ] Filters clichés (threshold ≥0.6 for phrases)
- [ ] Balances frequency (not too common, not too rare)
- [ ] Provides quality scores in output
- [ ] Documents quality thresholds used
- [ ] Explains why this quality metric matters for this method

### Method Proposal Template

```
Name: [Method Name]
Category: [Ideation/Semantic/Form/Visual/System]
Goal: [One sentence: what creative problem does this solve?]

Inputs:
- parameter1 (type, default): description
- parameter2 (type, default): description

Algorithm:
1. [Step 1: data gathering]
2. [Step 2: processing/filtering]
3. [Step 3: quality scoring]
4. [Step 4: output formatting]

Outputs:
- [What the poet receives]
- [Example output]

Quality Integration:
- [How quality scoring is used]
- [Thresholds and why]

Novel Contribution:
- [What gap does this fill?]
- [How is this different from existing methods?]

Example Use Case:
- [Concrete scenario where poet would use this]
```

### Integration Points

New methods can hook into existing infrastructure:

**Data sources:**
- `get_diverse_gutenberg_documents()`: Literature corpus
- `get_lexicon_data()`: 100K word vocabulary with POS/syllables/frequency
- `get_semantic_space()`: Vector space for similarity
- `DatamuseAPI()`: Rhyme, synonym, collocation data
- `WordValidator()`: English word validation

**Quality scoring:**
- `get_quality_scorer()`: Unified scoring system
- `QualityScore` object: 5-dimension quality analysis
- Cliché detection: Phrase-level and word-level
- Concreteness ratings: 232 Brysbaert norms

**Output formatting:**
- `format_as_rich()`: Terminal-friendly tables
- `format_as_markdown()`: Documentation-style
- `format_as_json()`: Structured data
- `format_as_simple()`: Plain text lists

### Success Metrics

Good generative methods:
- **High usage**: Poets use it regularly (vs. one-time novelty)
- **Quality output**: Average output quality ≥0.7
- **Low frustration**: <20% "no useful results" runs
- **Clear value**: Provides something poet couldn't easily do manually
- **Fast enough**: <10s for typical generation (or clear progress indication)

---

**Last Updated:** 2025-11-09 (Strategy Engine addition)
**Version:** 2.2 (21 procedures + Strategy Engine orchestration)

**Document Purpose:**
- User reference for existing methods
- AI assistant resource for understanding system capabilities
- Ideation framework for proposing new methods
- Quality standards documentation
