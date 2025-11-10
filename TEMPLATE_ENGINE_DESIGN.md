# Template Engine Design Document

## Phase 2.1: Existing Patterns Analysis

### Current Generator Outputs

#### 1. MetaphorGenerator → Metaphor dataclass
```python
@dataclass
class Metaphor:
    text: str                      # "Love is a red rose"
    source: str                    # Tenor (what's being described): "love"
    target: str                    # Vehicle (what it's compared to): "rose"
    metaphor_type: MetaphorType    # SIMILE, DIRECT, IMPLIED, POSSESSIVE, etc.
    quality_score: float           # 0-1, overall quality
    grounds: List[str]             # Connecting attributes: ["beauty", "thorns"]
    source_text: Optional[str]     # Gutenberg origin if applicable
```

**Metaphor Types**: SIMILE, DIRECT, IMPLIED, POSSESSIVE, APPOSITIVE, COMPOUND, EXTENDED, CONCEPTUAL, SYNESTHETIC

#### 2. LineSeedGenerator → LineSeed dataclass
```python
@dataclass
class LineSeed:
    text: str                        # "The wind remembers..."
    seed_type: SeedType              # OPENING, PIVOT, IMAGE, EMOTIONAL, SONIC, CLOSING, FRAGMENT
    strategy: GenerationStrategy     # JUXTAPOSITION, SYNESTHESIA, INCOMPLETE_METAPHOR, etc.
    momentum: float                  # 0-1, forward movement
    openness: float                  # 0-1, directional possibilities
    quality_score: float             # 0-1, universal quality
    notes: Optional[str]             # Generation notes
    style_fit_score: float           # 0-1, style matching (personalized)
    style_components: Optional[dict] # {line_length, pos, concreteness, phonetic}
```

**Seed Types**: OPENING, PIVOT, IMAGE, EMOTIONAL, SONIC, CLOSING, FRAGMENT
**Strategies**: JUXTAPOSITION, SYNESTHESIA, INCOMPLETE_METAPHOR, RHYTHMIC_BREAK, QUESTION_IMPLIED, TEMPORAL_SHIFT, PERSPECTIVE_BLUR

#### 3. PersonalizedLineSeedGenerator
- Extends LineSeedGenerator with StyleFingerprint
- **Hybrid vocabulary**: Blends author's words with generic vocab (strictness parameter 0.0-1.0)
- **Style dimensions**: line_length, POS patterns, concreteness, phonetic patterns
- **Dual scoring**: 70% quality + 30% style_fit

### Key Scaffolding Dimensions Identified

1. **Semantic Themes/Domains**
   - MetaphorGenerator uses `vocabulary.concept_domains`
   - Cross-domain connections (nature → technology, emotion → weather)

2. **Metaphor Patterns**
   - Simile: "X is like Y"
   - Direct: "X is Y"
   - Possessive: "Y's X" or "X of Y"
   - Implied: verb associations
   - 9 total metaphor types

3. **POS Structure**
   - Line seeds use grammatical templates
   - PersonalizedLineSeedGenerator tracks POS sequence patterns
   - Example: ['DET', 'NOUN', 'VERB', 'ADV']

4. **Rhythmic/Syllabic Patterns**
   - Syllable counts per line
   - Style fingerprint tracks typical line lengths
   - Momentum and openness scores

5. **Emotional Tone/Register**
   - quality_scorer has EmotionalTone (DARK, LIGHT, NEUTRAL, MIXED)
   - quality_scorer has FormalityLevel (ARCHAIC, FORMAL, CONVERSATIONAL, CASUAL)
   - Concreteness ratios (abstract vs concrete)

### Design Goals for PoemTemplate

**Input**: An existing poem (user-provided or generated)
**Output**: A template that can regenerate poems with similar structure

**Template should capture**:
1. Line-by-line POS patterns
2. Syllable/rhythm structure
3. Metaphor types and positions
4. Semantic domain mappings
5. Emotional tone and register
6. Line types (opening, pivot, image, closing)
7. Quality thresholds

**Use cases**:
- Extract template from favorite haiku → generate 10 more with same structure
- Extract template from Shakespeare sonnet → create variations
- Blend templates from multiple sources
- Save/load templates for reuse

### Next Steps (Phase 2.2)
Create `PoemTemplate` dataclass with:
- Metadata: title, source, author
- Structure: lines, syllable_pattern, pos_patterns
- Semantics: domains, metaphor_types, emotional_tone
- Quality: min_quality_score, style_components
- Methods: to_dict(), from_dict(), validate()
