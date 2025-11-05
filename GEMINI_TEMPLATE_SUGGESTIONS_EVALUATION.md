# Gemini Template Suggestions - Evaluation & Implementation Plan

## Overview
Gemini provided excellent suggestions for expanding the use of our new grammatical template system beyond just haiku/tanka/senryu generation. This document evaluates each suggestion and provides implementation guidance.

## Current Template System Status ✅

**COMPLETED (Phases 1-3):**
- ✅ Phase 1: POS-Tagged Word Bank (`pos_vocabulary.py`)
- ✅ Phase 2: Template System (`grammatical_templates.py`)
- ✅ Phase 3: Integration with Forms Generator (`forms.py`)

**Result**: Haiku/tanka/senryu now generate grammatically coherent output like:
```
wings entertaining
the teenage angle revealed
billed another drops
```

Instead of syllable soup like:
```
tomorrow did had going
bright do good tomorrow day
had going bright
```

---

## Gemini's Suggestions - Evaluation

### 1. 🌱 Line Seeds Generator ⭐⭐⭐⭐⭐
**Priority: HIGHEST** | **Impact: VERY HIGH** | **Effort: MEDIUM**

#### Current State
- **File**: `generativepoetry/line_seeds.py` (523 lines)
- **Implementation**: Uses string-based pattern filling like:
  ```python
  self.opening_patterns = [
      "The {noun} {verb} {adjective} before...",
      "In {adjective} {noun}, {pronoun} {verb}...",
  ]
  ```
- **Word Selection**: Randomly chooses words from generic lists
  ```python
  replacements = {
      "{verb}": lambda: random.choice(["carries", "holds", "breaks", "turns", "waits"]),
      "{adjective}": lambda: random.choice(["distant", "quiet", "sharp", "soft", "strange"]),
  }
  ```

#### The Problem
Gemini is **100% correct** - the current seeds are not inspiring:
- ❌ "bright tomorrow going" - just random words, no grammar
- ❌ Generic replacements don't create actual **grammatical fragments**
- ❌ No syllable awareness
- ❌ No POS validation

#### Gemini's Recommendation
Use our new POS templates to generate **actual grammatical fragments**:
```python
# Instead of string patterns:
"The {noun} {verb}..."

# Use POS templates:
GrammaticalTemplate(
    pattern=["ADJ", "NOUN"],
    target_syllables=3,
    name="seed-adjective-noun"
)
# Result: "lonely commuter", "ancient temple", "morning silence"
```

#### Implementation Plan

**New Method**: `_generate_template_based_fragment()`
```python
def _generate_template_based_fragment(
    self,
    seed_words: List[str],
    target_syllables: int = None
) -> str:
    """Generate grammatical fragment using POS templates.

    Args:
        seed_words: Words to guide generation
        target_syllables: Optional syllable constraint (2-5 for fragments)

    Returns:
        Grammatically coherent fragment like "lonely commuter" or "wind whispers"
    """
    # Lazy import to avoid circular dependency
    from .grammatical_templates import TemplateGenerator
    from .pos_vocabulary import POSVocabulary

    # Use templates for 2-5 syllable fragments
    if target_syllables is None:
        target_syllables = random.choice([2, 3, 4, 5])

    # Generate using template system
    pos_vocab = POSVocabulary()
    template_gen = TemplateGenerator(pos_vocab)

    line, template = template_gen.generate_line(target_syllables)
    return line if line else self._fallback_fragment(seed_words)
```

**Benefits**:
- ✅ Actual grammatical phrases, not word soup
- ✅ Syllable-aware (useful for constraining seed length)
- ✅ Reuses our POS vocabulary and templates
- ✅ Still incorporates seed_words for thematic guidance

**Estimated Effort**: 1-2 days
- Modify `line_seeds.py` to add template-based generation
- Update `generate_fragment()` and `generate_image_seed()` methods
- Add tests to verify grammatical coherence
- Maintain backward compatibility with pattern-based approach

---

### 2. 🚢 Ship of Theseus Transformer ⭐⭐⭐⭐
**Priority: HIGH** | **Impact: HIGH** | **Effort: MEDIUM-HIGH**

#### Current State
- **Roadmap mentions**: "Ship of Theseus Transformer - Gradually transforms existing poems"
- **Actual file**: **DOES NOT EXIST** ❌
- **Status**: Appears to be planned but not implemented

#### Gemini's Recommendation
Use POS tagging to ensure word replacements maintain grammatical structure:

```python
# Original line
"My heart aches"  # NOUN VERB

# Current random replacement (BROKEN):
"My tree runs"    # NOUN VERB - OK
"My heart slow"   # NOUN ADJ - BROKEN!

# POS-constrained replacement (CORRECT):
"My soul trembles"  # NOUN VERB - maintains grammar
```

#### Implementation Plan

**New File**: `generativepoetry/ship_of_theseus.py`

```python
class ShipOfTheseus Transformer:
    """Gradually transform poems while maintaining grammatical structure."""

    def __init__(self):
        self.pos_vocab = POSVocabulary()
        # POS tagger for input text
        import nltk
        self.pos_tagger = nltk.pos_tag

    def transform_line(
        self,
        line: str,
        replacement_ratio: float = 0.3,
        preserve_pos: bool = True,
        preserve_syllables: bool = True
    ) -> str:
        """Transform a line by replacing words with POS constraints.

        Args:
            line: Original line of poetry
            replacement_ratio: Fraction of words to replace (0.0-1.0)
            preserve_pos: Maintain part-of-speech when replacing
            preserve_syllables: Keep same syllable count

        Returns:
            Transformed line with partial replacements
        """
        words = line.split()

        # Tag the original line
        tagged = self.pos_tagger(words)

        # Select words to replace
        num_to_replace = int(len(words) * replacement_ratio)
        indices = random.sample(range(len(words)), num_to_replace)

        # Replace each selected word
        new_words = words.copy()
        for idx in indices:
            word, penn_tag = tagged[idx]
            universal_pos = PENN_TO_UNIVERSAL.get(penn_tag)

            if not universal_pos:
                continue  # Skip if no mapping

            # Get replacement candidates
            if preserve_syllables:
                syllables = count_syllables(word)
                candidates = self.pos_vocab.get_words(universal_pos, syllables)
            else:
                # Get any words of this POS
                candidates = []
                for syl_count in range(1, 6):
                    candidates.extend(self.pos_vocab.get_words(universal_pos, syl_count))

            if candidates:
                new_words[idx] = random.choice(candidates)

        return " ".join(new_words)

    def gradual_transform(
        self,
        original: str,
        steps: int = 5
    ) -> List[str]:
        """Perform gradual transformation over multiple steps.

        Returns list showing progression from original to fully transformed.
        """
        transformations = [original]
        current = original

        for step in range(steps):
            # Increase replacement ratio gradually
            ratio = (step + 1) / steps
            current = self.transform_line(current, replacement_ratio=ratio)
            transformations.append(current)

        return transformations
```

**Benefits**:
- ✅ Maintains grammatical structure during transformation
- ✅ Optional syllable preservation (useful for metrical poetry)
- ✅ Gradual transformation shows "Ship of Theseus" thought experiment
- ✅ Can be used for generating variations of successful poems

**Estimated Effort**: 2-3 days
- Create `ship_of_theseus.py` module
- Implement POS-constrained replacement logic
- Add CLI integration
- Write comprehensive tests
- Add documentation and examples

---

### 3. 🎨 Grammatical Concrete Poems ⭐⭐⭐
**Priority: MEDIUM** | **Impact: MEDIUM** | **Effort: LOW-MEDIUM**

#### Current State
- **File**: No concrete poetry generator found
- **Roadmap**: Mentions "Visual Poetry Generators" but no specific concrete poetry tool

#### Gemini's Recommendation
Create visual poems using grammatical fragments (not random words):

```python
# Current chaotic concrete (if it existed):
Random placement: "had", "going", "bright", "tomorrow"

# Grammatical concrete:
Clustered phrases: "lonely commuter", "wind whispers", "ancient temple"
```

#### Implementation Plan

**New File**: `generativepoetry/grammatical_concrete.py`

```python
def generate_grammatical_concrete_poem(
    seed_words: List[str],
    num_fragments: int = 12,
    canvas_size: Tuple[int, int] = (80, 40)
) -> str:
    """Generate concrete poem using grammatical fragments.

    Creates visual arrangement of 2-3 word grammatical fragments.
    """
    # Generate grammatical fragments (2-3 syllables each)
    template_gen = TemplateGenerator(POSVocabulary())
    fragments = []

    for _ in range(num_fragments):
        syllables = random.choice([2, 3])
        line, _ = template_gen.generate_line(syllables)
        if line:
            fragments.append(line)

    # Arrange fragments spatially (concrete poetry layout logic)
    # ... positioning algorithm ...

    return visual_layout
```

**Benefits**:
- ✅ Fragments are locally meaningful ("old gate" vs "had going")
- ✅ Maintains coherence even in chaotic visual arrangement
- ✅ Reuses template system

**Estimated Effort**: 1-2 days
- Simple implementation (reuse templates + add positioning logic)
- Mostly a creative application of existing infrastructure

---

### 4. 🔮 Metaphor Generator Enhancement ⭐⭐
**Priority: LOW** | **Impact: LOW** | **Effort: LOW**

#### Current State
- **File**: `generativepoetry/metaphor_generator.py` (624 lines)
- **Already has extensive template patterns**:
  ```python
  self.simile_patterns = [
      "{source} is like {target}",
      "{source} like {target}",
      "{source}, like {target},",
      "as {adjective} as {target}",
  ]

  self.direct_patterns = [
      "{source} is {target}",
      "{source}: {target}",
      "the {target} of {source}",
  ]

  self.possessive_patterns = [
      "{target}'s {source}",
      "the {source} of {target}",
  ]

  self.appositive_patterns = [
      "{source}, that {target}",
      "{source}, the {target}",
  ]
  ```

#### Gemini's Recommendation
Expand with more template variations like "X as Y", "X, a [ADJ] Y", etc.

#### Evaluation
**Already well-implemented** ✅
- Current system has 4 major pattern types with multiple variations each
- Total of ~15 different metaphor pattern templates
- Additional patterns would be incremental improvement, not transformative

#### Recommendation
**Low priority** - Current implementation is sufficient. Focus on higher-impact suggestions (#1 and #2).

If implementing later, simply add to existing pattern lists:
```python
self.simile_patterns.extend([
    "{source} as {adjective} {target}",
    "{source}, a {adjective} {target}",
])
```

---

## Implementation Roadmap

### Phase 4: Template Expansion (NEW) - 1-2 weeks

#### 4.1 Template-Based Line Seeds ⚡ HIGHEST PRIORITY
**Estimated Effort: 1-2 days**

- [ ] Add `_generate_template_based_fragment()` method to `line_seeds.py`
- [ ] Update `generate_fragment()` to use templates by default
- [ ] Update `generate_image_seed()` for grammatical seed images
- [ ] Add `use_templates` parameter for backward compatibility
- [ ] Write tests comparing template-based vs pattern-based seeds
- [ ] Update CLI to showcase improved seed quality

**Expected Improvement**:
```
Before: "bright tomorrow going"
After:  "lonely commuter", "wind whispers", "ancient temple"
```

#### 4.2 Ship of Theseus Transformer 🚢 HIGH PRIORITY
**Estimated Effort: 2-3 days**

- [ ] Create `generativepoetry/ship_of_theseus.py`
- [ ] Implement `ShipOfTheseusTransformer` class
- [ ] Add `transform_line()` with POS preservation
- [ ] Add `gradual_transform()` for multi-step transformation
- [ ] Integrate with CLI as new action
- [ ] Write comprehensive tests
- [ ] Add documentation with examples

**Use Cases**:
- Generating variations of successful poems
- Exploring alternative phrasings while maintaining structure
- Teaching tool for understanding POS roles

#### 4.3 Grammatical Concrete Poetry 🎨 MEDIUM PRIORITY
**Estimated Effort: 1-2 days**

- [ ] Create `generativepoetry/grammatical_concrete.py`
- [ ] Implement fragment generation using templates
- [ ] Add spatial positioning algorithm
- [ ] Integrate with visual poetry CLI
- [ ] Write tests
- [ ] Add examples

**Expected Improvement**:
```
Before: Random words scattered: "had", "going", "tomorrow"
After:  Grammatical clusters: "lonely commuter", "wind whispers"
```

#### 4.4 Metaphor Template Expansion (OPTIONAL)
**Estimated Effort: 1 hour**

- [ ] Add 3-5 new metaphor pattern templates
- [ ] Update tests
- [ ] Document new patterns

**Low priority - current implementation is sufficient**

---

## Summary

### Gemini's Evaluation Quality: ⭐⭐⭐⭐⭐ EXCELLENT

All four suggestions are:
1. **Technically sound** - Leverage existing POS infrastructure correctly
2. **High impact** - Meaningfully improve output quality
3. **Well-explained** - Clear problem statements and examples
4. **Practical** - Implementable with reasonable effort

### Recommended Implementation Order

1. **Week 1**: Template-Based Line Seeds (#1) - Highest impact, moderate effort
2. **Week 2**: Ship of Theseus Transformer (#2) - High impact, fills gap in roadmap
3. **Week 3**: Grammatical Concrete Poetry (#3) - Creative application
4. **Optional**: Metaphor expansion (#4) - Already well-implemented

### Integration with Existing Roadmap

These suggestions fit naturally into the existing Phase 4-6 structure:
- Phase 4: Template Expansion (NEW - incorporates Gemini's suggestions)
- Phase 5: CLI Integration (existing plan)
- Phase 6: Testing & Documentation (existing plan)

---

## Technical Notes

### Avoiding Circular Dependencies

When integrating templates into `line_seeds.py` and other modules, use lazy imports:

```python
# At module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grammatical_templates import TemplateGenerator

# In methods
def _generate_template_based_fragment(self, ...):
    # Import at runtime to avoid circular dependency
    from .grammatical_templates import TemplateGenerator
    from .pos_vocabulary import POSVocabulary

    template_gen = TemplateGenerator(POSVocabulary())
    # ...
```

### Performance Considerations

- POS Vocabulary loading is cached (~1s after first build)
- Template generation is fast (milliseconds per line)
- No additional external API calls needed
- Suitable for interactive CLI use

### Backward Compatibility

All new features should:
- ✅ Maintain existing API
- ✅ Add `use_templates` parameter for toggling
- ✅ Fallback to legacy behavior if template generation fails
- ✅ No breaking changes

---

## Conclusion

Gemini's suggestions are **exceptionally valuable** and should be prioritized for implementation. They:

1. ✅ Leverage our new template infrastructure across the entire codebase
2. ✅ Address real quality issues in existing features
3. ✅ Fill gaps in planned features (Ship of Theseus)
4. ✅ Provide clear implementation guidance

**Recommendation**: Implement suggestions #1 and #2 immediately (1-2 weeks), then #3 as time permits.

This will maximize the return on investment from the grammatical template system we just built (Phases 1-3).
