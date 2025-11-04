"""
Poem Transformer - Ship of Theseus Style

Takes a poem and gradually transforms it through multiple passes using Datamuse API,
creating variations that drift from the original while attempting to maintain
some semantic coherence. Like the old Google Translate telephone game.
"""

import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from datamuse import datamuse

from .lexigen import contextually_linked_words, similar_meaning_words, similar_sounding_words
from .word_validator import WordValidator


@dataclass
class TransformationStep:
    """Records a single word transformation"""

    original_word: str
    new_word: str
    transformation_type: str  # 'semantic', 'contextual', 'sonic'
    line_number: int
    confidence: float = 0.0  # How good we think this replacement is


@dataclass
class PoemTransformation:
    """Complete record of a poem transformation"""

    original_poem: str
    transformed_poem: str
    steps: List[TransformationStep] = field(default_factory=list)
    pass_number: int = 1
    coherence_score: float = 0.0


class PoemTransformer:
    """Transforms poems through iterative Datamuse API replacements"""

    def __init__(self):
        self.datamuse_api = datamuse.Datamuse()
        self.word_validator = WordValidator()

        # Words to avoid transforming (preserve poem structure)
        self.preserve_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "where",
            "when",
            "why",
            "how",
            "not",
            "no",
            "yes",
            "so",
            "if",
            "then",
            "else",
            "than",
            "as",
            "like",
        }

    def transform_poem_iteratively(
        self, poem_text: str, num_passes: int = 5, words_per_pass: int = 8
    ) -> List[PoemTransformation]:
        """Transform a poem through multiple passes, like telephone game"""
        transformations = []
        current_poem = poem_text.strip()

        print(f"Starting Ship of Theseus transformation with {num_passes} passes...")

        for pass_num in range(1, num_passes + 1):
            print(f"\nPass {pass_num}/{num_passes}...")

            transformation = self._single_transformation_pass(
                current_poem, pass_num, words_per_pass
            )

            if transformation:
                transformations.append(transformation)
                current_poem = transformation.transformed_poem

                # Show progress
                changed_words = len(transformation.steps)
                print(f"  Changed {changed_words} words in this pass")

                # Brief pause to avoid API rate limits
                time.sleep(0.5)
            else:
                print(f"  No changes made in pass {pass_num}")
                break

        return transformations

    def _single_transformation_pass(
        self, poem_text: str, pass_number: int, target_changes: int
    ) -> Optional[PoemTransformation]:
        """Perform a single transformation pass on the poem"""
        lines = poem_text.split("\n")
        new_lines = lines.copy()
        transformation_steps = []

        # Get all transformable words from the poem
        transformable_words = self._get_transformable_words(poem_text)

        if not transformable_words:
            return None

        # Randomly select words to transform this pass
        words_to_transform = random.sample(
            transformable_words, min(target_changes, len(transformable_words))
        )

        for word_info in words_to_transform:
            word, line_idx = word_info

            # Choose transformation type (weighted toward maintaining meaning early on)
            if pass_number <= 2:
                # Early passes: prefer semantic similarity
                weights = [0.6, 0.3, 0.1]  # semantic, contextual, sonic
            else:
                # Later passes: more experimental
                weights = [0.4, 0.4, 0.2]

            transformation_type = random.choices(
                ["semantic", "contextual", "sonic"], weights=weights
            )[0]

            # Get replacement word
            replacement = self._get_replacement_word(word, transformation_type)

            if replacement and replacement != word:
                # Apply the replacement to the line
                old_line = new_lines[line_idx]
                new_line = self._replace_word_in_line(old_line, word, replacement)
                new_lines[line_idx] = new_line

                # Record the transformation
                step = TransformationStep(
                    original_word=word,
                    new_word=replacement,
                    transformation_type=transformation_type,
                    line_number=line_idx + 1,
                    confidence=self._calculate_replacement_confidence(
                        word, replacement, transformation_type
                    ),
                )
                transformation_steps.append(step)

        if transformation_steps:
            transformed_poem = "\n".join(new_lines)
            coherence_score = self._calculate_coherence_score(poem_text, transformed_poem)

            return PoemTransformation(
                original_poem=poem_text,
                transformed_poem=transformed_poem,
                steps=transformation_steps,
                pass_number=pass_number,
                coherence_score=coherence_score,
            )

        return None

    def _get_transformable_words(self, poem_text: str) -> List[Tuple[str, int]]:
        """Get list of words that can be transformed, with their line numbers"""
        lines = poem_text.split("\n")
        transformable = []

        for line_idx, line in enumerate(lines):
            # Extract words, preserving their position
            words = re.findall(r"\b[a-zA-Z]+\b", line)

            for word in words:
                word_lower = word.lower()

                # Skip if word should be preserved
                if word_lower in self.preserve_words:
                    continue

                # Skip very short words
                if len(word) < 3:
                    continue

                # Skip if not a valid English word
                if not self.word_validator.is_valid_english_word(word):
                    continue

                transformable.append((word, line_idx))

        return transformable

    def _get_replacement_word(self, word: str, transformation_type: str) -> Optional[str]:
        """Get a replacement word using the specified transformation type"""
        try:
            if transformation_type == "semantic":
                replacements = similar_meaning_words(word, sample_size=5, datamuse_api_max=15)
            elif transformation_type == "contextual":
                replacements = contextually_linked_words(word, sample_size=5, datamuse_api_max=15)
            elif transformation_type == "sonic":
                replacements = similar_sounding_words(word, sample_size=5, datamuse_api_max=15)
            else:
                return None

            # Filter and validate replacements
            valid_replacements = [
                r for r in replacements if r != word and self._is_good_replacement(word, r)
            ]

            if valid_replacements:
                return random.choice(valid_replacements)

        except Exception as e:
            print(f"Error getting replacement for '{word}': {e}")

        return None

    def _is_good_replacement(self, original: str, replacement: str) -> bool:
        """Check if a replacement word is suitable"""
        # Basic validation
        if not replacement or len(replacement) < 2:
            return False

        # Must be alphabetic
        if not replacement.isalpha():
            return False

        # Avoid overly long words
        if len(replacement) > 15:
            return False

        # Avoid words that are too similar (just adding/removing letters)
        if abs(len(original) - len(replacement)) > 4:
            return False

        # Use word validator
        return self.word_validator.is_valid_english_word(replacement)

    def _replace_word_in_line(self, line: str, old_word: str, new_word: str) -> str:
        """Replace a word in a line, preserving capitalization and punctuation"""
        # Handle different capitalization patterns
        patterns = [
            (r"\b" + re.escape(old_word) + r"\b", new_word),  # exact match
            (
                r"\b" + re.escape(old_word.capitalize()) + r"\b",
                new_word.capitalize(),
            ),  # capitalized
            (r"\b" + re.escape(old_word.upper()) + r"\b", new_word.upper()),  # uppercase
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, line):
                return re.sub(pattern, replacement, line, count=1)

        return line

    def _calculate_replacement_confidence(
        self, original: str, replacement: str, transformation_type: str
    ) -> float:
        """Calculate confidence score for a replacement (0-1)"""
        # Base confidence by transformation type
        base_confidence = {"semantic": 0.8, "contextual": 0.6, "sonic": 0.4}

        confidence = base_confidence.get(transformation_type, 0.5)

        # Adjust based on word characteristics
        if len(original) == len(replacement):
            confidence += 0.1  # Same length is often good

        if original[0] == replacement[0]:
            confidence += 0.05  # Same first letter

        return min(1.0, confidence)

    def _calculate_coherence_score(self, original: str, transformed: str) -> float:
        """Calculate how coherent the transformed poem is (very basic heuristic)"""
        # This is a simple heuristic - could be much more sophisticated
        original_words = set(re.findall(r"\b[a-zA-Z]+\b", original.lower()))
        transformed_words = set(re.findall(r"\b[a-zA-Z]+\b", transformed.lower()))

        # Measure word overlap (higher = more coherent)
        if not original_words:
            return 0.0

        overlap = len(original_words & transformed_words) / len(original_words)
        return overlap

    def list_poems_in_directory(self, directory_path: str) -> List[Tuple[str, str]]:
        """List available poems in a directory"""
        directory = Path(directory_path)
        if not directory.exists():
            return []

        poems = []
        for file_path in directory.glob("*.txt"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # Skip empty files
                        poems.append((file_path.stem, str(file_path)))
            except Exception:
                continue

        # Also check for .md files
        for file_path in directory.glob("*.md"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        poems.append((file_path.stem, str(file_path)))
            except Exception:
                continue

        return sorted(poems)

    def load_poem(self, file_path: str) -> str:
        """Load and clean a poem from a file"""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Remove markdown formatting but preserve line breaks
        content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)  # Remove bold
        content = re.sub(r"\*(.*?)\*", r"\1", content)  # Remove italic
        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)  # Remove code blocks
        content = re.sub(r"\\$", "", content, flags=re.MULTILINE)  # Remove line continuation

        # Clean up whitespace
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def generate_transformation_report(self, transformations: List[PoemTransformation]) -> str:
        """Generate a report showing the transformation process"""
        if not transformations:
            return "No transformations performed."

        report = []
        report.append("SHIP OF THESEUS POEM TRANSFORMATION")
        report.append("=" * 60)

        # Show original
        original = transformations[0].original_poem
        final = transformations[-1].transformed_poem

        report.append("\nORIGINAL POEM:")
        report.append("-" * 30)
        report.append(original)

        # Show transformation summary
        report.append("\nTRANSFORMATION SUMMARY:")
        report.append("-" * 30)

        total_changes = sum(len(t.steps) for t in transformations)
        report.append(f"Total passes: {len(transformations)}")
        report.append(f"Total word changes: {total_changes}")

        final_coherence = transformations[-1].coherence_score
        report.append(f"Final coherence score: {final_coherence:.2f}")

        # Show each pass
        for i, transformation in enumerate(transformations, 1):
            report.append(f"\nPASS {i} CHANGES:")
            report.append("-" * 20)

            for step in transformation.steps:
                conf_str = f"({step.confidence:.2f})" if step.confidence > 0 else ""
                report.append(
                    f"  Line {step.line_number}: '{step.original_word}' → '{step.new_word}' ({step.transformation_type}) {conf_str}"
                )

        # Show final result
        report.append("\nFINAL TRANSFORMED POEM:")
        report.append("-" * 30)
        report.append(final)

        # Show word mapping
        all_changes = {}
        for transformation in transformations:
            for step in transformation.steps:
                if step.original_word not in all_changes:
                    all_changes[step.original_word] = []
                all_changes[step.original_word].append(step.new_word)

        if all_changes:
            report.append("\nWORD EVOLUTION CHAINS:")
            report.append("-" * 30)
            for original, replacements in sorted(all_changes.items()):
                chain = " → ".join([original, *replacements])
                report.append(f"  {chain}")

        return "\n".join(report)
