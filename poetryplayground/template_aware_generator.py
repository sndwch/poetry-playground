"""Template-aware poem generation.

This module orchestrates the generation of poems using PoemTemplate structures.
It combines metaphor generation, line seed generation, and quality filtering to
create poems that match specific structural and stylistic templates.

Example:
    >>> from poetryplayground.template_library import get_template_library
    >>> from poetryplayground.template_aware_generator import TemplateAwareGenerator
    >>>
    >>> library = get_template_library()
    >>> template = library.get_template("haiku_nature")
    >>>
    >>> generator = TemplateAwareGenerator()
    >>> result = generator.generate_from_template(template)
    >>> print(result.poem)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from poetryplayground.core.quality_scorer import GenerationContext
from poetryplayground.corpus_analyzer import PersonalCorpusAnalyzer, StyleFingerprint
from poetryplayground.line_seeds import LineSeedGenerator
from poetryplayground.logger import logger
from poetryplayground.metaphor_generator import MetaphorGenerator
from poetryplayground.personalized_seeds import PersonalizedLineSeedGenerator
from poetryplayground.poem_template import LineTemplate, LineType, PoemTemplate


@dataclass
class GeneratedLine:
    """A single generated line with metadata.

    Attributes:
        text: The generated line text
        line_number: Zero-based line number
        template: The LineTemplate used for generation
        source_method: Which method generated this line (metaphor, seed, fallback)
        quality_score: Overall quality score
        style_score: Style fit score (if applicable)
    """

    text: str
    line_number: int
    template: LineTemplate
    source_method: str
    quality_score: float
    style_score: Optional[float] = None


@dataclass
class GenerationResult:
    """Complete result of template-based generation.

    Attributes:
        poem: The complete poem text (lines joined by newlines)
        lines: Individual generated lines with metadata
        template: The PoemTemplate used for generation
        success: Whether generation was successful
        error_message: Error message if generation failed
        metadata: Additional generation metadata
    """

    poem: str
    lines: List[GeneratedLine] = field(default_factory=list)
    template: Optional[PoemTemplate] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)


class TemplateAwareGenerator:
    """Generate poems using structural templates.

    This orchestrator combines:
    - MetaphorGenerator for metaphor-rich lines
    - LineSeedGenerator (or PersonalizedLineSeedGenerator) for opening/closing/pivot lines
    - Quality and style filtering based on template constraints

    Attributes:
        metaphor_generator: Generator for metaphors
        seed_generator: Generator for line seeds (standard or personalized)
        fingerprint: Optional StyleFingerprint for personalized generation
        is_personalized: Whether using personalized seed generation
    """

    def __init__(
        self,
        user_poems_dir: Optional[Path] = None,
        fingerprint: Optional[StyleFingerprint] = None,
        strictness: float = 0.7,
    ):
        """Initialize the template-aware generator.

        Args:
            user_poems_dir: Directory containing user's poems for style matching.
                           If provided, will analyze to create StyleFingerprint.
            fingerprint: Pre-computed StyleFingerprint (overrides user_poems_dir).
            strictness: 0.0-1.0, vocabulary bias for personalized generation
                       (0=generic, 1=only fingerprint). Only used if fingerprint provided.
        """
        # Initialize metaphor generator
        self.metaphor_generator = MetaphorGenerator()

        # Initialize seed generator (personalized or standard)
        self.fingerprint = fingerprint
        self.is_personalized = False

        if fingerprint:
            # Use provided fingerprint
            self.seed_generator = PersonalizedLineSeedGenerator(
                fingerprint=fingerprint,
                strictness=strictness,
            )
            self.is_personalized = True
            logger.info("TemplateAwareGenerator initialized with personalized seed generation")
        elif user_poems_dir:
            # Analyze user poems to create fingerprint
            analyzer = PersonalCorpusAnalyzer()
            self.fingerprint = analyzer.analyze_directory(user_poems_dir)
            self.seed_generator = PersonalizedLineSeedGenerator(
                fingerprint=self.fingerprint,
                strictness=strictness,
            )
            self.is_personalized = True
            logger.info(
                f"TemplateAwareGenerator initialized with personalized seed generation from {user_poems_dir}"
            )
        else:
            # Use standard seed generator
            self.seed_generator = LineSeedGenerator()
            logger.info("TemplateAwareGenerator initialized with standard seed generation")

    def generate_from_template(
        self,
        template: PoemTemplate,
        count: int = 1,
        allow_fallback: bool = True,
    ) -> List[GenerationResult]:
        """Generate poems matching a template.

        Args:
            template: The PoemTemplate to use for generation
            count: Number of poems to generate (default: 1)
            allow_fallback: Allow fallback generation if primary method fails

        Returns:
            List of GenerationResult objects

        Example:
            >>> generator = TemplateAwareGenerator()
            >>> template = library.get_template("haiku_nature")
            >>> results = generator.generate_from_template(template, count=3)
        """
        results = []

        for i in range(count):
            try:
                result = self._generate_single_poem(template, allow_fallback)
                results.append(result)
                logger.debug(f"Generated poem {i + 1}/{count} from template '{template.title}'")
            except Exception as e:
                logger.error(f"Error generating poem {i + 1}: {e}")
                results.append(
                    GenerationResult(
                        poem="",
                        template=template,
                        success=False,
                        error_message=str(e),
                    )
                )

        return results

    def _generate_single_poem(
        self,
        template: PoemTemplate,
        allow_fallback: bool,
    ) -> GenerationResult:
        """Generate a single poem from template.

        Args:
            template: The PoemTemplate to use
            allow_fallback: Allow fallback generation if primary method fails

        Returns:
            GenerationResult for this poem
        """
        generated_lines = []

        # Generate each line according to its template
        for i, line_template in enumerate(template.line_templates):
            generated_line = self._generate_line(
                line_template,
                line_number=i,
                template=template,
                allow_fallback=allow_fallback,
            )
            generated_lines.append(generated_line)

        # Assemble poem
        poem_text = "\n".join(line.text for line in generated_lines)

        # Calculate overall quality
        avg_quality = sum(line.quality_score for line in generated_lines) / len(generated_lines)
        avg_style = None
        if all(line.style_score is not None for line in generated_lines):
            avg_style = sum(line.style_score for line in generated_lines) / len(generated_lines)

        # Create result
        result = GenerationResult(
            poem=poem_text,
            lines=generated_lines,
            template=template,
            success=True,
            metadata={
                "average_quality": avg_quality,
                "average_style_fit": avg_style,
                "template_title": template.title,
                "template_source": template.source,
            },
        )

        logger.info(f"Generated poem from template '{template.title}': {avg_quality:.2f} quality")
        return result

    def _generate_line(
        self,
        line_template: LineTemplate,
        line_number: int,
        template: PoemTemplate,
        allow_fallback: bool,
    ) -> GeneratedLine:
        """Generate a single line from its template.

        Args:
            line_template: The LineTemplate for this line
            line_number: Zero-based line number
            template: The overall PoemTemplate (for context)
            allow_fallback: Allow fallback generation

        Returns:
            GeneratedLine
        """
        # Try metaphor generation if template has metaphor type
        if line_template.metaphor_type:
            line = self._try_metaphor_generation(line_template, template)
            if line:
                return GeneratedLine(
                    text=line["text"],
                    line_number=line_number,
                    template=line_template,
                    source_method="metaphor",
                    quality_score=line["quality"],
                    style_score=line.get("style_fit"),
                )

        # Try line seed generation
        line = self._try_seed_generation(line_template, template)
        if line:
            return GeneratedLine(
                text=line["text"],
                line_number=line_number,
                template=line_template,
                source_method="seed",
                quality_score=line["quality"],
                style_score=line.get("style_fit"),
            )

        # Fallback if allowed
        if allow_fallback:
            line = self._fallback_generation(line_template)
            return GeneratedLine(
                text=line["text"],
                line_number=line_number,
                template=line_template,
                source_method="fallback",
                quality_score=line["quality"],
                style_score=None,
            )

        # If no fallback allowed, raise error
        raise ValueError(f"Failed to generate line {line_number} with template constraints")

    def _try_metaphor_generation(
        self,
        line_template: LineTemplate,
        template: PoemTemplate,
    ) -> Optional[Dict]:
        """Try to generate a line using MetaphorGenerator.

        Args:
            line_template: The LineTemplate for this line
            template: The overall PoemTemplate

        Returns:
            Dict with text, quality, style_fit, or None if generation failed
        """
        try:
            # Use source words from semantic domains if available
            source_words = None
            if template.semantic_domains:
                from poetryplayground.core.vocabulary import vocabulary

                # Get words from primary domain
                domain = template.semantic_domains[0]
                if domain in vocabulary.concept_domains:
                    source_words = vocabulary.concept_domains[domain][:20]

            # Generate metaphors matching the template
            metaphors = self.metaphor_generator.generate_metaphor_from_template(
                line_template=line_template,
                source_words=source_words,
                count=5,
            )

            if not metaphors:
                return None

            # Return the best metaphor
            best = metaphors[0]
            return {
                "text": best.metaphor,
                "quality": best.quality,
                "style_fit": None,  # Metaphors don't have style scores
            }

        except Exception as e:
            logger.debug(f"Metaphor generation failed: {e}")
            return None

    def _try_seed_generation(
        self,
        line_template: LineTemplate,
        template: PoemTemplate,
    ) -> Optional[Dict]:
        """Try to generate a line using seed generator (standard or personalized).

        Args:
            line_template: The LineTemplate for this line
            template: The overall PoemTemplate

        Returns:
            Dict with text, quality, style_fit, or None if generation failed
        """
        try:
            # Try template-aware generation if available
            if hasattr(self.seed_generator, "generate_line_seed_from_template"):
                # PersonalizedLineSeedGenerator has this method
                seeds = self.seed_generator.generate_line_seed_from_template(
                    line_template=line_template,
                    count=5,
                )
            else:
                # LineSeedGenerator - use basic generation
                # Map LineType to SeedType
                from poetryplayground.line_seeds import SeedType

                seed_type_map = {
                    LineType.OPENING: SeedType.OPENING,
                    LineType.CLOSING: SeedType.CLOSING,
                    LineType.PIVOT: SeedType.PIVOT,
                    LineType.TRANSITION: SeedType.TRANSITION,
                    LineType.EMOTIONAL: SeedType.EMOTIONAL,
                    LineType.IMAGE: SeedType.IMAGE,
                    LineType.SONIC: SeedType.SONIC,
                }
                seed_type = seed_type_map.get(line_template.line_type, SeedType.IMAGE)
                seeds = self.seed_generator.generate_seeds(
                    seed_type=seed_type,
                    count=5,
                )

            if not seeds:
                return None

            # Return the best seed
            best = seeds[0]
            return {
                "text": best.text,
                "quality": best.quality_score,
                "style_fit": getattr(best, "style_fit_score", None),
            }

        except Exception as e:
            logger.debug(f"Seed generation failed: {e}")
            return None

    def _fallback_generation(
        self,
        line_template: LineTemplate,
    ) -> Dict:
        """Generate a fallback line using robust word generation tools.

        This creates a line as a last resort when both metaphor and seed
        generation fail. Uses the grammatical template system with quality
        scoring to generate a coherent line matching the POS pattern.

        Args:
            line_template: The LineTemplate for this line

        Returns:
            Dict with text and quality
        """
        from poetryplayground.core.quality_scorer import get_quality_scorer
        from poetryplayground.grammatical_templates import (
            TemplateLibrary as GrammaticalTemplateLibrary,
        )

        try:
            # Use the grammatical template system
            template_lib = GrammaticalTemplateLibrary()
            scorer = get_quality_scorer()
            context = GenerationContext()

            # Try to generate a line matching the syllable count
            best_line = None
            best_quality = 0.0

            # Try multiple attempts to find a good line
            for _ in range(5):
                try:
                    # Generate using the template library
                    result = template_lib.generate_line(
                        syllable_count=line_template.syllable_count,
                        pos_pattern=line_template.pos_pattern
                        if line_template.pos_pattern
                        else None,
                    )

                    if result:
                        # Score the line
                        quality = scorer.score_word(result, context).overall

                        if quality > best_quality:
                            best_quality = quality
                            best_line = result

                except Exception:
                    continue

            if best_line:
                return {
                    "text": best_line,
                    "quality": best_quality,
                }

        except Exception as e:
            logger.debug(f"Fallback generation error: {e}")

        # Ultimate fallback: simple placeholder
        return {
            "text": "a simple line",
            "quality": 0.3,
        }

    def generate_variations(
        self,
        template: PoemTemplate,
        count: int = 5,
        vary_domains: bool = True,
        vary_tone: bool = True,
    ) -> List[GenerationResult]:
        """Generate variations of a template.

        Creates multiple poems from the same structural template while
        varying semantic domains and emotional tone.

        Args:
            template: The PoemTemplate to vary
            count: Number of variations to generate
            vary_domains: Whether to vary semantic domains
            vary_tone: Whether to vary emotional tone

        Returns:
            List of GenerationResult objects
        """
        results = []

        for _i in range(count):
            # Create template variation
            varied_template = self._vary_template(
                template,
                vary_domains=vary_domains,
                vary_tone=vary_tone,
            )

            # Generate poem from varied template
            result = self._generate_single_poem(varied_template, allow_fallback=True)
            results.append(result)

        return results

    def _vary_template(
        self,
        template: PoemTemplate,
        vary_domains: bool,
        vary_tone: bool,
    ) -> PoemTemplate:
        """Create a variation of a template.

        Args:
            template: The original template
            vary_domains: Whether to vary semantic domains
            vary_tone: Whether to vary emotional tone

        Returns:
            Modified PoemTemplate
        """
        import random

        from poetryplayground.core.quality_scorer import EmotionalTone
        from poetryplayground.core.vocabulary import vocabulary

        # Copy the template
        varied = PoemTemplate(
            title=f"{template.title} (variation)",
            source=template.source,
            author=template.author,
            lines=template.lines,
            line_templates=template.line_templates.copy(),
            syllable_pattern=template.syllable_pattern.copy(),
            semantic_domains=template.semantic_domains.copy(),
            metaphor_types=template.metaphor_types.copy(),
            emotional_tone=template.emotional_tone,
            formality_level=template.formality_level,
            concreteness_ratio=template.concreteness_ratio,
            min_quality_score=template.min_quality_score,
            style_components=template.style_components,
            notes=template.notes,
        )

        # Vary semantic domains
        if vary_domains and vocabulary.concept_domains:
            available_domains = list(vocabulary.concept_domains.keys())
            num_domains = len(template.semantic_domains)
            varied.semantic_domains = random.sample(
                available_domains,
                min(num_domains, len(available_domains)),
            )

        # Vary emotional tone
        if vary_tone:
            tones = list(EmotionalTone)
            varied.emotional_tone = random.choice(tones)

        return varied
