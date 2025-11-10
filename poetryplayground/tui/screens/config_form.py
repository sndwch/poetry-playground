"""Configuration form screen for generation procedures."""

from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static


class ConfigFormScreen(Screen):
    """Configuration and execution screen for a specific generation procedure."""

    CSS = """
    ConfigFormScreen {
        align: center middle;
    }

    #config-container {
        width: 80;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 1 2;
    }

    #title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1 0;
    }

    #description {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #form-content {
        height: auto;
        margin: 1 0;
    }

    .input-label {
        margin-top: 1;
        margin-bottom: 0;
        color: $text;
    }

    Input {
        margin-bottom: 1;
    }

    #button-row {
        align: center middle;
        margin-top: 1;
    }

    #generate-btn {
        margin: 0 1;
    }

    #back-btn {
        margin: 0 1;
    }

    #status {
        text-align: center;
        color: $accent;
        margin-top: 1;
        height: 3;
    }
    """

    # Procedure metadata and configuration
    PROCEDURE_CONFIG: ClassVar[dict] = {
        "haiku": {
            "name": "Haiku Generator",
            "description": "Generate 5-7-5 syllable haiku with grammatical templates",
            "inputs": [
                ("seed_words", "Seed Words (optional, comma-separated)", ""),
            ],
        },
        "tanka": {
            "name": "Tanka Generator",
            "description": "Generate 5-7-5-7-7 syllable tanka",
            "inputs": [
                ("seed_words", "Seed Words (optional, comma-separated)", ""),
            ],
        },
        "senryu": {
            "name": "Senryu Generator",
            "description": "Generate 5-7-5 syllable senryu focused on human nature",
            "inputs": [
                ("seed_words", "Seed Words (optional, comma-separated)", ""),
            ],
        },
        "metaphor": {
            "name": "Metaphor Generator",
            "description": "Generate fresh metaphors from Project Gutenberg texts",
            "inputs": [
                ("count", "Number of metaphors", "10"),
            ],
        },
        "lineseeds": {
            "name": "Line Seeds Generator",
            "description": "Generate evocative incomplete phrases and line beginnings",
            "inputs": [
                ("seed_words", "Seed Words (space or comma-separated)", ""),
                ("count", "Number of seeds", "10"),
            ],
        },
        "personalized_lineseeds": {
            "name": "Personalized Line Seeds",
            "description": "Generate line seeds matching your personal style fingerprint",
            "inputs": [
                ("corpus_dir", "Corpus Directory Path", ""),
                ("strictness", "Strictness (0.0-1.0)", "0.7"),
                ("count", "Number of seeds", "15"),
                ("min_quality", "Min Quality (0.0-1.0)", "0.5"),
                ("min_style_fit", "Min Style Fit (0.0-1.0)", "0.4"),
            ],
        },
        "ideas": {
            "name": "Poetry Idea Generator",
            "description": "Mine creative seeds from classic literature",
            "inputs": [
                ("count", "Number of ideas", "10"),
            ],
        },
        "fragments": {
            "name": "Resonant Fragment Miner",
            "description": "Extract poetic sentence fragments from literature",
            "inputs": [
                ("count", "Number of fragments", "50"),
            ],
        },
        "equidistant": {
            "name": "Equidistant Word Finder",
            "description": "Find words equidistant from two anchor words",
            "inputs": [
                ("word_a", "First anchor word", ""),
                ("word_b", "Second anchor word", ""),
                ("mode", "Mode (orth/phono)", "orth"),
                ("window", "Window (distance tolerance)", "0"),
            ],
        },
        "corpus": {
            "name": "Personal Corpus Analyzer",
            "description": "Analyze your personal poetry collection for style insights",
            "inputs": [
                ("directory", "Poetry directory path", "/Users/jparker/Desktop/free-verse"),
            ],
        },
        "theseus": {
            "name": "Ship of Theseus Transformer",
            "description": "Gradually transform a poem while maintaining structure",
            "inputs": [
                ("poem_text", "Original poem text", ""),
                ("steps", "Number of transformation steps", "5"),
                ("preserve_pos", "Preserve part-of-speech (yes/no)", "yes"),
                ("preserve_syllables", "Preserve syllable counts (yes/no)", "yes"),
            ],
        },
        "sixdegrees": {
            "name": "Six Degrees Word Convergence",
            "description": "Find convergence paths between two words",
            "inputs": [
                ("word_a", "First word", ""),
                ("word_b", "Second word", ""),
            ],
        },
        "futurist": {
            "name": "Futurist Poem",
            "description": "Marinetti-inspired mathematical word connections",
            "inputs": [
                ("input_words", "Input words (comma-separated)", ""),
                ("num_lines", "Number of lines", "25"),
            ],
        },
        "markov": {
            "name": "Stochastic Jolastic (Markov)",
            "description": "Joyce-like wordplay with rhyme schemes",
            "inputs": [
                ("input_words", "Input words (comma-separated)", ""),
                ("num_lines", "Number of lines", "10"),
            ],
        },
        "puzzle": {
            "name": "Visual Puzzle Poem",
            "description": "Word list-based terminal poem",
            "inputs": [
                ("input_words", "Input words (comma-separated)", ""),
            ],
        },
        "chaotic": {
            "name": "Chaotic Concrete Poem",
            "description": "PDF-only visual spatial arrangements",
            "inputs": [],
        },
        "charsoup": {
            "name": "Character Soup Poem",
            "description": "PDF-only typographic experiments",
            "inputs": [],
        },
        "wordsoup": {
            "name": "Stop Word Soup Poem",
            "description": "PDF-only minimalist word placement",
            "inputs": [],
        },
        "semantic_path": {
            "name": "Semantic Geodesic Finder",
            "description": "Find transitional paths through meaning-space",
            "inputs": [
                ("start_word", "Starting word", ""),
                ("end_word", "Ending word", ""),
                ("steps", "Number of steps (min 3)", "5"),
                ("alternatives", "Alternative words per step", "3"),
                ("method", "Path method (linear/bezier/shortest)", "linear"),
            ],
        },
        "conceptual_cloud": {
            "name": "Conceptual Cloud Generator",
            "description": "Multi-dimensional word associations",
            "inputs": [
                ("center_word", "Center word", ""),
                ("k_per_cluster", "Words per cluster", "10"),
                ("sections", "Sections (all or comma-separated)", "all"),
                ("output_format", "Output format (simple/markdown/json/rich)", "simple"),
            ],
        },
        "deps": {
            "name": "Check System Dependencies",
            "description": "Verify spaCy, NLTK, and other dependencies",
            "inputs": [],
        },
    }

    def __init__(self, procedure_id: str):
        """Initialize config form for a specific procedure.

        Args:
            procedure_id: ID of the generation procedure to configure
        """
        super().__init__()
        self.procedure_id = procedure_id
        self.config = self.PROCEDURE_CONFIG.get(
            procedure_id,
            {
                "name": "Unknown Procedure",
                "description": "Configuration not yet implemented",
                "inputs": [],
            },
        )

    def compose(self) -> ComposeResult:
        """Create form widgets."""
        with Container(id="config-container"):
            yield Label(self.config["name"], id="title")
            yield Label(self.config["description"], id="description")

            with Vertical(id="form-content"):
                # Create input fields based on configuration
                for field_id, field_label, default_value in self.config["inputs"]:
                    yield Label(field_label, classes="input-label")
                    yield Input(value=default_value, id=f"input-{field_id}")

            with Horizontal(id="button-row"):
                yield Button("Generate", variant="primary", id="generate-btn")
                yield Button("Back to Menu", variant="default", id="back-btn")

            yield Static("", id="status")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "generate-btn":
            await self._handle_generate()
        elif event.button.id == "back-btn":
            await self._handle_back()

    async def _handle_generate(self) -> None:
        """Handle generate button click."""
        # Extract configuration from input fields
        config_values = {}
        for field_id, _, _ in self.config["inputs"]:
            input_widget = self.query_one(f"#input-{field_id}", Input)
            config_values[field_id] = input_widget.value

        # Update status
        status_widget = self.query_one("#status", Static)
        status_widget.update("🔄 Generating...")

        # Run generation in worker thread
        self.run_generation_worker(self.procedure_id, config_values)

    @work(exclusive=True, thread=True)
    def run_generation_worker(self, procedure_id: str, config_values: dict) -> None:
        """Run generation in background thread.

        Args:
            procedure_id: ID of the procedure to run
            config_values: Configuration values from form
        """
        try:
            # Import generators here to avoid loading everything on startup
            result = self._execute_generator(procedure_id, config_values)

            # Post result back to main thread
            self.app.call_from_thread(self._on_generation_complete, result)

        except Exception as e:
            # Post error back to main thread
            self.app.call_from_thread(self._on_generation_error, str(e))

    def _execute_generator(self, procedure_id: str, config: dict) -> str:
        """Execute the appropriate generator based on procedure ID.

        Args:
            procedure_id: ID of the procedure to run
            config: Configuration dictionary

        Returns:
            Generated text output

        Raises:
            ValueError: If procedure ID is not recognized
        """
        if procedure_id in ["haiku", "tanka", "senryu"]:
            from poetryplayground.forms import FormGenerator

            generator = FormGenerator()
            seed_words = None
            if config.get("seed_words"):
                seed_words = [
                    w.strip() for w in config["seed_words"].replace(",", " ").split() if w.strip()
                ]

            if procedure_id == "haiku":
                lines, validation = generator.generate_haiku(seed_words=seed_words, strict=True)
            elif procedure_id == "tanka":
                lines, validation = generator.generate_tanka(seed_words=seed_words, strict=True)
            else:  # senryu
                lines, validation = generator.generate_senryu(seed_words=seed_words, strict=True)

            result = "\n".join(lines)
            if validation.valid:
                result += f"\n\n✓ Valid {procedure_id} ({validation.syllable_pattern})"
            else:
                result += "\n\n" + validation.get_report()

            return result

        elif procedure_id == "metaphor":
            from poetryplayground.metaphor_display import format_metaphors
            from poetryplayground.metaphor_generator import MetaphorGenerator

            count = int(config.get("count", 10))
            generator = MetaphorGenerator()

            # Extract metaphor patterns from Project Gutenberg texts
            # (Using extract_metaphor_patterns since no source words provided in TUI config)
            num_texts = max(3, count // 3)  # Use more texts for larger counts
            metaphors = generator.extract_metaphor_patterns(num_texts=num_texts, verbose=False)

            # Use compact formatter optimized for TUI (60 chars wide, box drawing)
            if metaphors:
                max_per_cat = max(3, count // 3)  # Show more if they requested more
                return format_metaphors(
                    metaphors, mode="compact", max_per_category=max_per_cat, show_context=False
                )
            else:
                return "No metaphors found. Try again or adjust parameters."

        elif procedure_id == "lineseeds":
            from poetryplayground.line_seeds import LineSeedGenerator

            seed_words_str = config.get("seed_words", "")
            seed_words = (
                [w.strip() for w in seed_words_str.replace(",", " ").split() if w.strip()]
                if seed_words_str
                else None
            )
            count = int(config.get("count", 10))

            generator = LineSeedGenerator()
            seeds = generator.generate_seed_collection(seed_words=seed_words, num_seeds=count)

            # Format the line seeds for display
            result = []
            for i, seed in enumerate(seeds, 1):
                result.append(f"{i}. [{seed.seed_type.value}] {seed.text}")
                result.append(
                    f"   Quality: {seed.quality_score:.2f}, Momentum: {seed.momentum:.2f}"
                )
                if seed.notes:
                    result.append(f"   Note: {seed.notes}")

            return "\n".join(result)

        elif procedure_id == "personalized_lineseeds":
            import os

            from poetryplayground.corpus_analyzer import PersonalCorpusAnalyzer
            from poetryplayground.personalized_seeds import PersonalizedLineSeedGenerator

            corpus_dir = config.get("corpus_dir", "").strip()
            if not corpus_dir or not os.path.exists(corpus_dir):
                return "Error: Invalid or missing corpus directory path"

            try:
                strictness = float(config.get("strictness", 0.7))
                if not 0.0 <= strictness <= 1.0:
                    strictness = 0.7
            except ValueError:
                strictness = 0.7

            try:
                count = int(config.get("count", 15))
            except ValueError:
                count = 15

            try:
                min_quality = float(config.get("min_quality", 0.5))
                if not 0.0 <= min_quality <= 1.0:
                    min_quality = 0.5
            except ValueError:
                min_quality = 0.5

            try:
                min_style_fit = float(config.get("min_style_fit", 0.4))
                if not 0.0 <= min_style_fit <= 1.0:
                    min_style_fit = 0.4
            except ValueError:
                min_style_fit = 0.4

            # Analyze corpus
            analyzer = PersonalCorpusAnalyzer()
            fingerprint = analyzer.analyze_directory(corpus_dir)

            if fingerprint.metrics.total_poems == 0:
                return "Error: No poems found in the specified directory"

            # Generate personalized seeds
            generator = PersonalizedLineSeedGenerator(fingerprint, strictness=strictness)
            seeds = generator.generate_personalized_collection(
                count=count, min_quality=min_quality, min_style_fit=min_style_fit
            )

            if not seeds:
                return "No seeds met the quality/style thresholds. Try lowering min_quality or min_style_fit."

            # Format output
            result = [
                f"Generated {len(seeds)} personalized seeds from {fingerprint.metrics.total_poems} poems\n"
            ]
            for i, seed in enumerate(seeds, 1):
                quality = seed.quality_score
                style_fit = seed.style_fit_score
                combined = 0.7 * quality + 0.3 * style_fit

                result.append(f"{i}. [{seed.seed_type.value}] {seed.text}")
                result.append(
                    f"   Quality: {quality:.2f} | Style: {style_fit:.2f} | Combined: {combined:.2f}"
                )

                # Show style breakdown for top 5
                if i <= 5 and seed.style_components:
                    comp = seed.style_components
                    result.append(
                        f"   Style breakdown: length={comp['line_length']:.2f}, "
                        f"pos={comp['pos_pattern']:.2f}, "
                        f"concrete={comp['concreteness']:.2f}, "
                        f"phonetic={comp['phonetic']:.2f}"
                    )
                result.append("")

            return "\n".join(result)

        elif procedure_id == "ideas":
            from poetryplayground.idea_generator import PoetryIdeaGenerator

            count = int(config.get("count", 10))
            generator = PoetryIdeaGenerator()
            collection = generator.generate_ideas(num_ideas=count)

            # Get mixed selection of ideas across all types
            ideas = collection.get_random_mixed_selection(count=count, prefer_quality=True)

            result_lines = []
            for i, idea in enumerate(ideas, 1):
                result_lines.append(f"{i}. [{idea.idea_type.value}] {idea.text}")
                result_lines.append(f"   Prompt: {idea.creative_prompt}")
                if idea.keywords:
                    result_lines.append(f"   Keywords: {', '.join(idea.keywords[:3])}")
                result_lines.append(f"   Quality: {idea.quality_score:.2f}")
                result_lines.append("")

            return "\n".join(result_lines)

        elif procedure_id == "fragments":
            from poetryplayground.causal_poetry import ResonantFragmentMiner

            count = int(config.get("count", 50))
            miner = ResonantFragmentMiner()
            collection = miner.mine_fragments(target_count=count, num_texts=5)

            result_lines = []
            for i, frag in enumerate(collection.get_all_fragments()[:count], 1):
                result_lines.append(f"{i}. {frag.text}")
                result_lines.append(
                    f"   Pattern: {frag.pattern_type} | Score: {frag.poetic_score:.2f}"
                )
                result_lines.append("")

            return "\n".join(result_lines)

        elif procedure_id == "equidistant":
            from poetryplayground.finders import find_equidistant

            word_a = config.get("word_a", "").strip()
            word_b = config.get("word_b", "").strip()

            if not word_a or not word_b:
                return "Error: Both anchor words are required"

            mode = config.get("mode", "orth")
            if mode not in ["orth", "phono"]:
                mode = "orth"

            try:
                window = int(config.get("window", 0))
            except ValueError:
                window = 0

            hits = find_equidistant(a=word_a, b=word_b, mode=mode, window=window)

            if not hits:
                return f"No equidistant words found for '{word_a}' and '{word_b}'"

            result_lines = [f"Found {len(hits)} equidistant words:\n"]
            for i, hit in enumerate(hits[:20], 1):
                syl_str = f"{hit.syllables}s" if hit.syllables else "?"
                pos_str = hit.pos if hit.pos else "?"
                result_lines.append(
                    f"{i:2d}. {hit.word:15s} (d={hit.dist_a}/{hit.dist_b}, "
                    f"{syl_str}, {pos_str}, score={hit.score:.2f})"
                )

            if len(hits) > 20:
                result_lines.append(f"\n... and {len(hits) - 20} more results")

            return "\n".join(result_lines)

        elif procedure_id == "semantic_path":
            from poetryplayground.semantic_geodesic import find_semantic_path, get_semantic_space

            start_word = config.get("start_word", "").strip()
            end_word = config.get("end_word", "").strip()

            if not start_word or not end_word:
                return "Error: Both start and end words are required"

            try:
                steps = int(config.get("steps", 5))
                if steps < 3:
                    steps = 3
            except ValueError:
                steps = 5

            try:
                alternatives = int(config.get("alternatives", 3))
                if alternatives < 1:
                    alternatives = 1
            except ValueError:
                alternatives = 3

            method = config.get("method", "linear").lower()
            if method not in ["linear", "bezier", "shortest"]:
                method = "linear"

            try:
                # Load semantic space (cached)
                semantic_space = get_semantic_space()

                # Find path
                path = find_semantic_path(
                    start_word,
                    end_word,
                    steps=steps,
                    k=alternatives,
                    method=method,
                    semantic_space=semantic_space,
                )

                # Format output
                primary_path = path.get_primary_path()
                result_lines = [
                    f"Semantic Path ({method} method):",
                    f"{' → '.join(primary_path)}",
                    "",
                    "Quality Metrics:",
                    f"  Smoothness: {path.smoothness_score:.3f} {'★' * int(path.smoothness_score * 5)}",
                    f"  Deviation:  {path.deviation_score:.3f}",
                    f"  Diversity:  {path.diversity_score:.3f}",
                    "",
                ]

                # Add alternatives if k > 1
                if alternatives > 1 and path.bridges:
                    result_lines.append("Alternatives at each step:")
                    for i, step in enumerate(path.bridges, 1):
                        if step:
                            alts = ", ".join([f"{b.word} ({b.similarity:.3f})" for b in step[:3]])
                            result_lines.append(f"  Step {i}: {alts}")
                    result_lines.append("")

                return "\n".join(result_lines)

            except Exception as e:
                import traceback

                return f"Error finding semantic path:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id == "conceptual_cloud":
            from poetryplayground.conceptual_cloud import (
                format_as_json,
                format_as_markdown,
                format_as_rich,
                format_as_simple,
                generate_conceptual_cloud,
            )

            center_word = config.get("center_word", "").strip()

            if not center_word:
                return "Error: Center word is required"

            try:
                k_per_cluster = int(config.get("k_per_cluster", 10))
                if k_per_cluster < 1:
                    k_per_cluster = 10
            except ValueError:
                k_per_cluster = 10

            sections_input = config.get("sections", "all").strip()
            if sections_input.lower() == "all" or not sections_input:
                sections = None
            else:
                sections = [s.strip() for s in sections_input.split(",")]

            output_format = config.get("output_format", "simple").strip().lower()
            if output_format not in ["rich", "json", "markdown", "simple"]:
                output_format = "simple"  # Default to simple for TUI compatibility

            try:
                # Generate cloud
                cloud = generate_conceptual_cloud(
                    center_word=center_word,
                    k_per_cluster=k_per_cluster,
                    sections=sections,
                )

                # Format output based on user preference
                if output_format == "json":
                    return format_as_json(cloud)
                elif output_format == "markdown":
                    return format_as_markdown(cloud, show_scores=True)
                elif output_format == "rich":
                    # Strip ANSI codes from Rich output for plain text TUI display
                    rich_output = format_as_rich(cloud, show_scores=True)
                    return self._strip_ansi_codes(rich_output)
                else:  # simple (default for TUI)
                    return format_as_simple(cloud)

            except Exception as e:
                import traceback

                return f"Error generating conceptual cloud:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id == "corpus":
            from poetryplayground.corpus_analyzer import PersonalCorpusAnalyzer

            directory = config.get("directory", "").strip()

            if not directory:
                return "Error: Directory path is required"

            try:
                # Initialize analyzer
                analyzer = PersonalCorpusAnalyzer()

                # Analyze the directory
                fingerprint = analyzer.analyze_directory(directory)

                # Generate reports
                style_report = analyzer.generate_style_report(fingerprint)
                inspiration_report = analyzer.generate_inspiration_report(fingerprint)

                # Combine reports for display
                result_lines = [
                    "=" * 60,
                    "STYLE ANALYSIS",
                    "=" * 60,
                    "",
                    style_report,
                    "",
                    "=" * 60,
                    "CREATIVE INSPIRATIONS",
                    "=" * 60,
                    "",
                    inspiration_report,
                ]

                return "\n".join(result_lines)

            except FileNotFoundError:
                return f"Error: Directory not found: {directory}\n\nPlease check the path and try again."
            except Exception as e:
                import traceback

                return f"Error during corpus analysis:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id == "theseus":
            from poetryplayground.ship_of_theseus import ShipOfTheseusTransformer

            poem_text = config.get("poem_text", "").strip()

            if not poem_text:
                return "Error: Poem text is required"

            try:
                steps = int(config.get("steps", 5))
                if steps < 1:
                    steps = 5
            except ValueError:
                steps = 5

            preserve_pos = config.get("preserve_pos", "yes").lower() in ["yes", "true", "1"]
            preserve_syllables = config.get("preserve_syllables", "yes").lower() in [
                "yes",
                "true",
                "1",
            ]

            try:
                # Initialize transformer
                transformer = ShipOfTheseusTransformer()

                # Perform gradual transformation
                results = transformer.gradual_transform(
                    original=poem_text,
                    steps=steps,
                    preserve_pos=preserve_pos,
                    preserve_syllables=preserve_syllables,
                )

                # Format results showing progression
                result_lines = [
                    f"Ship of Theseus Transformation ({steps} steps)",
                    "=" * 60,
                    "",
                ]

                for i, result in enumerate(results):
                    result_lines.append(f"Step {i}: {result.replacement_ratio:.0%} replaced")
                    result_lines.append(result.transformed)
                    result_lines.append("")

                result_lines.append("=" * 60)
                result_lines.append("Transformation Complete!")
                result_lines.append(
                    f"Total replacements: {results[-1].num_replacements if results else 0} words"
                )

                return "\n".join(result_lines)

            except Exception as e:
                import traceback

                return f"Error during transformation:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id == "sixdegrees":
            from poetryplayground.six_degrees import SixDegrees

            word_a = config.get("word_a", "").strip()
            word_b = config.get("word_b", "").strip()

            if not word_a or not word_b:
                return "Error: Both words are required"

            try:
                # Initialize six degrees explorer
                sd = SixDegrees()

                # Find convergence
                convergence = sd.find_convergence(word_a, word_b)

                if convergence:
                    # Format the convergence report
                    report = sd.format_convergence_report(convergence)
                    return report
                else:
                    return (
                        f"❌ No convergence found between '{word_a}' and '{word_b}'\n\n"
                        "These words may be too semantically distant to connect\n"
                        "within a reasonable number of steps."
                    )

            except Exception as e:
                import traceback

                return f"Error during convergence search:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id == "futurist":
            from poetryplayground.lexigen import phonetically_related_words
            from poetryplayground.poemgen import PoemGenerator

            input_words_str = config.get("input_words", "").strip()

            if not input_words_str:
                return "Error: Input words are required"

            # Parse comma-separated input words
            input_words = [w.strip() for w in input_words_str.split(",") if w.strip()]

            try:
                num_lines = int(config.get("num_lines", 25))
                if num_lines < 1:
                    num_lines = 25
            except ValueError:
                num_lines = 25

            try:
                # Generate futurist poem
                import random

                connectors = [" + ", " - ", " * ", " % ", " = ", " != ", " :: "]
                word_list = input_words + phonetically_related_words(input_words)
                poem_lines = []
                pgen = PoemGenerator()

                for _ in range(num_lines):
                    random.shuffle(word_list)
                    poem_lines.append(
                        pgen.poem_line_from_word_list(
                            word_list, connectors=connectors, max_line_length=40
                        )
                    )

                return "\n".join(poem_lines)

            except Exception as e:
                import traceback

                return f"Error generating futurist poem:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id == "markov":
            from poetryplayground.poemgen import PoemGenerator

            input_words_str = config.get("input_words", "").strip()

            if not input_words_str:
                return "Error: Input words are required"

            # Parse comma-separated input words
            input_words = [w.strip() for w in input_words_str.split(",") if w.strip()]

            try:
                num_lines = int(config.get("num_lines", 10))
                if num_lines < 1:
                    num_lines = 10
            except ValueError:
                num_lines = 10

            try:
                # Generate markov poem
                pgen = PoemGenerator()
                poem_text = pgen.poem_from_markov(input_words, num_lines=num_lines)

                return poem_text

            except Exception as e:
                import traceback

                return f"Error generating markov poem:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id == "puzzle":
            from poetryplayground.poemgen import PoemGenerator

            input_words_str = config.get("input_words", "").strip()

            if not input_words_str:
                return "Error: Input words are required"

            # Parse comma-separated input words
            input_words = [w.strip() for w in input_words_str.split(",") if w.strip()]

            try:
                # Generate puzzle poem
                pgen = PoemGenerator()
                poem_text = pgen.poem_from_word_list(input_words)

                return poem_text

            except Exception as e:
                import traceback

                return f"Error generating puzzle poem:\n{e!s}\n\n{traceback.format_exc()}"

        elif procedure_id in ["chaotic", "charsoup", "wordsoup"]:
            # PDF-only generators
            generator_names = {
                "chaotic": "Chaotic Concrete Poem",
                "charsoup": "Character Soup Poem",
                "wordsoup": "Stop Word Soup Poem",
            }
            name = generator_names.get(procedure_id, "This generator")

            return (
                f"{name} is a PDF-only visual poetry generator.\n\n"
                "This generator creates spatial/visual arrangements that cannot be\n"
                "displayed in text-only format. To generate these poems, use the\n"
                "legacy PDF generator:\n\n"
                "  poetry-playground\n\n"
                "Or import the PDF generator directly:\n\n"
                "  from poetryplayground.pdf import ChaoticConcretePoemPDFGenerator\n"
                "  generator = ChaoticConcretePoemPDFGenerator()\n"
                "  generator.generate_pdf(input_words=['your', 'words'])"
            )

        elif procedure_id == "deps":
            from poetryplayground.system_utils import check_system_dependencies

            try:
                # Run dependency checks
                result_lines = ["=" * 60, "SYSTEM DEPENDENCY CHECK", "=" * 60, ""]

                # Get dependency status
                deps_report = check_system_dependencies()

                # Format the report for display
                result_lines.append(deps_report)

                return "\n".join(result_lines)

            except Exception as e:
                import traceback

                return f"Error checking dependencies:\n{e!s}\n\n{traceback.format_exc()}"

        else:
            return f"Generator for '{procedure_id}' not yet implemented in TUI.\n\nUse the CLI interface for now:\n  poetry-playground"

    @staticmethod
    def _strip_ansi_codes(text: str) -> str:
        """Strip ANSI escape codes from text for plain text display.

        Args:
            text: Text potentially containing ANSI escape sequences

        Returns:
            Clean text without ANSI codes
        """
        import re

        # Strip ANSI escape sequences (colors, formatting)
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        text = ansi_escape.sub("", text)

        # Strip other control sequences
        ansi_control = re.compile(r"\x1b\[[^m]*m?")
        text = ansi_control.sub("", text)

        return text

    def _on_generation_complete(self, result: str) -> None:
        """Handle successful generation completion (main thread).

        Args:
            result: Generated text output
        """
        # Update status
        status_widget = self.query_one("#status", Static)
        status_widget.update("✓ Generation complete!")

        # Show output screen
        from poetryplayground.tui.screens.output_view import OutputViewScreen

        self.app.push_screen(OutputViewScreen(result, self.config["name"]))

    def _on_generation_error(self, error_msg: str) -> None:
        """Handle generation error (main thread).

        Args:
            error_msg: Error message to display
        """
        status_widget = self.query_one("#status", Static)
        status_widget.update(f"❌ Error: {error_msg}")

    async def _handle_back(self) -> None:
        """Handle back button click."""
        self.app.pop_screen()
