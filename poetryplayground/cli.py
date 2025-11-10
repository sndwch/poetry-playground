#!/usr/bin/env python3
import argparse
import os
import re

from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table

# Import the pronouncing patch first to suppress pkg_resources warning
import poetryplayground.pronouncing_patch  # noqa: F401
from poetryplayground.causal_poetry import ResonantFragmentMiner
from poetryplayground.config import init_config
from poetryplayground.corpus_analyzer import PersonalCorpusAnalyzer
from poetryplayground.finders import find_equidistant
from poetryplayground.forms import FormGenerator
from poetryplayground.idea_generator import IdeaType, PoetryIdeaGenerator
from poetryplayground.line_seeds import LineSeedGenerator, SeedType
from poetryplayground.logger import enable_profiling, set_log_level
from poetryplayground.metaphor_generator import MetaphorGenerator, MetaphorType
from poetryplayground.pdf import (
    ChaoticConcretePoemPDFGenerator,
    CharacterSoupPoemPDFGenerator,
    FuturistPoemPDFGenerator,
    MarkovPoemPDFGenerator,
    StopwordSoupPoemPDFGenerator,
)
from poetryplayground.poem_transformer import PoemTransformer
from poetryplayground.poemgen import PoemGenerator, print_poem
from poetryplayground.rich_console import console
from poetryplayground.rich_output import display_poem_output
from poetryplayground.seed_manager import format_seed_message, set_global_seed
from poetryplayground.setup_models import setup as setup_models
from poetryplayground.six_degrees import SixDegrees
from poetryplayground.system_utils import check_system_dependencies
from poetryplayground.utils import get_input_words

reuse_words_prompt = "\nType yes to use the same words again, Otherwise just hit enter.\n"


def parse_syllable_range(value: str) -> tuple:
    """Parse a syllable range string like '1..3', '2', '2..', or '..5'.

    Args:
        value: Range string to parse

    Returns:
        Tuple of (min_syllables, max_syllables)

    Raises:
        argparse.ArgumentTypeError: If the format is invalid
    """
    if ".." in value:
        parts = value.split("..")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid range format: {value}")
        try:
            min_s = int(parts[0]) if parts[0] else 0
            max_s = int(parts[1]) if parts[1] else 999  # Effectively unlimited
            return (min_s, max_s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid range format: {value}") from None
    else:
        try:
            count = int(value)
            return (count, count)  # Exact syllable count
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid syllable count: {value}") from None


def _list_procedures():
    """Display available generation procedures in a Rich table."""
    table = Table(
        title="✨ Available Generation Procedures ✨",
        header_style="bold magenta",
        show_header=True,
        border_style="blue",
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Procedure", style="bold cyan", width=30)
    table.add_column("Description", style="white", no_wrap=False)
    table.add_column("Category", style="yellow", width=15)

    # Define procedures with their categories
    procedures = [
        # Visual/Concrete Poetry
        ("1", "Futurist Poem", "Marinetti-inspired mathematical word connections", "Visual Poetry"),
        ("2", "Stochastic Jolastic", "Joyce-like wordplay with rhyme schemes", "Visual Poetry"),
        ("3", "Chaotic Concrete", "Abstract spatial arrangements", "Visual Poetry"),
        ("4", "Character Soup", "Pure visual typographic experiments", "Visual Poetry"),
        ("5", "Stop Word Soup", "Minimalist word placement", "Visual Poetry"),
        ("6", "Visual Puzzle", "Interactive terminal-based concrete poetry", "Visual Poetry"),
        # Poetry Ideation Tools
        ("7", "Line Seeds", "Opening lines and pivotal fragments", "Ideation"),
        ("8", "Metaphor Generator", "AI-assisted metaphor creation", "Ideation"),
        ("9", "Corpus Analyzer", "Analyze personal poetry collection", "Ideation"),
        ("10", "Ship of Theseus", "Transform existing poems", "Ideation"),
        ("11", "Poetry Ideas", "Mine classic literature for creative seeds", "Ideation"),
        ("12", "Six Degrees", "Explore connections between concepts", "Ideation"),
        ("13", "Fragment Miner", "Extract poetic fragments from Gutenberg", "Ideation"),
        # Syllable-Constrained Forms
        ("14", "Haiku Generator", "Traditional 5-7-5 syllable form", "Forms"),
        ("15", "Tanka Generator", "Extended 5-7-5-7-7 syllable form", "Forms"),
        ("16", "Senryu Generator", "Human-focused 5-7-5 form", "Forms"),
    ]

    # Add rows with section separators between categories
    current_category = None
    for num, name, desc, category in procedures:
        if category != current_category:
            if current_category is not None:
                table.add_section()  # Visual separator
            current_category = category
        table.add_row(num, name, desc, category)

    console.print(table)
    console.print("\n[dim]Use poetry-playground to access all procedures interactively.[/dim]")


def _list_fonts():
    """Display available fonts in a responsive column layout."""
    # Standard PostScript fonts (always available with reportlab)
    standard_fonts = [
        "Courier",
        "Courier-Bold",
        "Courier-BoldOblique",
        "Courier-Oblique",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-BoldOblique",
        "Helvetica-Oblique",
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
        "Symbol",
        "ZapfDingbats",
    ]

    # Try to get additionally registered fonts
    custom_fonts = []
    try:
        from reportlab.pdfbase import pdfmetrics

        registered = pdfmetrics.getRegisteredFontNames()
        if registered and len(registered) > len(standard_fonts):
            custom_fonts = sorted([f for f in registered if f not in standard_fonts])
    except Exception:
        # Silently continue if font registry is unavailable
        pass

    # Create styled renderables for standard fonts
    font_renderables = [f"[cyan]•[/] [yellow]{font}[/]" for font in standard_fonts]

    # Create Columns (auto-arranges based on terminal width)
    font_columns = Columns(font_renderables, equal=False, expand=False)

    # Wrap in Panel
    panel = Panel(
        font_columns,
        title="🎨 Standard PostScript Fonts",
        subtitle="[dim]Always available[/dim]",
        border_style="blue",
        padding=(1, 2),
    )

    console.print(panel)

    # Show custom fonts if any
    if custom_fonts:
        console.print()  # Add spacing
        custom_renderables = [f"[cyan]•[/] [yellow]{font}[/]" for font in custom_fonts]
        custom_columns = Columns(custom_renderables, equal=False, expand=False)
        custom_panel = Panel(
            custom_columns,
            title="📦 Additionally Registered Fonts",
            border_style="green",
            padding=(1, 2),
        )
        console.print(custom_panel)

    console.print(
        "\n[dim]Note: Custom fonts can be registered using reportlab.pdfbase.pdfmetrics[/dim]"
    )


def _handle_equidistant(args):
    """Handle the --equidistant command to find words equidistant from two anchors.

    Args:
        args: Parsed command-line arguments containing equidistant and filter options
    """
    a, b = args.equidistant  # Two anchor words

    with console.status(f"[green]Searching {args.mode} echoes for '{a}' and '{b}'..."):
        try:
            hits = find_equidistant(
                a=a,
                b=b,
                mode=args.mode,
                window=args.window,
                min_zipf=args.min_zipf,
                pos_filter=args.pos,
                syllable_filter=args.syllables,
            )
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return

    if not hits:
        console.print("[yellow]No results found with current parameters.[/yellow]")
        console.print("\n[dim]Try:")
        console.print("  • Increasing --window (e.g., --window 1)")
        console.print("  • Lowering --min-zipf (e.g., --min-zipf 2.0)")
        console.print("  • Removing filters (--pos, --syllables)[/dim]")
        return

    target_d = hits[0].target_distance
    window_str = f"±{args.window}" if args.window > 0 else "exact"
    title = f"✨ Equidistant Echoes: '{a}' ⟷ '{b}' ✨"
    subtitle = f"[dim]Target distance: {target_d} ({window_str}) | Mode: {args.mode}[/dim]"

    table = Table(title=title, subtitle=subtitle, show_lines=True, border_style="blue")
    table.add_column("Word", style="bold cyan", no_wrap=True)
    table.add_column("d(A,X)", style="magenta", justify="center", width=6)
    table.add_column("d(B,X)", style="magenta", justify="center", width=6)
    table.add_column("Syl", style="green", justify="center", width=4)
    table.add_column("POS", style="yellow", width=8)
    table.add_column("Zipf", style="blue", justify="right", width=6)
    table.add_column("Score", style="bold white", justify="right", width=7)

    # Show top 50 results
    for hit in hits[:50]:
        table.add_row(
            hit.word,
            str(hit.dist_a),
            str(hit.dist_b),
            str(hit.syllables) if hit.syllables else "—",
            hit.pos if hit.pos != "UNKNOWN" else "—",
            f"{hit.zipf_frequency:.2f}",
            f"{hit.score:.2f}",
        )

    console.print(table)

    # Show summary
    total_hits = len(hits)
    shown = min(50, total_hits)
    console.print(f"\n[dim]Showing top {shown} of {total_hits} results[/dim]")

    # Show filters if any were applied
    filters = []
    if args.pos:
        filters.append(f"POS={args.pos}")
    if args.syllables:
        min_s, max_s = args.syllables
        if min_s == max_s:
            filters.append(f"syllables={min_s}")
        else:
            filters.append(f"syllables={min_s}..{max_s}")
    if args.min_zipf != 3.0:
        filters.append(f"min_zipf={args.min_zipf}")

    if filters:
        console.print(f"[dim]Filters: {', '.join(filters)}[/dim]")


def interactive_loop(poetry_generator):
    exit_loop = False
    input_words = get_input_words()
    while not exit_loop:
        poetry_generator.generate_pdf(input_words=input_words)
        png_created = poetry_generator.generate_png(poetry_generator.pdf_filepath)
        if png_created:
            print(f"Generated: {poetry_generator.pdf_filepath} and PNG version")
        else:
            print(f"Generated: {poetry_generator.pdf_filepath}")
        print(reuse_words_prompt)
        if input() != "yes":
            exit_loop = True


def futurist_poem_action():
    fppg = FuturistPoemPDFGenerator()
    interactive_loop(fppg)


def markov_poem_action():
    mppg = MarkovPoemPDFGenerator()
    interactive_loop(mppg)


def chaotic_concrete_poem_action():
    ccppg = ChaoticConcretePoemPDFGenerator()
    interactive_loop(ccppg)


def character_soup_poem_action():
    csppg = CharacterSoupPoemPDFGenerator()
    csppg.generate_pdf()


def stopword_soup_poem_action():
    ssppg = StopwordSoupPoemPDFGenerator()
    ssppg.generate_pdf()


def visual_puzzle_poem_action():
    exit_loop = False
    pg = PoemGenerator()
    input_words = get_input_words()
    while not exit_loop:
        print_poem(pg.poem_from_word_list(input_words))
        print(reuse_words_prompt)
        if input() != "yes":
            exit_loop = True


def check_dependencies_action():
    check_system_dependencies()
    print("\nPress Enter to continue...")
    input()


def metaphor_generator_action():
    """Generate metaphors for poetry ideation."""
    generator = MetaphorGenerator()
    exit_loop = False
    input_words = get_input_words()

    while not exit_loop:
        console.print("\n" + "=" * 60)
        console.print(f"Metaphor Generation for: {', '.join(input_words)}")
        console.print("=" * 60 + "\n")

        # Generate metaphors with status spinner
        with console.status("[bold green]Generating metaphors...", spinner="dots12"):
            console.log("Generating metaphor batch...")
            metaphors = generator.generate_metaphor_batch(input_words, count=15)

            console.log("Mining literary patterns from diverse sources...")
            gutenberg_patterns = generator.extract_metaphor_patterns(num_texts=5)

        console.log("✓ Generation complete!")

        # Group by type
        by_type = {}
        for metaphor in metaphors:
            if metaphor.metaphor_type not in by_type:
                by_type[metaphor.metaphor_type] = []
            by_type[metaphor.metaphor_type].append(metaphor)

        # Display results
        print("\nSIMILES & COMPARISONS:")
        print("-" * 40)
        similes = by_type.get(MetaphorType.SIMILE, [])
        for m in similes[:4]:
            print(f"  • {m.text}")
            if m.grounds:
                print(f"    (connecting: {', '.join(m.grounds[:2])})")

        print("\nDIRECT METAPHORS:")
        print("-" * 40)
        direct = by_type.get(MetaphorType.DIRECT, [])
        for m in direct[:4]:
            print(f"  • {m.text}")

        print("\nIMPLIED METAPHORS:")
        print("-" * 40)
        implied = by_type.get(MetaphorType.IMPLIED, [])
        for m in implied[:4]:
            print(f"  • {m.text}")

        print("\nPOSSESSIVE FORMS:")
        print("-" * 40)
        possessive = by_type.get(MetaphorType.POSSESSIVE, [])
        for m in possessive[:3]:
            print(f"  • {m.text}")

        # Generate and show an extended metaphor for the first word
        if input_words:
            print("\nEXTENDED METAPHOR:")
            print("-" * 40)
            # Find a good target for extended metaphor
            targets = generator._find_target_domains(input_words[0])
            if targets:
                extended = generator.generate_extended_metaphor(input_words[0], targets[0])
                for line in extended.text.split("\n"):
                    print(f"  {line}")

        # Show some Gutenberg-inspired patterns if found
        if gutenberg_patterns:
            print("\nINSPIRED BY CLASSIC LITERATURE:")
            print("-" * 40)

            # Group patterns by source text to show diversity
            text_groups = {}
            for source, target, sentence in gutenberg_patterns:
                text_key = sentence[:50]  # Use first 50 chars as grouping key
                if text_key not in text_groups:
                    text_groups[text_key] = []
                text_groups[text_key].append((source, target, sentence))

            # Show patterns from different texts
            shown_count = 0
            for _text_key, patterns in text_groups.items():
                if shown_count >= 3:
                    break
                source, target, sentence = patterns[0]  # Take first pattern from this text
                print(f"  • {source} like {target}")
                print(f'    From: "{sentence[:80]}..."')
                shown_count += 1  # noqa: SIM113

            if len(text_groups) > 1:
                print(f"    (Patterns from {len(text_groups)} different classic texts)")

        # Generate a synesthetic metaphor
        print("\nSYNESTHETIC (CROSS-SENSORY):")
        print("-" * 40)
        for word in input_words[:2]:
            synesthetic = generator.generate_synesthetic_metaphor(word)
            if synesthetic:
                print(f"  • {synesthetic.text}")

        print("\n" + "=" * 60)
        print("\nOptions:")
        print("  1. Generate new metaphors with same words")
        print("  2. Use different words")
        print("  3. Mine more Gutenberg texts")
        print("  4. Return to main menu")

        choice = input("\nYour choice (1-4): ").strip()

        if choice == "1":
            continue  # Regenerate with same words
        elif choice == "2":
            input_words = get_input_words()  # Get new words
        elif choice == "3":
            print("\nMining additional Gutenberg texts...")
            patterns = generator.extract_metaphor_patterns(num_texts=8)
            if patterns:
                # Group by text source to show diversity
                text_groups = {}
                for source, target, sentence in patterns:
                    text_key = sentence[:60]
                    if text_key not in text_groups:
                        text_groups[text_key] = []
                    text_groups[text_key].append((source, target, sentence))

                print(
                    f"Found {len(patterns)} metaphorical patterns from {len(text_groups)} different texts"
                )

                # Show examples from different texts
                shown_texts = 0
                for _text_key, group_patterns in text_groups.items():
                    if shown_texts >= 3:
                        break
                    source, target, sentence = group_patterns[0]
                    print(f"  • {source} like {target}")
                    print(f'    From: "{sentence[:80]}..."')
                    shown_texts += 1  # noqa: SIM113

                if len(text_groups) > 3:
                    print(f"    (Plus patterns from {len(text_groups) - 3} more texts)")
            else:
                print("No additional patterns found")
        else:
            exit_loop = True


def line_seeds_action():
    """Generate line seeds for poetry ideation."""
    generator = LineSeedGenerator()
    exit_loop = False
    input_words = get_input_words()

    while not exit_loop:
        print("\n" + "=" * 50)
        print(f"Line Seeds for: {', '.join(input_words)}")
        print("=" * 50 + "\n")

        # Generate a collection of seeds
        seeds = generator.generate_seed_collection(input_words, num_seeds=10)

        # Display by category
        print("OPENING LINES:")
        print("-" * 30)
        opening_seeds = [s for s in seeds if s.seed_type == SeedType.OPENING]
        if not opening_seeds:
            opening_seeds = [generator.generate_opening_line(input_words)]
        for seed in opening_seeds[:3]:
            print(f"  {seed.text}")
            if seed.notes:
                print(f"    ({seed.notes})")

        print("\nPIVOTAL FRAGMENTS:")
        print("-" * 30)
        pivot_seeds = [s for s in seeds if s.seed_type in [SeedType.PIVOT, SeedType.FRAGMENT]]
        for seed in pivot_seeds[:3]:
            print(f"  {seed.text}")

        print("\nIMAGE SEEDS:")
        print("-" * 30)
        image_seeds = [s for s in seeds if s.seed_type == SeedType.IMAGE]
        if not image_seeds:
            image_seeds = [generator.generate_image_seed(input_words)]
        for seed in image_seeds[:3]:
            print(f"  {seed.text}")

        print("\nSONIC PATTERNS:")
        print("-" * 30)
        sonic_seeds = [s for s in seeds if s.seed_type == SeedType.SONIC]
        if not sonic_seeds:
            sonic_seeds = [generator.generate_sonic_pattern(input_words)]
        for seed in sonic_seeds[:2]:
            print(f"  {seed.text}")
            if seed.notes:
                print(f"    ({seed.notes})")

        print("\nENDING APPROACHES:")
        print("-" * 30)
        closing_seeds = [s for s in seeds if s.seed_type == SeedType.CLOSING]
        if not closing_seeds:
            # Get an opening to potentially echo
            opening = opening_seeds[0].text if opening_seeds else None
            closing_seeds = [generator.generate_ending_approach(input_words, opening)]
        for seed in closing_seeds[:2]:
            print(f"  {seed.text}")
            if seed.notes:
                print(f"    ({seed.notes})")

        print("\n" + "=" * 50)
        print("\nOptions:")
        print("  1. Generate new seeds with same words")
        print("  2. Use different words")
        print("  3. Return to main menu")

        choice = input("\nYour choice (1-3): ").strip()

        if choice == "1":
            continue  # Loop with same words
        elif choice == "2":
            input_words = get_input_words()  # Get new words
        else:
            exit_loop = True  # Exit


def haiku_action():
    """Generate haiku poems (5-7-5 syllable pattern)."""
    generator = FormGenerator()
    exit_loop = False

    # Get optional seed words
    print("\n" + "=" * 50)
    print("HAIKU GENERATOR (5-7-5 syllable pattern)")
    print("=" * 50)
    print("\nOptional: Enter seed words to guide generation")
    print("(or press Enter to generate without seeds)")
    seed_input = input("\nSeed words (space or comma separated): ").strip()
    seed_words = [w for w in re.split(r"[\s,]", seed_input) if w] if seed_input else None

    while not exit_loop:
        console.print("\n" + "=" * 50)
        console.print("Generating Haiku...")
        console.print("=" * 50 + "\n")

        try:
            with console.status("[bold green]Generating haiku...", spinner="dots"):
                console.log("Loading POS vocabulary...")
                console.log("Selecting grammatical templates...")
                lines, validation = generator.generate_haiku(seed_words=seed_words, strict=True)

            # Get seed for metadata
            from poetryplayground.seed_manager import get_global_seed

            # Display the haiku using Rich Panel
            metadata = {"form": "haiku", "syllables": "5-7-5"}
            seed = get_global_seed()
            if seed is not None:
                metadata["seed"] = str(seed)

            display_poem_output(lines, title="Haiku", metadata=metadata)

            # Show validation if verbose
            if validation.valid:
                console.print("\n[green]✓ Valid haiku (5-7-5 syllables)[/green]")
            else:
                console.print("\n" + validation.get_report())

        except ValueError as e:
            print(f"\nError generating haiku: {e}")
            print("Try again or use different seed words.")

        print("\n" + "=" * 50)
        print("\nOptions:")
        print("  1. Generate another haiku with same seeds")
        print("  2. Use different seed words")
        print("  3. Return to main menu")

        choice = input("\nYour choice (1-3): ").strip()

        if choice == "1":
            continue  # Loop with same seeds
        elif choice == "2":
            seed_input = input("\nSeed words (space or comma separated): ").strip()
            seed_words = [w for w in re.split(r"[\s,]", seed_input) if w] if seed_input else None
        else:
            exit_loop = True


def tanka_action():
    """Generate tanka poems (5-7-5-7-7 syllable pattern)."""
    generator = FormGenerator()
    exit_loop = False

    # Get optional seed words
    print("\n" + "=" * 50)
    print("TANKA GENERATOR (5-7-5-7-7 syllable pattern)")
    print("=" * 50)
    print("\nOptional: Enter seed words to guide generation")
    print("(or press Enter to generate without seeds)")
    seed_input = input("\nSeed words (space or comma separated): ").strip()
    seed_words = [w for w in re.split(r"[\s,]", seed_input) if w] if seed_input else None

    while not exit_loop:
        print("\n" + "=" * 50)
        print("Generating Tanka...")
        print("=" * 50 + "\n")

        try:
            lines, validation = generator.generate_tanka(seed_words=seed_words, strict=True)

            # Display the tanka
            for line in lines:
                print(f"  {line}")

            # Show validation if verbose
            if validation.valid:
                print("\n✓ Valid tanka (5-7-5-7-7 syllables)")
            else:
                print("\n" + validation.get_report())

        except ValueError as e:
            print(f"\nError generating tanka: {e}")
            print("Try again or use different seed words.")

        print("\n" + "=" * 50)
        print("\nOptions:")
        print("  1. Generate another tanka with same seeds")
        print("  2. Use different seed words")
        print("  3. Return to main menu")

        choice = input("\nYour choice (1-3): ").strip()

        if choice == "1":
            continue  # Loop with same seeds
        elif choice == "2":
            seed_input = input("\nSeed words (space or comma separated): ").strip()
            seed_words = [w for w in re.split(r"[\s,]", seed_input) if w] if seed_input else None
        else:
            exit_loop = True


def senryu_action():
    """Generate senryu poems (5-7-5 syllable pattern, human-focused)."""
    generator = FormGenerator()
    exit_loop = False

    # Get optional seed words
    print("\n" + "=" * 50)
    print("SENRYU GENERATOR (5-7-5 syllable pattern)")
    print("=" * 50)
    print("\nSenryu uses the same structure as haiku but focuses on")
    print("human nature, emotions, and relationships.")
    print("\nOptional: Enter seed words to guide generation")
    print("(or press Enter to generate without seeds)")
    seed_input = input("\nSeed words (space or comma separated): ").strip()
    seed_words = [w for w in re.split(r"[\s,]", seed_input) if w] if seed_input else None

    while not exit_loop:
        print("\n" + "=" * 50)
        print("Generating Senryu...")
        print("=" * 50 + "\n")

        try:
            lines, validation = generator.generate_senryu(seed_words=seed_words, strict=True)

            # Display the senryu
            for line in lines:
                print(f"  {line}")

            # Show validation if verbose
            if validation.valid:
                print("\n✓ Valid senryu (5-7-5 syllables)")
            else:
                print("\n" + validation.get_report())

        except ValueError as e:
            print(f"\nError generating senryu: {e}")
            print("Try again or use different seed words.")

        print("\n" + "=" * 50)
        print("\nOptions:")
        print("  1. Generate another senryu with same seeds")
        print("  2. Use different seed words")
        print("  3. Return to main menu")

        choice = input("\nYour choice (1-3): ").strip()

        if choice == "1":
            continue  # Loop with same seeds
        elif choice == "2":
            seed_input = input("\nSeed words (space or comma separated): ").strip()
            seed_words = [w for w in re.split(r"[\s,]", seed_input) if w] if seed_input else None
        else:
            exit_loop = True


def corpus_analyzer_action():
    """Analyze a personal poetry corpus for style insights."""
    analyzer = PersonalCorpusAnalyzer()

    print("\n" + "=" * 60)
    print("PERSONAL CORPUS ANALYZER")
    print("=" * 60)
    print("\nThis will analyze your personal poetry collection to identify")
    print("stylistic patterns, vocabulary preferences, and suggest expansions.")

    # Default to the user's known directory, but allow override
    default_dir = "/Users/jparker/Desktop/free-verse"
    print(f"\nDefault directory: {default_dir}")
    directory = input("Enter poetry directory (or press Enter for default): ").strip()

    if not directory:
        directory = default_dir

    try:
        print(f"\nAnalyzing poetry in: {directory}")
        print("This may take a moment...")

        fingerprint = analyzer.analyze_directory(directory)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        # Generate and display the style report
        report = analyzer.generate_style_report(fingerprint)
        print(report)

        # Generate inspiration report
        print("\n" + "=" * 60)
        print("GENERATING CREATIVE INSPIRATIONS...")
        print("=" * 60)
        inspiration_report = analyzer.generate_inspiration_report(fingerprint)
        print(inspiration_report)

        # Provide expansion suggestions
        print("\n" + "=" * 60)
        print("CREATIVE EXPANSION SUGGESTIONS")
        print("=" * 60)
        suggestions = analyzer.suggest_expansions(fingerprint)

        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"\n{i}. {suggestion}")
        else:
            print("\nNo specific suggestions at this time.")
            print("Your style shows good balance and variety!")

        # Vocabulary insights for ideation
        print("\n" + "=" * 60)
        print("IDEATION INSIGHTS")
        print("=" * 60)

        if fingerprint.vocabulary.signature_words:
            print("\nYour most distinctive words:")
            signature_list = [word for word, _ in fingerprint.vocabulary.signature_words[:8]]
            print(f"  {', '.join(signature_list)}")
            print("\nTry building poems around these - they're authentically 'you'")

        if fingerprint.themes.semantic_clusters:
            print("\nThematic word groups you naturally use:")
            for theme, words in fingerprint.themes.semantic_clusters[:3]:
                print(f"  • {theme}: {', '.join(words[:4])}")
            print("\nConsider cross-pollinating between these groups")

        print("\n" + "=" * 60)
        print("\nAnalysis saved. You can use these insights with other")
        print("generative poetry tools to maintain your authentic voice")
        print("while exploring new creative directions.")

    except FileNotFoundError:
        print(f"\nError: Directory not found: {directory}")
        print("Please check the path and try again.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("Please check that the directory contains readable .txt files.")

    print("\nPress Enter to return to main menu...")
    input()


def poem_transformer_action():
    """Transform a poem through Ship of Theseus style replacements"""
    transformer = PoemTransformer()

    print("\n" + "=" * 60)
    print("SHIP OF THESEUS POEM TRANSFORMER")
    print("=" * 60)
    print("\nThis will gradually transform a poem through multiple passes,")
    print("replacing words with similar meaning, contextual, or sound-alike")
    print("alternatives, like the old Google Translate telephone game!")

    # Default to the user's known directory, but allow override
    default_dir = "/Users/jparker/Desktop/free-verse"
    print(f"\nDefault directory: {default_dir}")
    directory = input("Enter poetry directory (or press Enter for default): ").strip()

    if not directory:
        directory = default_dir

    try:
        # List available poems
        poems = transformer.list_poems_in_directory(directory)

        if not poems:
            print(f"No poetry files found in {directory}")
            print("Make sure the directory contains .txt or .md files")
            print("\nPress Enter to return to main menu...")
            input()
            return

        print(f"\nFound {len(poems)} poems:")
        for i, (title, _path) in enumerate(poems, 1):
            print(f"  {i}. {title}")

        # Let user choose a poem
        while True:
            choice = input(f"\nChoose a poem (1-{len(poems)}): ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(poems):
                    selected_title, selected_path = poems[choice_num - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(poems)}")
            except ValueError:
                print("Please enter a valid number")

        # Load the selected poem
        print(f"\nLoading '{selected_title}'...")
        poem_text = transformer.load_poem(selected_path)

        print(f"\nOriginal poem ({len(poem_text.split())} words):")
        print("-" * 40)
        print(poem_text[:300] + ("..." if len(poem_text) > 300 else ""))

        # Get transformation parameters
        print("\nTransformation settings:")

        try:
            passes = input("Number of transformation passes (default: 5): ").strip()
            num_passes = int(passes) if passes else 5
            num_passes = max(1, min(10, num_passes))  # Limit 1-10
        except ValueError:
            num_passes = 5

        try:
            words_per = input("Words to change per pass (default: 8): ").strip()
            words_per_pass = int(words_per) if words_per else 8
            words_per_pass = max(1, min(20, words_per_pass))  # Limit 1-20
        except ValueError:
            words_per_pass = 8

        print(f"\nStarting transformation: {num_passes} passes, ~{words_per_pass} words per pass")
        print("This may take a moment due to API calls...")

        # Perform the transformation
        transformations = transformer.transform_poem_iteratively(
            poem_text, num_passes=num_passes, words_per_pass=words_per_pass
        )

        if transformations:
            # Generate and display the report
            print("\n" + "=" * 60)
            print("TRANSFORMATION COMPLETE")
            print("=" * 60)

            report = transformer.generate_transformation_report(transformations)
            print(report)

            # Offer to save the result
            save_choice = input("\nSave transformed poem? (y/n): ").strip().lower()
            if save_choice == "y":
                output_filename = f"{selected_title}_transformed.txt"
                output_path = os.path.join(directory, output_filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"# Ship of Theseus Transformation of '{selected_title}'\n\n")
                    f.write("## Original:\n")
                    f.write(poem_text)
                    f.write("\n\n## Transformed:\n")
                    f.write(transformations[-1].transformed_poem)
                    f.write("\n\n## Transformation Log:\n")
                    f.write(report)

                print(f"Saved to: {output_path}")

        else:
            print("\nNo transformations were applied. The poem may be too short")
            print("or contain mostly function words that can't be transformed.")

    except FileNotFoundError:
        print(f"\nError: Directory not found: {directory}")
        print("Please check the path and try again.")
    except Exception as e:
        print(f"\nError during transformation: {e}")
        print("Please try again with different settings.")

    print("\nPress Enter to return to main menu...")
    input()


def idea_generator_action():
    """Generate poetry ideas from classic literature"""
    generator = PoetryIdeaGenerator()

    print("\n" + "=" * 60)
    print("POETRY IDEA GENERATOR")
    print("=" * 60)
    print("\nStruggling with writer's block? Let classic literature spark")
    print("your creativity! This tool mines Project Gutenberg texts for")
    print("evocative fragments, scenarios, and concepts to inspire poems.")

    # Get number of ideas to generate
    try:
        num_input = input("\nHow many ideas would you like? (default: 20): ").strip()
        num_ideas = int(num_input) if num_input else 20
        num_ideas = max(5, min(50, num_ideas))  # Limit 5-50
    except ValueError:
        num_ideas = 20

    # Ask about preferred categories
    print("\nIdea categories available:")
    categories = {
        1: ("Emotional Moments", IdeaType.EMOTIONAL_MOMENT),
        2: ("Vivid Imagery", IdeaType.VIVID_IMAGERY),
        3: ("Character Situations", IdeaType.CHARACTER_SITUATION),
        4: ("Philosophical Fragments", IdeaType.PHILOSOPHICAL_FRAGMENT),
        5: ("Setting Descriptions", IdeaType.SETTING_DESCRIPTION),
        6: ("Dialogue Sparks", IdeaType.DIALOGUE_SPARK),
        7: ("Opening Lines", IdeaType.OPENING_LINE),
        8: ("Sensory Details", IdeaType.SENSORY_DETAIL),
        9: ("Conflict Scenarios", IdeaType.CONFLICT_SCENARIO),
        10: ("Metaphysical Concepts", IdeaType.METAPHYSICAL_CONCEPT),
    }

    for num, (name, _) in categories.items():
        print(f"  {num}. {name}")

    print("  11. All categories (mixed)")

    try:
        category_choice = input("\nChoose category (1-11, default: 11): ").strip()
        choice_num = int(category_choice) if category_choice else 11

        if 1 <= choice_num <= 10:
            preferred_types = [categories[choice_num][1]]
            print(f"Focusing on: {categories[choice_num][0]}")
        else:
            preferred_types = None  # All categories
            print("Using all categories")

    except ValueError:
        preferred_types = None
        print("Using all categories")

    print(f"\nGenerating {num_ideas} poetry ideas...")
    print("This may take a moment as we sample from classic literature...")

    try:
        # Generate the ideas
        collection = generator.generate_ideas(num_ideas, preferred_types)

        if collection.total_count() > 0:
            print("\n" + "=" * 60)
            print("IDEAS GENERATED")
            print("=" * 60)

            # Generate and display the report
            report = generator.generate_idea_report(collection)
            print(report)

            # Interactive options
            print("\n" + "=" * 60)
            print("OPTIONS:")
            print("1. Generate more ideas")
            print("2. Focus on a different category")
            print("3. Get random selection for immediate use")
            print("4. Return to main menu")

            choice = input("\nYour choice (1-4): ").strip()

            if choice == "1":
                # Generate more ideas
                more_ideas = generator.generate_ideas(10, preferred_types)
                print(f"\nGenerated {more_ideas.total_count()} additional ideas:")
                additional_report = generator.generate_idea_report(more_ideas)
                print(additional_report)

            elif choice == "2":
                # Recursive call with different category
                idea_generator_action()
                return

            elif choice == "3":
                # Random selection for immediate use
                random_ideas = collection.get_random_mixed_selection(5)
                print("\nRANDOM SELECTION FOR IMMEDIATE USE:")
                print("-" * 50)
                for i, idea in enumerate(random_ideas, 1):
                    category_name = idea.idea_type.value.replace("_", " ").title()
                    print(f'{i}. [{category_name}] "{idea.text}"')
                    print(f"   Prompt: {idea.creative_prompt}")
                    print()

        else:
            print("\nUnable to generate ideas at this time.")
            print("This might be due to temporary Gutenberg API issues.")
            print("Please try again in a moment.")

    except Exception as e:
        print(f"\nError generating ideas: {e}")
        print("Please try again with different settings.")

    print("\nPress Enter to return to main menu...")
    input()


def six_degrees_action():
    """Explore word convergence paths"""
    sd = SixDegrees()

    print("\n" + "=" * 60)
    print("SIX DEGREES - WORD CONVERGENCE EXPLORER")
    print("=" * 60)
    print("\nDiscover the hidden pathways between words!")
    print("This explores how two words can connect through semantic relationships,")
    print("like the Wikipedia phenomenon where you can reach any page through links.")

    while True:
        print("\n" + "-" * 40)
        word_a = input("Enter first word (or 'exit' to return): ").strip()
        if word_a.lower() == "exit":
            break

        word_b = input("Enter second word: ").strip()
        if not word_a or not word_b:
            print("Please enter both words.")
            continue

        print(f"\n🔍 Searching for convergence between '{word_a}' and '{word_b}'...")

        try:
            convergence = sd.find_convergence(word_a, word_b)
            if convergence:
                print("\n" + sd.format_convergence_report(convergence))
            else:
                print(f"\n❌ No convergence found between '{word_a}' and '{word_b}'")
                print("These words may be too semantically distant to connect")
                print("within a reasonable number of steps.")

        except Exception as e:
            print(f"\nError during convergence search: {e}")

        print("\nTry another pair? (Enter to continue, 'exit' to return)")
        if input().lower() == "exit":
            break


def semantic_geodesic_action():
    """Find semantic paths between words"""
    from poetryplayground.semantic_geodesic import find_semantic_path, get_semantic_space
    from rich.panel import Panel
    from rich.table import Table

    console.print("\n[bold cyan]🌉 SEMANTIC GEODESIC FINDER[/bold cyan]")
    console.print("=" * 60)
    console.print("\n[dim]Draw a straight line through meaning-space and find the words along that path.[/dim]")
    console.print("[dim]Explore gradual transformations: fire → flame → heat → warmth → cool → frost → ice[/dim]\n")

    # Initialize semantic space (with progress indicator)
    with console.status("[bold green]Loading semantic space (this may take 10-30 seconds)...", spinner="dots"):
        try:
            semantic_space = get_semantic_space()
        except Exception as e:
            console.print(f"\n[bold red]Error loading semantic space:[/bold red] {e}")
            console.print("\n[yellow]Tip: Make sure you have en_core_web_lg installed:[/yellow]")
            console.print("  python -m spacy download en_core_web_lg")
            return

    console.print("[green]✓[/green] Semantic space loaded!\n")

    while True:
        console.print("[bold]─[/bold]" * 40)
        word_a = input("Enter start word (or 'exit' to return): ").strip()
        if word_a.lower() == "exit":
            break

        word_b = input("Enter end word: ").strip()
        if not word_a or not word_b:
            console.print("[yellow]Please enter both words.[/yellow]")
            continue

        # Get parameters
        steps_input = input("Steps (default 5): ").strip()
        steps = int(steps_input) if steps_input else 5

        k_input = input("Alternatives per step (default 3): ").strip()
        k = int(k_input) if k_input else 3

        method_input = input("Method - linear/bezier/shortest (default linear): ").strip()
        method = method_input.lower() if method_input in ("linear", "bezier", "shortest") else "linear"

        console.print(f"\n[bold green]Finding {method} path from '{word_a}' to '{word_b}'...[/bold green]")

        try:
            path = find_semantic_path(
                word_a, word_b,
                steps=steps,
                k=k,
                method=method,
                semantic_space=semantic_space
            )

            # Display primary path
            primary_path = path.get_primary_path()
            console.print(f"\n[bold cyan]Primary Path:[/bold cyan] {' → '.join(primary_path)}\n")

            # Display alternatives table
            if k > 1 and path.bridges:
                table = Table(title="Alternatives at Each Step", show_header=True)
                table.add_column("Step", style="cyan")
                table.add_column("Position", style="dim")
                table.add_column("Alternatives", style="yellow")

                for i, step in enumerate(path.bridges, 1):
                    if step:
                        alts = ", ".join([f"{b.word} ({b.similarity:.3f})" for b in step[:3]])
                        pos = f"{step[0].position:.2f}"
                        table.add_row(str(i), pos, alts)

                console.print(table)

            # Display quality metrics
            metrics_panel = Panel(
                f"[cyan]Smoothness:[/cyan] {path.smoothness_score:.3f} "
                f"{'★' * int(path.smoothness_score * 5)}\n"
                f"[cyan]Deviation:[/cyan] {path.deviation_score:.3f}\n"
                f"[cyan]Diversity:[/cyan] {path.diversity_score:.3f}",
                title="[bold]Quality Metrics[/bold]",
                border_style="green"
            )
            console.print(f"\n{metrics_panel}")

            # Suggestions
            console.print("\n[dim]💡 Try different parameters:[/dim]")
            console.print(f"[dim]  • More steps: --steps 10 for finer gradations[/dim]")
            console.print(f"[dim]  • Bezier curve: --method bezier for curved paths[/dim]")
            console.print(f"[dim]  • Shortest path: --method shortest for graph-based paths[/dim]")

        except ValueError as e:
            console.print(f"\n[bold red]Invalid input:[/bold red] {e}")
        except Exception as e:
            console.print(f"\n[bold red]Error finding path:[/bold red] {e}")
            import traceback
            traceback.print_exc()

        console.print("\n[dim]Try another pair? (Enter to continue, 'exit' to return)[/dim]")
        if input().strip().lower() == "exit":
            break


def conceptual_cloud_action():
    """Generate conceptual cloud of word associations"""
    from poetryplayground.conceptual_cloud import (
        generate_conceptual_cloud,
        format_as_rich,
        format_as_json,
        format_as_markdown,
        format_as_simple,
        ClusterType,
    )
    from rich.panel import Panel

    console.print("\n[bold cyan]🌥️  CONCEPTUAL CLOUD GENERATOR[/bold cyan]")
    console.print("=" * 60)
    console.print("\n[dim]Multi-dimensional word associations - a poet's radar[/dim]")
    console.print("[dim]Generates 6 types of clusters: semantic, contextual, opposite, phonetic, imagery, rare[/dim]\n")

    while True:
        console.print("[bold]─[/bold]" * 40)
        center_word = input("Enter center word or phrase (or 'exit' to return): ").strip()
        if center_word.lower() == "exit":
            break

        if not center_word:
            console.print("[yellow]Please enter a word or phrase.[/yellow]")
            continue

        # Get parameters
        k_input = input("Words per cluster (default 10): ").strip()
        k = int(k_input) if k_input else 10

        sections_input = input(
            "Sections to include (comma-separated or 'all', default all):\n"
            "  [dim]Options: semantic, contextual, opposite, phonetic, imagery, rare[/dim]\n"
            "  > "
        ).strip()

        if sections_input.lower() == "all" or not sections_input:
            sections = None
        else:
            sections = [s.strip() for s in sections_input.split(",")]

        format_input = input(
            "Output format (rich/json/markdown/simple, default rich): "
        ).strip().lower()
        output_format = format_input if format_input in ("rich", "json", "markdown", "simple") else "rich"

        console.print(f"\n[bold green]Generating conceptual cloud for '{center_word}'...[/bold green]")

        try:
            # Generate cloud
            cloud = generate_conceptual_cloud(
                center_word=center_word,
                k_per_cluster=k,
                sections=sections,
            )

            console.print(f"\n[green]✓[/green] Generated {cloud.total_terms} terms across {len(cloud.clusters)} clusters\n")

            # Format and display
            if output_format == "rich":
                output = format_as_rich(cloud, show_scores=True)
                console.print(output)
            elif output_format == "json":
                output = format_as_json(cloud)
                console.print(output)
            elif output_format == "markdown":
                output = format_as_markdown(cloud, show_scores=True)
                console.print(output)
            else:  # simple
                output = format_as_simple(cloud)
                console.print(output)

            # Offer to save
            save_input = input("\nSave to file? (y/N): ").strip().lower()
            if save_input == "y":
                from datetime import datetime
                from pathlib import Path

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in center_word.lower()).replace(" ", "_")
                ext = ".json" if output_format == "json" else ".md" if output_format == "markdown" else ".txt"
                filename = f"cloud_{safe_name}_{timestamp}{ext}"

                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                file_path = output_dir / filename

                file_path.write_text(output, encoding="utf-8")
                console.print(f"[green]✓[/green] Saved to {file_path}")

            # Usage tips
            console.print("\n[dim]💡 Tips:[/dim]")
            console.print("[dim]  • Try selecting specific sections for focused exploration[/dim]")
            console.print("[dim]  • Use JSON output for integration with other tools[/dim]")
            console.print("[dim]  • The 'rare' cluster finds unusual but connected words[/dim]")

        except ValueError as e:
            console.print(f"\n[bold red]Invalid input:[/bold red] {e}")
        except Exception as e:
            console.print(f"\n[bold red]Error generating cloud:[/bold red] {e}")
            import traceback
            traceback.print_exc()

        console.print("\n[dim]Try another word? (Enter to continue, 'exit' to return)[/dim]")
        if input().strip().lower() == "exit":
            break


def causal_poetry_action():
    """Mine resonant fragments from classic literature"""
    miner = ResonantFragmentMiner()

    print("\n" + "=" * 60)
    print("RESONANT FRAGMENT MINER")
    print("=" * 60)
    print("\nDiscovers evocative sentence fragments from classic literature")
    print("that could serve as seeds for compression poetry. Mines multiple")
    print("sentence patterns for fragments with poetic weight and causal resonance.")

    while True:
        print("\n" + "-" * 40)
        print("1. Mine 50 fragments (recommended)")
        print("2. Mine 100 fragments")
        print("3. Mine with custom count")
        print("4. Return to main menu")

        choice = input("\nChoice (1-4): ").strip()

        if choice == "1":
            print("\n🔍 Mining 50 resonant fragments...")
            collection = miner.mine_fragments(target_count=50, num_texts=5)
            print("\n" + miner.format_fragment_collection(collection))

        elif choice == "2":
            print("\n🔍 Mining 100 resonant fragments...")
            collection = miner.mine_fragments(target_count=100, num_texts=8)
            print("\n" + miner.format_fragment_collection(collection))

        elif choice == "3":
            try:
                count_input = input("How many fragments to mine? (10-200, default 50): ").strip()
                count = int(count_input) if count_input else 50
                count = max(10, min(200, count))
            except ValueError:
                count = 50

            num_texts = max(3, count // 20)  # Scale text count with fragment target
            print(f"\n🔍 Mining {count} resonant fragments from {num_texts} texts...")
            collection = miner.mine_fragments(target_count=count, num_texts=num_texts)
            print("\n" + miner.format_fragment_collection(collection))

        elif choice == "4":
            break
        else:
            print("Invalid choice. Please enter 1-4.")


def equidistant_action():
    """Find words equidistant from two anchor words"""
    print("\n" + "=" * 60)
    print("EQUIDISTANT WORD FINDER")
    print("=" * 60)
    print("\nDiscovery words that are equally distant from two anchor words,")
    print("measured by Levenshtein edit distance. Useful for finding bridges,")
    print("phonetic echoes, and unexpected connections between concepts.")

    while True:
        print("\n" + "-" * 40)
        word_a = input("Enter first anchor word (or 'exit' to return): ").strip()
        if word_a.lower() == "exit":
            break

        word_b = input("Enter second anchor word: ").strip()
        if not word_a or not word_b:
            print("Please enter both words.")
            continue

        # Ask for mode
        print("\nDistance mode:")
        print("1. Orthographic (spelling-based)")
        print("2. Phonetic (sound-based)")
        mode_choice = input("Choice (1-2, default 1): ").strip()
        mode = "phono" if mode_choice == "2" else "orth"

        # Ask for filters
        print("\nOptional filters (press Enter to skip):")
        window_input = input("Window (±distance tolerance, default 0): ").strip()
        window = int(window_input) if window_input else 0

        min_zipf_input = input("Min frequency (1-10, default 3.0): ").strip()
        min_zipf = float(min_zipf_input) if min_zipf_input else 3.0

        pos_input = input("Part of speech (NOUN/VERB/ADJ/ADV, default any): ").strip().upper()
        pos_filter = pos_input if pos_input in ["NOUN", "VERB", "ADJ", "ADV"] else None

        syl_input = input("Syllables (e.g., '2', '1..3', '2..', default any): ").strip()
        syllable_filter = parse_syllable_range(syl_input) if syl_input else None

        # Perform search
        try:
            print("\n🔍 Searching for equidistant words...")
            hits = find_equidistant(
                a=word_a,
                b=word_b,
                mode=mode,
                window=window,
                min_zipf=min_zipf,
                pos_filter=pos_filter,
                syllable_filter=syllable_filter,
            )

            if hits:
                print(f"\n✓ Found {len(hits)} equidistant words:\n")
                # Show top 20 results in simple table format
                for i, hit in enumerate(hits[:20], 1):
                    syl_str = f"{hit.syllables}s" if hit.syllables else "?"
                    pos_str = hit.pos if hit.pos else "?"
                    print(
                        f"  {i:2d}. {hit.word:15s} "
                        f"(d={hit.dist_a}/{hit.dist_b}, {syl_str}, {pos_str}, "
                        f"score={hit.score:.2f})"
                    )

                if len(hits) > 20:
                    print(f"\n  ... and {len(hits) - 20} more results")
            else:
                print(f"\n❌ No equidistant words found for '{word_a}' and '{word_b}'")
                print("Try increasing the window or adjusting filters.")

        except ValueError as e:
            print(f"\nError: {e}")

        print("\nTry another pair? (Enter to continue, 'exit' to return)")
        if input().lower() == "exit":
            break


def main():
    """Main entry point for the generative poetry CLI.

    Parses command-line arguments and launches the interactive menu.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generative Poetry - Procedural poetry generation and ideation tools",
        epilog="For more information, visit: https://github.com/sndwch/poetryplayground-py",
    )

    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        metavar="PATH",
        help="Path to YAML config file (overrides pyproject.toml and environment variables)",
    )

    parser.add_argument(
        "--spacy-model",
        type=str,
        choices=["sm", "md", "lg"],
        metavar="MODEL",
        help="spaCy model size: sm (13MB, fast), md (40MB, balanced), lg (560MB, accurate)",
    )

    parser.add_argument(
        "--form",
        type=str,
        choices=["haiku", "tanka", "senryu", "free"],
        metavar="FORM",
        help="Poem form with syllable constraints: haiku (5-7-5), tanka (5-7-5-7-7), senryu (5-7-5), free (no constraints)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        help="Random seed for deterministic/reproducible generation (example: --seed 42)",
    )

    # Output control
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        metavar="PATH",
        help="Output directory for generated files (default: current directory)",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["png", "pdf", "svg", "txt"],
        metavar="FORMAT",
        help="Output format: png, pdf, svg, or txt (default: pdf)",
    )

    # CLI behavior
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output (show only warnings and errors)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including debug information",
    )

    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output for better compatibility"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without generating files"
    )

    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Enable performance profiling with detailed timing and cache statistics",
    )

    # Utility commands
    parser.add_argument("--list-fonts", action="store_true", help="List available fonts and exit")

    parser.add_argument(
        "--list-procedures",
        action="store_true",
        help="List available poem generation procedures and exit",
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download and install all required NLTK data and spaCy models, then exit",
    )

    # Equidistant word finder arguments
    parser.add_argument(
        "--equidistant",
        nargs=2,
        metavar=("WORD_A", "WORD_B"),
        help="Find words equidistant from two anchor words (e.g., --equidistant stone storm)",
    )

    parser.add_argument(
        "--mode",
        choices=["orth", "phono"],
        default="orth",
        help="Distance mode for --equidistant: 'orth' (orthographic/spelling) or 'phono' (phonetic/sound)",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Allow distance to be d±WINDOW for --equidistant (0=exact, 1=d±1, etc.)",
    )

    parser.add_argument(
        "--min-zipf",
        type=float,
        default=3.0,
        help="Minimum word frequency on Zipf scale (1-10) for --equidistant (default: 3.0)",
    )

    parser.add_argument(
        "--pos",
        choices=["NOUN", "VERB", "ADJ", "ADV"],
        default=None,
        help="Filter --equidistant results by part of speech (uses spaCy tags)",
    )

    parser.add_argument(
        "--syllables",
        type=parse_syllable_range,
        default=None,
        help="Filter --equidistant by syllable count (e.g., '2' or '1..3' or '2..' or '..5')",
    )

    parser.add_argument("--version", action="version", version="poetryplayground 0.3.4")

    args = parser.parse_args()

    # Handle --list-fonts command
    if args.list_fonts:
        _list_fonts()
        return

    # Handle --list-procedures command
    if args.list_procedures:
        _list_procedures()
        return

    # Handle --equidistant command
    if args.equidistant:
        _handle_equidistant(args)
        return

    # Handle --setup command
    if args.setup:
        print("=" * 70)
        print("Generative Poetry - Model Setup")
        print("=" * 70)
        print("\nThis will download and install all required models:")
        print("  • NLTK data: punkt, words, brown, wordnet, stopwords")
        print("  • spaCy model: en_core_web_sm (~40MB)")
        print("\nThis may take a few minutes on first run...\n")

        success = setup_models(quiet=args.quiet)

        if success:
            print("\n" + "=" * 70)
            print("✓ Setup complete! All models are installed and ready.")
            print("=" * 70)
            print("\nYou can now run poetry-playground to start creating poetry.")
            return 0
        else:
            print("\n" + "=" * 70)
            print("✗ Setup failed. Please check the errors above.")
            print("=" * 70)
            print("\nYou may need to:")
            print("  • Check your internet connection")
            print("  • Ensure you have write permissions to Python's site-packages")
            print("  • Try running with sudo (if on Unix/Linux)")
            return 1

    # Build CLI overrides dict from arguments
    from pathlib import Path

    cli_overrides = {}

    if args.seed is not None:
        cli_overrides["seed"] = args.seed

    if args.out:
        cli_overrides["output_dir"] = Path(args.out).resolve()

    if args.format:
        cli_overrides["output_format"] = args.format

    if args.spacy_model:
        cli_overrides["spacy_model"] = args.spacy_model

    if args.form:
        cli_overrides["poem_form"] = args.form

    if args.quiet:
        cli_overrides["quiet"] = True

    if args.verbose:
        cli_overrides["verbose"] = True

    if args.no_color:
        cli_overrides["no_color"] = True

    if args.dry_run:
        cli_overrides["dry_run"] = True

    if args.profile:
        cli_overrides["profile"] = True

    # Initialize config with proper priority: CLI > YAML > pyproject.toml > env > defaults
    config_file = Path(args.config) if args.config else None
    config = init_config(config_file=config_file, cli_overrides=cli_overrides)

    # Show dry-run message if enabled
    if config.dry_run:
        print("🔍 Dry-run mode: no files will be generated\n")

    # Set logging level based on config
    set_log_level(quiet=config.quiet, verbose=config.verbose)

    # Enable profiling if requested
    profiler = None
    if config.profile:
        profiler = enable_profiling()
        if not config.quiet:
            print("🔬 Performance profiling enabled\n")

    # Set random seed if provided
    seed = config.seed
    if seed is not None:
        set_global_seed(seed)
        if not config.quiet:
            print(f"\n{format_seed_message()}\n")
    else:
        # Generate and set a random seed for this session
        seed = set_global_seed()
        if not config.quiet:
            print(f"\n{format_seed_message()}\n")

    # Store seed in profiler if profiling is enabled
    if profiler:
        profiler.set_metadata("seed", seed)

    # Show output directory if specified
    if args.out and not config.quiet:
        print(f"📁 Output directory: {config.output_dir}\n")

    # Show output format if specified
    if args.format and not config.quiet:
        print(f"📄 Output format: {config.output_format}\n")

    # Show spaCy model if specified
    if args.spacy_model and not config.quiet:
        print(f"🧠 spaCy model: {config.spacy_model.value}\n")

    # Create and configure menu
    menu = ConsoleMenu("Generative Poetry Menu", "What kind of poem would you like to generate?")

    # Original generation items
    futurist_function_item = FunctionItem("Futurist Poem (PDF/Image)", futurist_poem_action)
    markov_function_item = FunctionItem(
        "Stochastic Jolatic (Markov) Poem (Image)", markov_poem_action
    )
    chaotic_concrete_function_item = FunctionItem(
        "Chaotic Concrete Poem (Image)", chaotic_concrete_poem_action
    )
    character_soup_function_item = FunctionItem(
        "Character Soup Poem (Image)", character_soup_poem_action
    )
    stopword_soup_function_item = FunctionItem(
        "Stop Word Soup Poem (Image)", stopword_soup_poem_action
    )
    simple_visual_function_item = FunctionItem(
        "Visual Puzzle Poem (Terminal-Based)", visual_puzzle_poem_action
    )

    # New ideation items
    line_seeds_item = FunctionItem("🌱 Generate Line Seeds (Poetry Ideation)", line_seeds_action)
    metaphor_item = FunctionItem(
        "🔮 Generate Metaphors (Poetry Ideation)", metaphor_generator_action
    )
    corpus_item = FunctionItem("📊 Analyze Personal Poetry Corpus", corpus_analyzer_action)
    transformer_item = FunctionItem("🚢 Ship of Theseus Poem Transformer", poem_transformer_action)
    idea_item = FunctionItem(
        "💡 Poetry Idea Generator (Beat Writer's Block)", idea_generator_action
    )
    six_degrees_item = FunctionItem(
        "🔗 Six Degrees - Word Convergence Explorer", six_degrees_action
    )
    causal_poetry_item = FunctionItem(
        "🔍 Resonant Fragment Miner (Literary Discovery)", causal_poetry_action
    )
    equidistant_item = FunctionItem(
        "🎯 Equidistant Word Finder (Bridge Discovery)", equidistant_action
    )
    semantic_geodesic_item = FunctionItem(
        "🌉 Semantic Geodesic Finder (Transitional Paths)", semantic_geodesic_action
    )
    conceptual_cloud_item = FunctionItem(
        "🌥️  Conceptual Cloud Generator (Word Associations)", conceptual_cloud_action
    )

    # Syllable-constrained form generators
    haiku_item = FunctionItem("🌸 Generate Haiku (5-7-5 syllables)", haiku_action)
    tanka_item = FunctionItem("🎋 Generate Tanka (5-7-5-7-7 syllables)", tanka_action)
    senryu_item = FunctionItem("🌿 Generate Senryu (5-7-5 syllables)", senryu_action)

    # System item
    check_deps_item = FunctionItem("Check System Dependencies", check_dependencies_action)

    menu.append_item(futurist_function_item)
    menu.append_item(markov_function_item)
    menu.append_item(chaotic_concrete_function_item)
    menu.append_item(character_soup_function_item)
    menu.append_item(stopword_soup_function_item)
    menu.append_item(simple_visual_function_item)
    menu.append_item(line_seeds_item)
    menu.append_item(metaphor_item)
    menu.append_item(corpus_item)
    menu.append_item(transformer_item)
    menu.append_item(idea_item)
    menu.append_item(six_degrees_item)
    menu.append_item(causal_poetry_item)
    menu.append_item(equidistant_item)
    menu.append_item(semantic_geodesic_item)
    menu.append_item(conceptual_cloud_item)
    menu.append_item(haiku_item)
    menu.append_item(tanka_item)
    menu.append_item(senryu_item)
    menu.append_item(check_deps_item)

    menu.start()
    menu.join()

    # Print profiling report if profiling was enabled
    if profiler:
        profiler.print_report()

    # Echo seed at exit
    print(f"\n{format_seed_message()}")
    print("Session complete.\n")


if __name__ == "__main__":
    main()
