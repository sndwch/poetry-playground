#!/usr/bin/env python3
import argparse
import os

from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

# Import the pronouncing patch first to suppress pkg_resources warning
import generativepoetry.pronouncing_patch  # noqa: F401
from generativepoetry.causal_poetry import ResonantFragmentMiner
from generativepoetry.config import config
from generativepoetry.corpus_analyzer import PersonalCorpusAnalyzer
from generativepoetry.idea_generator import IdeaType, PoetryIdeaGenerator
from generativepoetry.line_seeds import LineSeedGenerator, SeedType
from generativepoetry.logger import set_log_level
from generativepoetry.metaphor_generator import MetaphorGenerator, MetaphorType
from generativepoetry.pdf import (
    ChaoticConcretePoemPDFGenerator,
    CharacterSoupPoemPDFGenerator,
    FuturistPoemPDFGenerator,
    MarkovPoemPDFGenerator,
    StopwordSoupPoemPDFGenerator,
)
from generativepoetry.poem_transformer import PoemTransformer
from generativepoetry.poemgen import PoemGenerator, print_poem
from generativepoetry.seed_manager import format_seed_message, set_global_seed
from generativepoetry.setup_models import setup as setup_models
from generativepoetry.six_degrees import SixDegrees
from generativepoetry.system_utils import check_system_dependencies
from generativepoetry.utils import get_input_words

reuse_words_prompt = "\nType yes to use the same words again, Otherwise just hit enter.\n"


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
        if input() != 'yes':
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
        if input() != 'yes':
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
        print("\n" + "="*60)
        print(f"Metaphor Generation for: {', '.join(input_words)}")
        print("="*60 + "\n")

        # Generate metaphors
        print("Generating metaphors...")
        metaphors = generator.generate_metaphor_batch(input_words, count=15)

        # Try to extract some patterns from multiple Gutenberg texts
        print("Mining literary patterns from diverse sources...")
        gutenberg_patterns = generator.extract_metaphor_patterns(num_texts=5)

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
                for line in extended.text.split('\n'):
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
            for text_key, patterns in text_groups.items():
                if shown_count >= 3:
                    break
                source, target, sentence = patterns[0]  # Take first pattern from this text
                print(f"  • {source} like {target}")
                print(f"    From: \"{sentence[:80]}...\"")
                shown_count += 1

            if len(text_groups) > 1:
                print(f"    (Patterns from {len(text_groups)} different classic texts)")

        # Generate a synesthetic metaphor
        print("\nSYNESTHETIC (CROSS-SENSORY):")
        print("-" * 40)
        for word in input_words[:2]:
            synesthetic = generator.generate_synesthetic_metaphor(word)
            if synesthetic:
                print(f"  • {synesthetic.text}")

        print("\n" + "="*60)
        print("\nOptions:")
        print("  1. Generate new metaphors with same words")
        print("  2. Use different words")
        print("  3. Mine more Gutenberg texts")
        print("  4. Return to main menu")

        choice = input("\nYour choice (1-4): ").strip()

        if choice == '1':
            continue  # Regenerate with same words
        elif choice == '2':
            input_words = get_input_words()  # Get new words
        elif choice == '3':
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

                print(f"Found {len(patterns)} metaphorical patterns from {len(text_groups)} different texts")

                # Show examples from different texts
                shown_texts = 0
                for text_key, group_patterns in text_groups.items():
                    if shown_texts >= 3:
                        break
                    source, target, sentence = group_patterns[0]
                    print(f"  • {source} like {target}")
                    print(f"    From: \"{sentence[:80]}...\"")
                    shown_texts += 1

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
        print("\n" + "="*50)
        print(f"Line Seeds for: {', '.join(input_words)}")
        print("="*50 + "\n")

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

        print("\n" + "="*50)
        print("\nOptions:")
        print("  1. Generate new seeds with same words")
        print("  2. Use different words")
        print("  3. Return to main menu")

        choice = input("\nYour choice (1-3): ").strip()

        if choice == '1':
            continue  # Loop with same words
        elif choice == '2':
            input_words = get_input_words()  # Get new words
        else:
            exit_loop = True  # Exit


def corpus_analyzer_action():
    """Analyze a personal poetry corpus for style insights."""
    analyzer = PersonalCorpusAnalyzer()

    print("\n" + "="*60)
    print("PERSONAL CORPUS ANALYZER")
    print("="*60)
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

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        # Generate and display the style report
        report = analyzer.generate_style_report(fingerprint)
        print(report)

        # Generate inspiration report
        print("\n" + "="*60)
        print("GENERATING CREATIVE INSPIRATIONS...")
        print("="*60)
        inspiration_report = analyzer.generate_inspiration_report(fingerprint)
        print(inspiration_report)

        # Provide expansion suggestions
        print("\n" + "="*60)
        print("CREATIVE EXPANSION SUGGESTIONS")
        print("="*60)
        suggestions = analyzer.suggest_expansions(fingerprint)

        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"\n{i}. {suggestion}")
        else:
            print("\nNo specific suggestions at this time.")
            print("Your style shows good balance and variety!")

        # Vocabulary insights for ideation
        print("\n" + "="*60)
        print("IDEATION INSIGHTS")
        print("="*60)

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

        print("\n" + "="*60)
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

    print("\n" + "="*60)
    print("SHIP OF THESEUS POEM TRANSFORMER")
    print("="*60)
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
        for i, (title, path) in enumerate(poems, 1):
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
            poem_text,
            num_passes=num_passes,
            words_per_pass=words_per_pass
        )

        if transformations:
            # Generate and display the report
            print("\n" + "="*60)
            print("TRANSFORMATION COMPLETE")
            print("="*60)

            report = transformer.generate_transformation_report(transformations)
            print(report)

            # Offer to save the result
            save_choice = input("\nSave transformed poem? (y/n): ").strip().lower()
            if save_choice == 'y':
                output_filename = f"{selected_title}_transformed.txt"
                output_path = os.path.join(directory, output_filename)

                with open(output_path, 'w', encoding='utf-8') as f:
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

    print("\n" + "="*60)
    print("POETRY IDEA GENERATOR")
    print("="*60)
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
            print("\n" + "="*60)
            print("IDEAS GENERATED")
            print("="*60)

            # Generate and display the report
            report = generator.generate_idea_report(collection)
            print(report)

            # Interactive options
            print("\n" + "="*60)
            print("OPTIONS:")
            print("1. Generate more ideas")
            print("2. Focus on a different category")
            print("3. Get random selection for immediate use")
            print("4. Return to main menu")

            choice = input("\nYour choice (1-4): ").strip()

            if choice == '1':
                # Generate more ideas
                more_ideas = generator.generate_ideas(10, preferred_types)
                print(f"\nGenerated {more_ideas.total_count()} additional ideas:")
                additional_report = generator.generate_idea_report(more_ideas)
                print(additional_report)

            elif choice == '2':
                # Recursive call with different category
                idea_generator_action()
                return

            elif choice == '3':
                # Random selection for immediate use
                random_ideas = collection.get_random_mixed_selection(5)
                print("\nRANDOM SELECTION FOR IMMEDIATE USE:")
                print("-" * 50)
                for i, idea in enumerate(random_ideas, 1):
                    category_name = idea.idea_type.value.replace('_', ' ').title()
                    print(f"{i}. [{category_name}] \"{idea.text}\"")
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

    print("\n" + "="*60)
    print("SIX DEGREES - WORD CONVERGENCE EXPLORER")
    print("="*60)
    print("\nDiscover the hidden pathways between words!")
    print("This explores how two words can connect through semantic relationships,")
    print("like the Wikipedia phenomenon where you can reach any page through links.")

    while True:
        print("\n" + "-"*40)
        word_a = input("Enter first word (or 'exit' to return): ").strip()
        if word_a.lower() == 'exit':
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
        if input().lower() == 'exit':
            break


def causal_poetry_action():
    """Mine resonant fragments from classic literature"""
    miner = ResonantFragmentMiner()

    print("\n" + "="*60)
    print("RESONANT FRAGMENT MINER")
    print("="*60)
    print("\nDiscovers evocative sentence fragments from classic literature")
    print("that could serve as seeds for compression poetry. Mines multiple")
    print("sentence patterns for fragments with poetic weight and causal resonance.")

    while True:
        print("\n" + "-"*40)
        print("1. Mine 50 fragments (recommended)")
        print("2. Mine 100 fragments")
        print("3. Mine with custom count")
        print("4. Return to main menu")

        choice = input("\nChoice (1-4): ").strip()

        if choice == '1':
            print("\n🔍 Mining 50 resonant fragments...")
            collection = miner.mine_fragments(target_count=50, num_texts=5)
            print("\n" + miner.format_fragment_collection(collection))

        elif choice == '2':
            print("\n🔍 Mining 100 resonant fragments...")
            collection = miner.mine_fragments(target_count=100, num_texts=8)
            print("\n" + miner.format_fragment_collection(collection))

        elif choice == '3':
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

        elif choice == '4':
            break
        else:
            print("Invalid choice. Please enter 1-4.")


def main():
    """Main entry point for the generative poetry CLI.

    Parses command-line arguments and launches the interactive menu.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generative Poetry - Procedural poetry generation and ideation tools',
        epilog='For more information, visit: https://github.com/sndwch/generativepoetry-py'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        metavar='INT',
        help='Random seed for deterministic/reproducible generation (example: --seed 42)'
    )

    # Output control
    parser.add_argument(
        '--out', '-o',
        type=str,
        metavar='PATH',
        help='Output directory for generated files (default: current directory)'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['png', 'pdf', 'svg', 'txt'],
        metavar='FORMAT',
        help='Output format: png, pdf, svg, or txt (default: pdf)'
    )

    # CLI behavior
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output (show only warnings and errors)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output including debug information'
    )

    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output for better compatibility'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview actions without generating files'
    )

    # Utility commands
    parser.add_argument(
        '--list-fonts',
        action='store_true',
        help='List available fonts and exit'
    )

    parser.add_argument(
        '--list-procedures',
        action='store_true',
        help='List available poem generation procedures and exit'
    )

    parser.add_argument(
        '--setup',
        action='store_true',
        help='Download and install all required NLTK data and spaCy models, then exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='generativepoetry 0.3.4'
    )

    args = parser.parse_args()

    # Handle --list-fonts command
    if args.list_fonts:
        print("Available fonts for PDF generation:\n")
        # List standard PDF fonts (always available with reportlab)
        standard_fonts = [
            'Courier', 'Courier-Bold', 'Courier-BoldOblique', 'Courier-Oblique',
            'Helvetica', 'Helvetica-Bold', 'Helvetica-BoldOblique', 'Helvetica-Oblique',
            'Times-Roman', 'Times-Bold', 'Times-Italic', 'Times-BoldItalic',
            'Symbol', 'ZapfDingbats'
        ]

        print("Standard PostScript Fonts (always available):")
        for i, font in enumerate(standard_fonts, 1):
            print(f"  {i:2}. {font}")

        # Try to list registered fonts from pdfmetrics
        try:
            from reportlab.pdfbase import pdfmetrics
            registered = pdfmetrics.getRegisteredFontNames()
            if registered and len(registered) > len(standard_fonts):
                print("\nAdditionally Registered Fonts:")
                custom = [f for f in registered if f not in standard_fonts]
                for i, font in enumerate(sorted(custom), 1):
                    print(f"  {i:2}. {font}")
        except Exception:
            pass

        print("\nNote: Custom fonts can be registered using reportlab.pdfbase.pdfmetrics")
        return

    # Handle --list-procedures command
    if args.list_procedures:
        print("Available poem generation procedures:\n")
        print("Visual/Concrete Poetry:")
        print("  1. Futurist Poem - Marinetti-inspired mathematical word connections")
        print("  2. Stochastic Jolastic (Markov) - Joyce-like wordplay with rhyme schemes")
        print("  3. Chaotic Concrete - Abstract spatial arrangements")
        print("  4. Character Soup - Pure visual typographic experiments")
        print("  5. Stop Word Soup - Minimalist word placement")
        print("  6. Visual Puzzle - Interactive terminal-based concrete poetry")
        print("\nPoetry Ideation Tools:")
        print("  7. Line Seeds Generator - Opening lines and pivotal fragments")
        print("  8. Metaphor Generator - AI-assisted metaphor creation")
        print("  9. Corpus Analyzer - Analyze personal poetry collection")
        print("  10. Ship of Theseus - Transform existing poems")
        print("  11. Poetry Idea Generator - Mine classic literature for creative seeds")
        print("  12. Six Degrees - Explore connections between concepts")
        print("  13. Resonant Fragment Miner - Extract poetic fragments from Gutenberg")
        print("\nUse generative-poetry-cli to access all procedures interactively.")
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
            print("\nYou can now run generative-poetry-cli to start creating poetry.")
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

    # Apply CLI flags to config
    if args.out:
        from pathlib import Path
        config.output_dir = Path(args.out).resolve()
        config.output_dir.mkdir(parents=True, exist_ok=True)

    if args.format:
        config.output_format = args.format

    if args.dry_run:
        config.dry_run = True
        print("🔍 Dry-run mode: no files will be generated\n")

    # Set logging level based on quiet/verbose
    set_log_level(quiet=args.quiet, verbose=args.verbose)

    if args.no_color:
        config.no_color = True
        # TODO: Implement color suppression in output functions

    # Set random seed if provided (via CLI arg or environment variable)
    seed = args.seed or config.seed
    if seed is not None:
        set_global_seed(seed)
        if not args.quiet:
            print(f"\n{format_seed_message()}\n")
    else:
        # Generate and set a random seed for this session
        seed = set_global_seed()
        if not args.quiet:
            print(f"\n{format_seed_message()}\n")

    # Show output directory if specified
    if args.out and not args.quiet:
        print(f"📁 Output directory: {config.output_dir}\n")

    # Show output format if specified
    if args.format and not args.quiet:
        print(f"📄 Output format: {config.output_format}\n")

    # Create and configure menu
    menu = ConsoleMenu("Generative Poetry Menu", "What kind of poem would you like to generate?")

    # Original generation items
    futurist_function_item = FunctionItem("Futurist Poem (PDF/Image)", futurist_poem_action)
    markov_function_item = FunctionItem("Stochastic Jolatic (Markov) Poem (Image)", markov_poem_action)
    chaotic_concrete_function_item = FunctionItem("Chaotic Concrete Poem (Image)", chaotic_concrete_poem_action)
    character_soup_function_item = FunctionItem("Character Soup Poem (Image)", character_soup_poem_action)
    stopword_soup_function_item = FunctionItem("Stop Word Soup Poem (Image)", stopword_soup_poem_action)
    simple_visual_function_item = FunctionItem("Visual Puzzle Poem (Terminal-Based)", visual_puzzle_poem_action)

    # New ideation items
    line_seeds_item = FunctionItem("🌱 Generate Line Seeds (Poetry Ideation)", line_seeds_action)
    metaphor_item = FunctionItem("🔮 Generate Metaphors (Poetry Ideation)", metaphor_generator_action)
    corpus_item = FunctionItem("📊 Analyze Personal Poetry Corpus", corpus_analyzer_action)
    transformer_item = FunctionItem("🚢 Ship of Theseus Poem Transformer", poem_transformer_action)
    idea_item = FunctionItem("💡 Poetry Idea Generator (Beat Writer's Block)", idea_generator_action)
    six_degrees_item = FunctionItem("🔗 Six Degrees - Word Convergence Explorer", six_degrees_action)
    causal_poetry_item = FunctionItem("🔍 Resonant Fragment Miner (Literary Discovery)", causal_poetry_action)

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
    menu.append_item(check_deps_item)

    menu.start()
    menu.join()

    # Echo seed at exit
    print(f"\n{format_seed_message()}")
    print("Session complete.\n")


if __name__ == '__main__':
    main()
