#!/usr/bin/env python3
# Import the patch first to suppress pkg_resources warning before pronouncing is loaded
from generativepoetry.pronouncing_patch import *

from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem
from generativepoetry.pdf import *
from generativepoetry.poemgen import *
from generativepoetry.utils import get_input_words
from generativepoetry.system_utils import check_system_dependencies
from generativepoetry.line_seeds import LineSeedGenerator, SeedType
from generativepoetry.metaphor_generator import MetaphorGenerator, MetaphorType
from generativepoetry.corpus_analyzer import PersonalCorpusAnalyzer

reuse_words_prompt = "\nType yes to use the same words again, Otherwise just hit enter.\n"


def interactive_loop(poetry_generator):
    exit_loop = False
    input_words = get_input_words()
    while exit_loop == False:
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
    while exit_loop == False:
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
        gutenberg_patterns = generator.extract_metaphor_patterns(num_texts=3)

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
            for source, target, sentence in gutenberg_patterns[:3]:
                print(f"  • {source} like {target}")
                print(f"    From: \"{sentence[:80]}...\"")

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
            patterns = generator.extract_metaphor_patterns(num_texts=5)
            if patterns:
                print(f"Found {len(patterns)} metaphorical patterns from 5 diverse texts")
                # Show a few examples
                for source, target, sentence in patterns[:3]:
                    print(f"  • {source} like {target}")
                    print(f"    From: \"{sentence[:80]}...\"")
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
            print(f"\nThematic word groups you naturally use:")
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


def main():
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
    menu.append_item(corpus_item)  # Add the corpus analyzer
    menu.append_item(check_deps_item)

    menu.start()
    menu.join()


if __name__ == '__main__':
    main()