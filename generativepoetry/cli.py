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

    # System item
    check_deps_item = FunctionItem("Check System Dependencies", check_dependencies_action)

    menu.append_item(futurist_function_item)
    menu.append_item(markov_function_item)
    menu.append_item(chaotic_concrete_function_item)
    menu.append_item(character_soup_function_item)
    menu.append_item(stopword_soup_function_item)
    menu.append_item(simple_visual_function_item)
    menu.append_item(line_seeds_item)  # Add the new feature
    menu.append_item(check_deps_item)

    menu.start()
    menu.join()


if __name__ == '__main__':
    main()