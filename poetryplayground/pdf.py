import os
import random
import string
from os.path import isfile
from typing import ClassVar, List, Optional, Tuple

from nltk.corpus import stopwords
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas

from .lexigen import phonetically_related_words
from .poemgen import PoemGenerator
from .setup_models import lazy_ensure_nltk_data
from .system_utils import check_poppler_installed, get_poppler_install_instructions
from .utils import filter_word_list, get_input_words, get_random_color

rgb_tuple = Tuple[float]


class VisualPoemString:
    """The text drawn by reportlab at an XY coordinate--can be a line, a word, or just a character."""

    def __init__(
        self, text, x: int, y: int, font: str, font_size: int, rgb: Optional[Tuple] = None
    ):
        self.text = text
        self.x = x
        self.y = y
        self.font = font
        self.font_size = font_size
        self.rgb = rgb


class PDFGenerator:
    default_font_sizes: ClassVar[List[int]] = [12, 14, 16, 18, 24, 32]
    # Use only built-in fonts that don't require external TTF files
    font_choices: ClassVar[List[str]] = [
        "Courier",
        "Courier-Bold",
        "Courier-BoldOblique",
        "Courier-Oblique",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-BoldOblique",
        "Helvetica-Oblique",
        "Times-Bold",
        "Times-BoldItalic",
        "Times-Italic",
        "Times-Roman",
    ]

    def __init__(self):
        # Skip font registration - use built-in fonts only
        # The font_choices list above contains only standard PDF fonts
        self.orientation = "landscape"
        self.drawn_strings: List[VisualPoemString] = []

    def get_font_size(self, line):
        if len(line) > 30:
            return 16
        elif len(line) >= 24:
            return random.choice([16, 18, 20])
        else:
            return random.choice(self.default_font_sizes)

    def get_max_x_coordinate(self, line, font_choice, font_size):
        if (
            (font_size >= 23 and len(line) >= 17)
            or (font_size >= 20 and len(line) > 30)
            or font_choice.startswith("Courier")
        ):  # Courier is the widest
            return 30 if self.orientation == "portrait" else 60
        elif (
            (font_size == 23 and len(line) > 14)
            or (font_size >= 20 and len(line) > 16)
            or len(line) > 20
        ):
            return 100 if self.orientation == "portrait" else 130
        else:
            return 250 if self.orientation == "portrait" else 280

    def set_filename(self, input_words, file_extension="pdf"):
        sequence = ""
        filename = f"{','.join(input_words)}{sequence}.{file_extension}"
        while isfile(filename):
            sequence = int(sequence or 0) + 1
            filename = f"{','.join(input_words)}({sequence}).{file_extension}"
        self.pdf_filepath = os.getcwd() + "/" + filename
        return filename

    def generate_png(self, input_filepath=None):
        """Generate PNG from PDF file.

        Args:
            input_filepath: Path to input PDF file

        Returns:
            bool: True if successful, False otherwise
        """
        if not check_poppler_installed():
            from .logger import logger

            logger.warning("PNG generation skipped - Poppler not installed")
            logger.info(get_poppler_install_instructions())
            return False

        try:
            pages = convert_from_path(input_filepath)
            for page in pages:
                page.save(f"{input_filepath[:-3]}png", "PNG")
            from .logger import logger

            logger.info(f"PNG generated: {input_filepath[:-3]}png")
            return True
        except PDFInfoNotInstalledError:
            from .logger import logger

            logger.error("Poppler's pdfinfo utility not found in PATH")
            logger.info("Even though Poppler was detected, pdfinfo is missing")
            logger.info(get_poppler_install_instructions())
            return False
        except PDFPageCountError as e:
            from .logger import logger

            logger.error(f"Failed to get PDF page count: {e}")
            logger.info("The PDF file may be corrupted or invalid")
            return False
        except PDFSyntaxError as e:
            from .logger import logger

            logger.error(f"PDF syntax error: {e}")
            logger.info("The PDF file appears to be malformed")
            return False
        except FileNotFoundError:
            from .logger import logger

            logger.error(f"PDF file not found: {input_filepath}")
            return False
        except Exception as e:
            from .logger import logger

            logger.error(f"Unexpected error generating PNG: {e}")
            logger.info("If the problem persists, please report this issue")
            return False


class ChaoticConcretePoemPDFGenerator(PDFGenerator):
    def generate_pdf(self, input_words: Optional[List[str]] = None, max_words=Optional[int]):
        self.drawn_strings = []
        if input_words is None:
            input_words = []
        input_words = get_input_words() if not len(input_words) else input_words
        output_words = input_words + phonetically_related_words(input_words)
        random.shuffle(output_words)
        filename = self.set_filename(input_words)
        c = canvas.Canvas(filename)
        for word in output_words[:200]:
            word = random.choice([word, word, word, word.upper()])
            x = random.randint(15, 440)
            y = random.randint(15, 800)
            font_choice = random.choice(self.font_choices)
            font_size = random.choice(self.default_font_sizes)
            rgb = get_random_color()
            vp_string = VisualPoemString(
                word, x=x, y=y, font=font_choice, font_size=font_size, rgb=rgb
            )
            c.setFont(vp_string.font, vp_string.font_size)
            c.setFillColorRGB(*vp_string.rgb)
            c.drawString(vp_string.x, vp_string.y, vp_string.text)
            self.drawn_strings.append(vp_string)
        c.showPage()
        c.save()


class CharacterSoupPoemPDFGenerator(PDFGenerator):
    def generate_pdf(self):
        c = canvas.Canvas("character_soup.pdf")
        for _i in range(20):
            char_sequence = random.choice(
                [string.ascii_lowercase, string.digits, string.punctuation]
            )
            for char in char_sequence:
                char = (
                    random.choice([char, char, char.upper(), char.upper()])
                    if char_sequence == string.ascii_lowercase
                    else char
                )
                font_choice = random.choice(self.font_choices)
                font_size = random.randint(6, 72)
                x = random.randint(10, 560)
                y = random.randint(10, 790)
                rgb = get_random_color()
                vp_string = VisualPoemString(
                    char, x=x, y=y, font=font_choice, font_size=font_size, rgb=rgb
                )
                c.setFillColorRGB(*rgb)
                c.setFont(vp_string.font, vp_string.font_size)
                c.drawString(vp_string.x, vp_string.y, vp_string.text)
                self.drawn_strings.append(vp_string)
        c.showPage()
        c.save()


class StopwordSoupPoemPDFGenerator(PDFGenerator):
    default_font_sizes: ClassVar[List[int]] = [6, 16, 24, 32, 48]

    def generate_pdf(self):
        # Ensure stopwords corpus is available
        lazy_ensure_nltk_data("corpora/stopwords", "stopwords", "Stopwords corpus")

        c = canvas.Canvas("stopword_soup.pdf")
        punctuation = list(string.punctuation)
        words_to_use = (
            filter_word_list(stopwords.words("english")) + punctuation
        )  # filter removes 1/2s of contractions
        for word in ["him", "her", "his", "they", "won"]:
            words_to_use.remove(word)
        for _i in range(3):
            words_to_use.extend(
                ["hmm", "ah", "umm", "uh", "ehh.", "psst..", "what?", "oh?", "ahem.."]
            )
        random.shuffle(words_to_use)
        for word in words_to_use:
            word = random.choice([word, word, word.upper(), word.upper()])
            font_choice = random.choice(self.font_choices)
            font_size = random.randint(6, 40)
            x = random.randint(10, 490)
            y = random.randint(10, 790)
            rgb = get_random_color(threshold=0.5)
            c.setFillColorRGB(*rgb)
            vp_string = VisualPoemString(
                word, x=x, y=y, font=font_choice, font_size=font_size, rgb=rgb
            )
            c.setFont(vp_string.font, vp_string.font_size)
            c.drawString(vp_string.x, vp_string.y, vp_string.text)
            self.drawn_strings.append(vp_string)
        c.showPage()
        c.save()


class MarkovPoemPDFGenerator(PDFGenerator):
    default_font_sizes: ClassVar[List[int]] = [15, 18, 21, 24, 28]

    def generate_pdf(
        self, input_words: Optional[List[str]] = None, orientation: string = "landscape"
    ):
        self.drawn_strings = []
        if input_words is None:
            input_words = []
        self.orientation = orientation
        if self.orientation.lower() == "landscape":
            num_lines = 14
            y_coordinate = 550
            min_line_words = 7
            max_line_words = 10
            max_line_length = 66
            min_x_coordinate = 60
        elif self.orientation.lower() == "portrait":
            num_lines = 24
            y_coordinate = 740
            min_line_words = 4
            max_line_words = 7
            max_line_length = 40
            min_x_coordinate = 15
        else:
            raise Exception("Must choose from the following orientations: portrait, landscape")
        input_words = get_input_words() if not len(input_words) else input_words
        poemgen = PoemGenerator()
        poem = poemgen.poem_from_markov(
            input_words=input_words,
            min_line_words=min_line_words,
            num_lines=num_lines,
            max_line_words=max_line_words,
            max_line_length=max_line_length,
        )
        font_choice, last_font_choice = None, None
        filename = self.set_filename(input_words)
        if orientation == "landscape":
            c = canvas.Canvas(filename, pagesize=landscape(letter))
        else:
            c = canvas.Canvas(filename)
        for line in poem.lines:
            random.choice([line, line, line, line.upper()])
            font_size = self.get_font_size(line)
            while font_choice is None or last_font_choice == font_choice:
                font_choice = random.choice(self.font_choices)
            max_x_coordinate = self.get_max_x_coordinate(line, font_choice, font_size)
            last_font_choice = font_choice
            x_coordinate = random.randint(min_x_coordinate, max_x_coordinate)
            vp_string = VisualPoemString(
                line, x=x_coordinate, y=y_coordinate, font=font_choice, font_size=font_size
            )
            c.setFont(vp_string.font, vp_string.font_size)
            c.drawString(vp_string.x, vp_string.y, vp_string.text)
            y_coordinate -= 32
            self.drawn_strings.append(vp_string)
        c.showPage()
        c.save()


class FuturistPoemPDFGenerator(PDFGenerator):
    connectors: ClassVar[List[str]] = [" + ", " - ", " * ", " % ", " = ", " != ", " :: "]
    default_font_sizes: ClassVar[List[int]] = [15, 18, 21, 24, 28]

    def generate_pdf(self, input_words: Optional[List[str]] = None):
        self.drawn_strings = []
        if input_words is None:
            input_words = []
        input_words = get_input_words() if not len(input_words) else input_words
        word_list = input_words + phonetically_related_words(input_words)
        poem_lines = []
        pgen = PoemGenerator()
        for _i in range(25):
            random.shuffle(word_list)
            poem_lines.append(
                pgen.poem_line_from_word_list(
                    word_list, connectors=self.connectors, max_line_length=40
                )
            )
        filename = self.set_filename(input_words)
        c = canvas.Canvas(filename)
        y_coordinate = 60
        for line in poem_lines:
            line = random.choice([line, line, line, line.upper()])
            font_choice = random.choice(self.font_choices)
            font_size = self.get_font_size(line)
            max_x_coordinate = self.get_max_x_coordinate(line, font_choice, font_size)
            x_coordinate = random.randint(15, max_x_coordinate)
            vp_string = VisualPoemString(
                line, x=x_coordinate, y=y_coordinate, font=font_choice, font_size=font_size
            )
            c.setFont(vp_string.font, vp_string.font_size)
            c.drawString(vp_string.x, vp_string.y, line)
            y_coordinate += 31
            self.drawn_strings.append(vp_string)
        c.showPage()
        c.save()
