""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re
import unicodedata

import phonemizer
from unidecode import unidecode

from matcha.text.numbers_de import normalize_numbers_de

# espeak-ng is a system dependency and cannot always be installed (e.g. on a compute cluster
# without root). If it is not available, fall back to the espeak-ng library and data shipped as a
# wheel by the optional `espeakng-loader` package. A system install - or an explicit override via
# PHONEMIZER_ESPEAK_LIBRARY - always takes precedence.
if not phonemizer.backend.EspeakBackend.is_available():
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except ImportError:
        pass

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


# Typographic variants that mean the same thing to a grapheme model. Everything on the left
# collapses onto a symbol that is in symbols.py; the fancy quotes are already there.
_grapheme_equivalents = [
    ("\u2019", "'"),  # right single quotation mark, the usual apostrophe in Swiss German text
    ("\u2018", "'"),
    ("\u201a", "'"),
    ("\u00b4", "'"),
    ("`", "'"),
    ("\u2013", "-"),  # en dash
    ("\u2014", "-"),  # em dash (kept distinct from the "—" ellipsis-like symbol Tacotron used)
    ("\u2011", "-"),  # non-breaking hyphen
    ("\u00a0", " "),  # no-break space
    ("\u00ad", ""),  # soft hyphen: invisible, only a line-break hint
    ("\u200b", ""),  # zero-width space
    ("ß", "ss"),  # Swiss Standard German has no ß; the corpora write "grossen", "heissen"
    ("%", " prozent"),
    ("&", " und "),
    ("/", " "),
]


def swiss_german_cleaners(text):
    """Character-level pipeline for Swiss German (or Standard German) text.

    Swiss German has no standard orthography and no espeak voice, so the text is *not*
    phonemised: the model learns grapheme-to-sound directly. That is why the umlauts are kept
    (no unidecode) and why the symbol set carries the German graphemes.

    Steps: Unicode NFC (so "ä" is one code point), spell out numbers in German (numbers_de.py),
    lowercase, fold typographic quotes/dashes and a few symbols onto plain ones, drop brackets,
    collapse whitespace. Anything still outside symbols.py will raise in text_to_sequence --
    run scripts/build_filelist.py first, it reports such characters.
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_numbers_de(text)
    text = lowercase(text)
    for src, dst in _grapheme_equivalents:
        text = text.replace(src, dst)
    text = remove_brackets(text)
    text = collapse_whitespace(text)
    return text.strip()


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def ipa_simplifier(text):
    replacements = [
        ("ɐ", "ə"),
        ("ˈə", "ə"),
        ("ʤ", "dʒ"),
        ("ʧ", "tʃ"),
        ("ᵻ", "ɪ"),
    ]
    for replacement in replacements:
        text = text.replace(replacement[0], replacement[1])
    phonemes = collapse_whitespace(text)
    return phonemes


# I am removing this due to incompatibility with several version of python
# However, if you want to use it, you can uncomment it
# and install piper-phonemize with the following command:
# pip install piper-phonemize

# import piper_phonemize
# def english_cleaners_piper(text):
#     """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
#     text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = expand_abbreviations(text)
#     phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
#     phonemes = collapse_whitespace(phonemes)
#     return phonemes
