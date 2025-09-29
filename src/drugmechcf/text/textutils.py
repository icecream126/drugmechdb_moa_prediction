"""
Misc text utils, can be used for normalizing names, before calling CharNodeMatcher
"""

from unidecode import unidecode
import unicodedata


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------


GREEK_CHARS_TO_STR = {
    'α': 'ALPHA',
    'Α': 'ALPHA',
    'β': 'BETA',
    'Β': 'BETA',
    'γ': 'GAMMA',
    'Γ': 'GAMMA',
    'δ': 'DELTA',
    'Δ': 'DELTA',
    'ε': 'EPSILON',
    'Ε': 'EPSILON',
    'ζ': 'ZETA',
    'Ζ': 'ZETA',
    'η': 'ETA',
    'Η': 'ETA',
    'θ': 'THETA',
    'Θ': 'THETA',
    'ι': 'IOTA',
    'Ι': 'IOTA',
    'κ': 'KAPPA',
    'Κ': 'KAPPA',
    'λ': 'LAMDA',
    'Λ': 'LAMDA',
    'μ': 'MU',
    'Μ': 'MU',
    'ν': 'NU',
    'Ν': 'NU',
    'ξ': 'XI',
    'Ξ': 'XI',
    'ο': 'OMICRON',
    'Ο': 'OMICRON',
    'π': 'PI',
    'Π': 'PI',
    'ρ': 'RHO',
    'Ρ': 'RHO',
    'σ': 'SIGMA',
    'Σ': 'SIGMA',
    'τ': 'TAU',
    'Τ': 'TAU',
    'υ': 'UPSILON',
    'φ': 'PHI',
    'Φ': 'PHI',
    'χ': 'CHI',
    'Χ': 'CHI',
    'ψ': 'PSI',
    'Ψ': 'PSI',
    'ω': 'OMEGA',
    'Ω': 'OMEGA',
}

ALL_GREEK_CHARS_TRANS_TABLE = str.maketrans(GREEK_CHARS_TO_STR)

# Translate all Unicode dash-like chars to stdd ASCII '-' (except 2 chars translate to '~')
ALL_DASHES_TO_DASH = {
    chr(45): '-',  # ... HYPHEN-MINUS, '-'
    chr(1418): '-',  # ... ARMENIAN HYPHEN, '֊'
    chr(1470): '-',  # ... HEBREW PUNCTUATION MAQAF, '־'
    chr(5120): '-',  # ... CANADIAN SYLLABICS HYPHEN, '᐀'
    chr(6150): '-',  # ... MONGOLIAN TODO SOFT HYPHEN, '᠆'
    chr(8208): '-',  # ... HYPHEN, '‐'
    chr(8209): '-',  # ... NON-BREAKING HYPHEN, '‑'
    chr(8210): '-',  # ... FIGURE DASH, '‒'
    chr(8211): '-',  # ... EN DASH, '–'
    chr(8212): '-',  # ... EM DASH, '—'
    chr(8213): '-',  # ... HORIZONTAL BAR, '―'
    chr(11799): '-',  # ... DOUBLE OBLIQUE HYPHEN, '⸗'
    chr(11802): '-',  # ... HYPHEN WITH DIAERESIS, '⸚'
    chr(11834): '-',  # ... TWO-EM DASH, '⸺'
    chr(11835): '-',  # ... THREE-EM DASH, '⸻'
    chr(11840): '-',  # ... DOUBLE HYPHEN, '⹀'
    chr(11869): '-',  # ... OBLIQUE HYPHEN, '⹝'
    chr(12316): '~',  # ... WAVE DASH, '〜'
    chr(12336): '~',  # ... WAVY DASH, '〰'
    chr(12448): '-',  # ... KATAKANA-HIRAGANA DOUBLE HYPHEN, '゠'
    chr(65073): '-',  # ... PRESENTATION FORM FOR VERTICAL EM DASH, '︱'
    chr(65074): '-',  # ... PRESENTATION FORM FOR VERTICAL EN DASH, '︲'
    chr(65112): '-',  # ... SMALL EM DASH, '﹘'
    chr(65123): '-',  # ... SMALL HYPHEN-MINUS, '﹣'
    chr(65293): '-',  # ... FULLWIDTH HYPHEN-MINUS, '－'
    chr(69293): '-',  # ... YEZIDI HYPHENATION MARK, '𐺭'
}

ALL_DASHES_TRANS_TABLE = str.maketrans(ALL_DASHES_TO_DASH)


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def standardize_chars_basic(text: str) -> str:
    # Separate combined chars, e.g. [ﬁ] in 'ﬁnancial' => 'fi...'
    text = unicodedata.normalize("NFKD", text)

    # Strip accents, e.g. [é] in 'Montréal' => 'Montreal'
    text = "".join([c for c in text if unicodedata.category(c) != "Mn"])

    return text


def standardize_chars_unidecode(text):
    # This seems to be a superset of `standardize_chars_basic`, and may be too aggressive for some use-cases.
    # E.g. this will convert 'μ-meter' to 'm-meter'
    # Strip(), as Standardization may add SPACE, e.g. standardize_chars_unidecode('北亰') = 'Bei Jing '
    text = unidecode(text).strip()
    return text


def translate_text(text: str) -> str:
    new_text = text.translate(ALL_GREEK_CHARS_TRANS_TABLE)
    return new_text
