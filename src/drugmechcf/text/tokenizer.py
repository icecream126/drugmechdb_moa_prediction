"""
String tokenizer
"""

import re
from typing import List

from nltk import WordNetLemmatizer, PorterStemmer


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class BasicTokenizer:

    NAME_WORD_STYLES = ["full", "lemmatized", "stemmed"]

    NONWORD_CHAR_PATT = re.compile(r"(?:\W|_)")

    GREEK_CHARS_TRANS_TABLE = str.maketrans(dict([('α', 'alpha'),
                                                  ('β', 'beta'),
                                                  ('γ', 'gamma'),
                                                  ('δ', 'delta'),
                                                  ('ε', 'epsilon'),
                                                  ('Α', 'Alpha'),
                                                  ('Β', 'Beta'),
                                                  ('Γ', 'Gamma'),
                                                  ('Δ', 'Delta'),
                                                  ('Ε', 'Epsilon')]))

    def __init__(self, word_style: str = "full"):

        assert word_style in self.NAME_WORD_STYLES, f"Invalid value for name_word_style={word_style}."
        self.word_style = word_style

        self.stemmer = None
        self.lemmatizer = None
        if self.word_style == "lemmatized":
            self.lemmatizer = WordNetLemmatizer()   # call as *.lemmatize(word)
        elif self.word_style == "stemmed":
            self.stemmer = PorterStemmer()          # call as *.stem(word)

        return

    def tokenize_fullwords(self, txt: str, to_lower: bool = True) -> List[str]:
        """
        Tokenize `txt` without Lemmatizing or Stemming.
        IF `to_lower` is True THEN convert `txt` to lower-case first ELSE do not change case.

        Example usage is for Acronyms, where you do not want any word contraction:
            acronym_tokens = tknzr.tokenize_fullwords(acronym, to_lower=False)

        """
        txt = self.NONWORD_CHAR_PATT.sub(" ", txt)
        txt = txt.translate(self.GREEK_CHARS_TRANS_TABLE)

        if to_lower:
            txt = txt.casefold()

        return txt.split()

    def tokenize(self, txt: str) -> List[str]:
        """
        Tokenize `txt` in lower-Case and with all the normalization specified in __init__()
        """
        tkns = self.tokenize_fullwords(txt, to_lower=True)
        if self.stemmer:
            tkns = [self.stemmer.stem(w) for w in tkns]
        elif self.lemmatizer:
            tkns = [self.lemmatizer.lemmatize(w) for w in tkns]

        return tkns
# /
