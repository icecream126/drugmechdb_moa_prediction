"""
Extracting a Summary selected option from LLM response,
and matching it to pre-defined list of options.
"""
import itertools
import re
from typing import List, Tuple, Union

import numpy as np

from drugmechcf.text.tfidfmatcher import TfdfParams, TfIdfMatchHelper


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

DEBUG = False


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class OptionsMatcher:
    """
    Example:
        >>> opts = {"Positive": ("A change (increase or decrease) in the levels of {name1} causes"
        >>>                      " the same change in the activity of {name2}."),
        >>>            "Negative": ("A change (increase or decrease) in the levels of {name1} causes"
        >>>                         " an opposite change in the activity of {name2}."),
        >>>            "NoEffect": "There is no evidence of any effect."
        >>>            }
        >>> opts_matcher = OptionsMatcher(opts)
        >>> llm_response: str = ...
        >>> llm_summary = opts_matcher.get_final_summary_para(llm_response,
        >>>                                                   min_len=int(min(map(len, opts.values())) * 0.8))
        >>> best_matched_opt, best_match_score = opts_matcher.get_best_matching_option(llm_summary,
        >>>                                              text_substitutions=[('FOXP3', '{name1}'), ('MYC', '{name2}')])
    """

    DEFAULT_TVZR_PARAMS = {
        "char_params": TfdfParams(ngram_widths_minmax=(2, 5)),
        "word_params": None,
    }

    def __init__(self,
                 response_options: dict[str, str],
                 tvzr_params: Union[str, dict[str, TfdfParams]] = None
                 ):
        """

        :param response_options: Dict[ OptionKey[str]: OptionText[str] ]
            Lists all the OptionText entries to match against.
            The OptionKey is returned for the matching entry.
            Simple example:
            {'One': 'This is option one.',
             'Two': This is option two.'
             }

        :param tvzr_params: Dict[ {"char_params", "word_params"} => TfdfParams ]
            `DEFAULT_TVZR_PARAMS` are used if None provided.
        """

        if tvzr_params is None:
            tvzr_params = self.DEFAULT_TVZR_PARAMS

        self._validate_data(response_options)

        self.opts_matcher = TfIdfMatchHelper("Options Matcher", tvzr_params)
        for opt_key, opt_text in response_options.items():
            self.opts_matcher.add_name("opt", opt_key, opt_text, opt_text.casefold())

        self.opts_matcher.build()

        self.default_opt_minlen = int(min(map(len, response_options.values())) * 0.8)
        self.default_opt_maxlen = int(max(map(len, response_options.values())) * 1.4)

        return

    def _validate_data(self, response_options: dict[str, str]):
        """
        Sets `self.options_texts_lower`, and
            ensures that no option is a prefix of any other.
        ELSE raises an exception
        """
        self.options_texts_lower = [t.lower() for t in response_options.values()]

        for (ki, a), (kj, b) in itertools.combinations(zip(response_options.keys(), self.options_texts_lower), 2):
            if a.startswith(b):
                raise ValueError(f"Invalid data: entry '{ki}' = '{response_options[ki]}' is a prefix of"
                                 f"entry '{kj}' = '{response_options[kj]}'.")
            if b.startswith(a):
                raise ValueError(f"Invalid data: entry '{kj}' = '{response_options[kj]}' is a prefix of"
                                 f"entry '{ki}' = '{response_options[ki]}'.")

        return

    def get_final_summary_para(self,
                               text: str,
                               heading: str = "Summary",
                               min_len: int = None,
                               strip_markup=True
                               ) -> str:
        """
        Extract the final summary para, of length >= `min_len`. It should be the last para in `text`,
        possibly preceded by the `heading`.

        :param text:
        :param heading: IF None then do not look for heading.
        :param min_len: Min length of returned summary.
            Default is to use 80% of the min option value.
        :param strip_markup: Whether to remove markup ("*") chars from around lines.
        """
        if min_len is None:
            min_len = self.default_opt_minlen

        text_len = len(text)
        if text_len < min_len:
            return text

        text_summary = None
        if heading is not None:
            # Look for the `heading``, in reverse
            hdg_patt = re.compile(f"\\b{heading}\\b", re.IGNORECASE)
            for m in list(re.finditer(hdg_patt, text))[::-1]:
                if text_len - m.end() >= min_len:
                    text_summary = text[m.end() + 1:].strip()
                    break

        if text_summary is None:
            text_summary = extract_last_para(text, min_length=min_len)

        if strip_markup:
            text_summary = re.sub(r'^(\*\s*)+', '', text_summary, flags=re.MULTILINE)
            text_summary = re.sub(r'\*+$', '', text_summary, flags=re.MULTILINE)

        return text_summary

    def get_option_para(self,
                        text: str,
                        heading: str = "Summary",
                        strip_markup=True
                        ) -> str:
        """
        Extract the final summary para, of length >= `min_len`. It should be the last para in `text`,
        possibly preceded by the `heading`.

        :param text:
        :param heading: IF None then do not look for heading.
        :param strip_markup: Whether to remove markup ("*") chars from around lines.
        """
        min_len = self.default_opt_minlen
        max_len = self.default_opt_maxlen

        text_len = len(text)
        if text_len < min_len:
            return text

        text_summary = None
        if heading is not None:
            # Look for the `heading``, in reverse
            # hdg_patt = re.compile(f"^[# *]+\\s*{heading}[*]*:?", re.IGNORECASE | re.MULTILINE)
            hdg_patt = re.compile(f"^#*\\s*[*]*\\s*{heading}[*]*:?" , re.IGNORECASE | re.MULTILINE)

            for m in list(re.finditer(hdg_patt, text))[::-1]:
                if text_len - m.end() >= min_len:
                    text_summary = text[m.end() + 1:].strip()
                    break

        if DEBUG:
            print(f'[1] text_summary = "{text_summary}"')

        if text_summary is None:
            text_summary = extract_last_para(text, min_length=min_len)

        text_summary = "\n".join([x for x in [l.strip() for l in text_summary.splitlines()] if x != ""])

        if DEBUG:
            print(f'[2] text_summary = "{text_summary}"')

        if strip_markup:
            text_summary = re.sub(r'^(\*\s*)+', '', text_summary, flags=re.MULTILINE)
            text_summary = re.sub(r'\*+$', '', text_summary, flags=re.MULTILINE)

        text_summary = text_summary[:max_len]

        if DEBUG:
            print(f'[3] text_summary = "{text_summary}"')

        # IF text starts with the options text, then trim it
        text_summary_lower = text_summary.lower()
        matching_opt_lens = [len(t) for t in self.options_texts_lower if text_summary_lower.startswith(t)]
        if matching_opt_lens:
            text_summary = text_summary[:max(matching_opt_lens)]

        return text_summary

    def get_best_matching_option(self,
                                 text: str,
                                 text_substitutions: List[Tuple[str, str]] = None,
                                 min_score: float = 0.4,
                                 warn_on_low_score: float = 0.0
                                 ) -> Tuple[str | None, float]:
        """
        Determines which of the possible options best matches `text`

        :param text: The minimal text containing LLM's chosen option.
        :param text_substitutions: List[ (Old, New), ...]
            Replace `Old` with `New` in text before matching to possible options.
        :param min_score:
        :param warn_on_low_score: Whether to print warning when options-match-score is below `min_score`.

        :return: best_matched_option_key, best_match_score
            best_matched_option_key is None if best_match_score < min_score
        """
        # print("Substts:", text_substitutions, "", sep="\n")
        # print(f"Text: [{text}]\n")

        if text_substitutions:
            # Substitute larger str's first.
            # When `Old.1` is substr of `Old.2`, `Old.2` will be substituted first.
            for old, new in sorted(text_substitutions, key=lambda x: -len(x[0])):
                text = re.sub(f"\\b{re.escape(old)}\\b", new, text)

        # submit as batch of 1
        matched_opts, match_scores, _, _ = \
            self.opts_matcher.get_matching_concepts_batched([text.casefold()])[0]

        if isinstance(matched_opts, np.ndarray) and len(matched_opts) > 0:
            try:
                # Select the highest match
                best_matched_opt = str(matched_opts[0][1])
                best_match_score = float(match_scores[0])
            except Exception as e:
                print("*** Exception ***")
                print("matched_opts =", matched_opts)
                print("match_scores =", match_scores)
                print("******")
                raise e
        else:
            best_matched_opt = None
            best_match_score = 0

        if best_match_score < warn_on_low_score:
            print(f"OptsMatch Low Score: {best_match_score:.3f} for [{text}].")

        if best_match_score < min_score:
            return None, best_match_score

        return best_matched_opt, best_match_score
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def extract_first_para(text: str, min_length: int = 0) -> str:
    """
    Extract first para in `text`, with paras delimitted by empty lines.
    :param text:
    :param min_length:
    :return:
    """
    if len(text) < min_length:
        return text

    resp_lines = [line.strip() for line in text.strip().splitlines()]
    s_idx = 0
    sfx_len = 0
    while s_idx < len(resp_lines):
        sfx_len += len(resp_lines[s_idx]) + 1
        if len(resp_lines[s_idx]) == 0 and sfx_len > min_length:
            break

        s_idx += 1

    return "\n".join(resp_lines[:s_idx])


def extract_last_para(text: str, min_length: int = 0) -> str:
    """
    Extract last para in `text`, with paras delimitted by empty lines.
    :param text:
    :param min_length:
    :return:
    """
    if len(text) < min_length:
        return text

    resp_lines = [line.strip() for line in text.strip().splitlines()]
    s_idx = len(resp_lines) - 1
    sfx_len = 0
    while s_idx > 0:
        if len(resp_lines[s_idx]) == 0 and sfx_len > min_length:
            s_idx += 1
            break

        sfx_len += len(resp_lines[s_idx]) + 1
        s_idx -= 1

    return "\n".join(resp_lines[s_idx:])
