"""
Wrapper for LLM multiple-choice queries
"""

import dataclasses
import textwrap
from typing import Any

from openai import APITimeoutError

from drugmechcf.text.optsmatcher import OptionsMatcher
from drugmechcf.llm.openai import OpenAICompletionOpts, OpenAICompletionClient


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

DEBUG = False


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class MultipleChoiceResponseInfo:

    prompt: str

    propmt_vars: dict[str, str]

    llm_response: str = None
    llm_stop_reason: str = None

    selected_option: str | None = None
    """Best matching option's Key from `IXN_TYPE_OPTIONS`."""

    match_score: float = None
    """Match-score for `ixn_type`."""

    response_summary: str = None
    """The excerpt from larger LLM response that was used to match to the options."""

    def is_complete_response(self) -> bool:
        return self.llm_stop_reason == "stop"

    def asdict(self) -> dict[str, Any]:
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

# /


class MultipleChoiceQuery:
    """
    This class uses the string.format() function to replace variables (mentioned in braces) with values.
    Trivial example: "String with {var}".format(var="value") = "String with value"
    """

    def __init__(self,
                 response_options: dict[str, str],
                 prompt_options_var: str = "options",
                 *,
                 default_opt: str = None,
                 llm_client: OpenAICompletionClient = None,
                 llm_opts: OpenAICompletionOpts = None,
                 timeout_secs: int = None,
                 ):
        """

        :param response_options: Dict: Option-Key[str] => Option-Text[str]
            Option-Text may be a string.format-template with variables,
            whose values are supplied during `self.invoke()`.

        :param prompt_options_var: The variable used in the prompt_template (see `self.invoke()`) where
            all the options are enumerated for the LLM.

        :param default_opt: The default Option-Key to select when matches are below threshold.
            IF not provided THEN the last option in `response_options` is used.

        :param llm_client:
            At least one of `llm_client` or `llm_opts` must be provided.

        :param llm_opts:
            At least one of `llm_client` or `llm_opts` must be provided.

        :param timeout_secs: LLM call timeout.
            Default value is usually 10 minutes for OpenAPI. See `OpenAICompletionClient`.
        """

        assert llm_client is not None or llm_opts is not None, \
            "At least one of `llm_client` or `llm_opts` must be specified."

        self.response_options = response_options

        # This is built in `build_prompt`
        self.opts_matcher = None

        self.prompt_options_var = prompt_options_var

        if default_opt is not None:
            assert default_opt in self.response_options, \
                f"Default option '{default_opt}' must be a key in the provided `response_options`."

            self.default_opt = default_opt
        else:
            # Select last option as the default
            self.default_opt = list(self.response_options.keys())[-1]

        self._llm_client = llm_client
        self.llm_opts = llm_opts
        self.timeout_secs = timeout_secs

        self.show_full_prompt = False
        self.show_response = False

        return

    @property
    def llm_client(self):
        return self._llm_client

    def build_prompt(self,
                     prompt_template: str,
                     prompt_vars: dict[str, str],
                     ) -> tuple[str, dict[str, str]]:

        # There might be variables in the options
        response_options = {key: "\n".join(textwrap.wrap(opt.format(**prompt_vars), width=90))
                            for key, opt in self.response_options.items()}

        self.opts_matcher = OptionsMatcher(response_options)

        multiple_options = "\n\n".join("* " + opt for opt in response_options.values())

        prompt_vars = prompt_vars.copy()
        prompt_vars[self.prompt_options_var] = multiple_options

        prompt = prompt_template.format(**prompt_vars)

        return prompt, prompt_vars

    def invoke(self,
               prompt_template: str,
               prompt_vars: dict[str, str],
               heading: str,
               min_score: float = 0.4,
               query_name: str = None,
               llm_verbose: bool = False,
               ) -> MultipleChoiceResponseInfo:
        """
        Asks LLM using `prompt` to select the best matching option among those provided.

        :param prompt_template: String.format template with variables (in braces), for the multiple-choice query.
            The prompt asks describes the query, asks the LLM for a response, and finally to summarize the answer
            by outputting a heading line followed by the chosen option.

            Required variable: self.prompt_options_var.
            Optional variables: `prompt_vars`

        :param prompt_vars: Dict: prompt-variable[str] => value[str]
            Values for the Optional variables mentioned above.

        :param heading: The final heading under which the LLM is asked to provide the chosen option.

        :param min_score: If the LLM's selected option does not match any of the available options
            with a score >= min_score, then the default option is chosen.

        :param query_name: Used in output trace, if `self.show_full_prompt = True`.

        :param llm_verbose: Whether call to LLM shows full response struct

        :return: MultipleChoiceResponseInfo with all fields filled in.
            IF APITimeoutError or Incomplete-response from LLM
            THEN only first 2 or 3 fields are filled in.
        """

        assert 0 < min_score < 1.0

        if self._llm_client is None:
            self._llm_client = OpenAICompletionClient(self.llm_opts, timeout_secs=self.timeout_secs)

        if query_name:
            query_name = f" for '{query_name}'"
        else:
            query_name = ""

        qprompt, prompt_vars = self.build_prompt(prompt_template, prompt_vars)

        response_info = MultipleChoiceResponseInfo(prompt=qprompt, propmt_vars=prompt_vars)

        if self.show_full_prompt:
            print(f"--- Prompt multiple-choice{query_name}:")
            print("   ", qprompt)
            print("---")
            print()

        try:
            llm_response = self._llm_client(user_prompt=qprompt, verbose=llm_verbose)

        except APITimeoutError:
            response_info.llm_stop_reason = "APITimeoutError"
            if self.show_response:
                print("--- LLM response:")
                print("    Request timed out")
                print()
            return response_info

        if self.show_response:
            print("--- LLM response:")
            print("    response complete =", llm_response.is_complete_response())
            print()
            print(llm_response.message)
            print("---")
            print()

        response_info.llm_response = llm_response.message
        response_info.llm_stop_reason = llm_response.finish_reason

        if not llm_response.is_complete_response():
            return response_info

        resp_sumry = self.opts_matcher.get_option_para(llm_response.message, heading=heading, strip_markup=True)

        if DEBUG:
            print(f"Extracted Summary:  [{resp_sumry}]", "", sep="\n")

        selected_opt, match_score = self.opts_matcher.get_best_matching_option(resp_sumry,
                                                                               min_score=min_score,
                                                                               warn_on_low_score=min_score)

        if selected_opt is None:
            selected_opt = self.default_opt

        response_info.response_summary = resp_sumry
        response_info.selected_option = selected_opt
        response_info.match_score = match_score

        return response_info

# /
