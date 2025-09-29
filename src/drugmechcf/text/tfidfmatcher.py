"""
Fuzzy String Match using char/word TF-IDF
"""

from collections import defaultdict
import dataclasses
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances, linear_kernel
import scipy.sparse as sp

# from utils.misc import ValidatedDataclass


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class TfdfParams:  # (ValidatedDataclass):
    """
    Parameters passed to TfidfVectorizer.
    """

    ngram_widths_minmax: Tuple[int, int] = (1, 1)
    """
    tuple (min_n, max_n) ... Ngrams of width [min_n, max_n] generated.
    """

    sim_weight: float = 1.0
    """
    When multiple (char, word) tf-idf, this is the coeff for wtd sum
    """

    # These reflect defaults in `TfidfVectorizer`
    stop_words: Union[str, List[str]] = None
    """
    List of words (str),
    or 'english' to use the built-in default in TfidfVectorizer, a list of 318 words.
    """

    max_features: int = None
    """
    If not None, build a vocabulary that only consider the top max_features
    ordered by term frequency across the corpus.
    """

    max_df: Union[int, float] = 1.0
    """
    Ignore terms that have a document frequency strictly higher than the given
    threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents,
    integer is an absolute count of nbr of documents.
    """

    min_df: Union[int, float] = 1
    """
    Ignore terms that have a document frequency strictly lower than the given threshold.
    This value is also called cut-off in the literature. If float, the parameter represents a
    proportion of documents, integer represents absolute count.
    """

    use_idf: bool = True
    """
    Enable inverse-document-frequency reweighting. If False, idf(t) = 1. Default is True.
    """

    normalize_tf_idf: bool = True
    """
    Meaning "l2" norm, which is the default
    """
# /


class TfIdfMatchHelper:
    """
    Helper class for matching concept mentions to concepts, that can use a combination of
    character n-gram and word n-gram based matching.

    Holds a reference dictionary of Concept names. Mentions are matched against this reference dictionary.
    Typical usage:
        # Initialize and Build
        tfdfm = TfIdfMatchHelper(...)
        for each concept:
            for each name in concept:
                tfdfm.add_name(...)

        tfdfm.build()
        ...
        # Use
        tfdfm.set_name_type_weights(...)    # Optional
        ...
        for mentions_batch in docs:
            matches = tfdfm.get_matching_concepts_batched(mentions_batch, ...)

    A Concept has the following properties:
    - A concept-type: str
    - A concept-id: str
    - A set of (original-name: str, normalized-name: str, name_type: int) defining alternative names and their types.
      + original-name: str.
        The name, in its raw un-transformed form
      + normalized-name: str.
        Standardized form actually used for mathing to mentions
      + name-type: int. must be < 100.
        The type of name for this concept, where a concept may have different types of names.
        For example, a concept may have a Primary Name, Synonyms, Acronymns, and some Common Names,
        where the Primary Name and Synonyms are unique to this concept, but the others might not be unique.

        Use method `set_name_type_weights()` to add weights to
        the match score according to name-type, to give some name types preference
        (e.g. Primary-Name gets the highest weight).

    Given a (normalized) mention, and a (normalized) name, name-type:
        match_score = w1 * char-n-gram-match(mention, name) +
                      w2 * word-n-gram-match(mention, name) +
                      name-type-wt[name-type]

        ... where:
            mention, name are strings, normalized (standardized) by the caller.
                This normalization could be a NOOP, as simple as str.casefold(),
                or something more elaborate perhapsl also involving tokenization.
            w1, w2 are weights set in `tvzr_params`, and either might be 0 (but not both).
            name-type-wt[name-type] is optional non-zero weight set using `set_name_type_weights()`.
            *-n-gram-match() is the match score obtained using corresponding `TfidfVectorizer`, whose
                parameters are set in `tvzr_params`.
    """

    # The following used by `sort_nz` to do a compound sort
    ROUND_NBR_DECIMALS = 5
    MAX_NAME_TYPE_STRICT = 100

    # noinspection PyTypeChecker
    def __init__(self,
                 name: str,
                 tvzr_params: Union[str, Dict[str, TfdfParams]]
                 ):
        """
        Wrapper wround TfidfVectorizer for matching concept names.

        :param name: Name for this class instace
        :param tvzr_params: Sets various options for using TfidfVectorizer for character-based and word-based matches.
            Can be a path to a JSON file, or a Dict. Either case will have the following optional keys:
                - "char_params": TfdfParams for char n-gram matching
                - "word_params": TfdfParams for word (token) n-gram matching
        """

        self.name = name

        self.char_tvzr: TfidfVectorizer = None
        self.word_tvzr: TfidfVectorizer = None
        #
        self.tvzrs: List[TfidfVectorizer] = []
        self.tvzrs_weights: List[float] = []
        #
        self._create_tvzrs(tvzr_params)

        self.name_type_weights = None
        self.set_name_type_weights()

        # The following are set during calls to `add_name()`

        # The following have one entry per name. Length = self.nbr_names.
        self.original_names: List[str] = []
        self.normalized_names: List[str] = []
        # `build()` converts the following into np.ndarray
        self.concept_type_and_id  = []          # ... List[Tuple[str, str]]
        self.name_types = []                    # ... List[int]

        # Dict: {name => [start, stop]} indices into the above names-seq
        # ... store index-pair as list for convenience
        self.concept_type_name_idx: Dict[str, List[int]] = dict()
        self.concept_name_idx: Dict[str, List[int]] = dict()

        self.nbr_names = 0
        self._prev_concept_id = None
        self._prev_concept_id_idx = None
        self._prev_concept_type = None
        self._prev_concept_type_idx = None

        # The following are initialized by `build()`

        self.char_tfidfs: sp.csr_matrix = None
        self.word_tfidfs: sp.csr_matrix = None
        # List of active tfidfs
        self.name_tfidfs: List[sp.csr_matrix] = []
        # Weights associated with `self.name_types`
        self.name_type_weight_vec: Optional[np.ndarray] = None

        return

    def _create_tvzrs(self,
                      tvzr_params: Union[str, Dict[str, TfdfParams]]):
        """
        Create the tvzr instances.

        :param tvzr_params: Path to Options file, or contents of that file
            Keys:
                char_params: TfDfParams
                word_params: TfDfParams
        """

        if isinstance(tvzr_params, str):
            with open(tvzr_params) as f:
                tvzr_params = json.load(f)

        char_tvzr_params = tvzr_params.get("char_params")
        if isinstance(char_tvzr_params, dict):
            char_tvzr_params = TfdfParams(**char_tvzr_params)

        word_tvzr_params = tvzr_params.get("word_params")
        if isinstance(word_tvzr_params, dict):
            word_tvzr_params = TfdfParams(**word_tvzr_params)

        assert char_tvzr_params is not None or word_tvzr_params is not None, \
            "At least one of 'char_params' or 'word_params' must be non-empty."

        if char_tvzr_params is not None:
            self.char_tvzr = TfidfVectorizer(input="content", analyzer="char",
                                             ngram_range=char_tvzr_params.ngram_widths_minmax,
                                             min_df=char_tvzr_params.min_df,
                                             max_df=char_tvzr_params.max_df,
                                             use_idf=char_tvzr_params.use_idf,
                                             norm="l2" if char_tvzr_params.normalize_tf_idf else None,
                                             stop_words=char_tvzr_params.stop_words,
                                             max_features=char_tvzr_params.max_features,
                                             # Caller's Tokenizer handles case, so lowercase=False
                                             lowercase=False,
                                             # these are defaults
                                             smooth_idf=True, sublinear_tf=False)
            self.tvzrs.append(self.char_tvzr)
            self.tvzrs_weights.append(char_tvzr_params.sim_weight)

        if word_tvzr_params is not None:
            self.word_tvzr = TfidfVectorizer(input="content", analyzer="word",
                                             ngram_range=word_tvzr_params.ngram_widths_minmax,
                                             min_df=word_tvzr_params.min_df,
                                             max_df=word_tvzr_params.max_df,
                                             use_idf=word_tvzr_params.use_idf,
                                             norm="l2" if word_tvzr_params.normalize_tf_idf else None,
                                             stop_words=word_tvzr_params.stop_words,
                                             max_features=word_tvzr_params.max_features,
                                             # Caller's Tokenizer tokenizes and combines tokens using SPACE as delim
                                             tokenizer=str.split,
                                             # Caller's Tokenizer handles case, so lowercase=False
                                             lowercase=False,
                                             # these are defaults
                                             smooth_idf=True, sublinear_tf=False)
            self.tvzrs.append(self.word_tvzr)
            self.tvzrs_weights.append(word_tvzr_params.sim_weight)

        # Normalize the weights to unit sum
        self.tvzrs_weights = [x / sum(self.tvzrs_weights) for x in self.tvzrs_weights]
        return

    def set_name_type_weights(self, name_type_weights: Optional[Dict[int, float]] = None):
        self.name_type_weights = defaultdict(float)

        if name_type_weights:
            self.name_type_weights.update(name_type_weights)

        self._build_name_type_weights()
        return

    def add_name(self, concept_type: str, concept_id: str, original_name: str, normalized_name: str,
                 name_type: int = 0):
        """
        Adds one name to the reference dictionary of concept names.

        Note: Names should be added in order of (concept_type, concept_id).
        A normalized name is not repeated for the same `concept_id` (i.e. a repeated name is not added).
        If not using name-types in your application Then leave `name_type` at its default value.
        """

        # Needed for `sorted_nz()`
        assert name_type < self.MAX_NAME_TYPE_STRICT, f"`name_type` must be < {self.MAX_NAME_TYPE_STRICT}"

        if concept_id == self._prev_concept_id:
            s, e = self._prev_concept_id_idx

            # Don't repeat names for same cuid
            if normalized_name in self.normalized_names[s : e]:
                return

            self._prev_concept_id_idx[1] += 1

        else:
            self._prev_concept_id = concept_id
            self._prev_concept_id_idx = [self.nbr_names, self.nbr_names + 1]
            self.concept_name_idx[concept_id] = self._prev_concept_id_idx

        if concept_type == self._prev_concept_type:
            self._prev_concept_type_idx[1] += 1

        else:
            self._prev_concept_type = concept_type
            self._prev_concept_type_idx = [self.nbr_names, self.nbr_names + 1]
            self.concept_type_name_idx[concept_type] = self._prev_concept_type_idx

        self.normalized_names.append(normalized_name)
        self.original_names.append(original_name)
        self.name_types.append(name_type)
        self.concept_type_and_id.append((concept_type, concept_id))

        self.nbr_names += 1

        return

    def build(self, verbose=False):
        """
        'Fit's the char_tfidf and word_tfidf structures, populating `self.name_tfidfs`.
        Converts other Lists to np.ndarray.
        """

        if verbose:
            print(f"Building {self.__class__.__name__} for {self.name} ...")
            print("  Initializing TfidfVectorizer(s) from all names ...", flush=True)

        if self.char_tvzr is not None:
            self.char_tfidfs = self.char_tvzr.fit_transform(self.normalized_names)
            self.name_tfidfs.append(self.char_tfidfs)

        if self.word_tvzr is not None:
            self.word_tfidfs = self.word_tvzr.fit_transform(self.normalized_names)
            self.name_tfidfs.append(self.word_tfidfs)

        # Reduce memory footprint, since we do not need this any more
        self.normalized_names = []

        self.concept_type_and_id = np.asarray(self.concept_type_and_id)
        self.name_types = np.asarray(self.name_types, dtype=np.int32)
        # Call this after setting `self.name_tfidfs`, which is used as a signal that `build` was called.
        self._build_name_type_weights()

        if verbose:
            print("Nbr names = {:9,d}".format(self.nbr_names))
            print(f"Nbr Concepts = {len(self.concept_name_idx):9,d}",
                  f"nbr Concept-Types = {len(self.concept_type_name_idx):,d}")
            for tid in sorted(self.concept_type_name_idx.keys()):
                start, stop = self.concept_type_name_idx[tid]
                print("   Nbr names in {:s} = {:9,d}".format(tid, stop - start))
            print(flush=True)

        return

    def _build_name_type_weights(self):
        """
        Sets `self.name_type_weight_vec` based on current `self.name_type_weights`.
        Uses non-empty `self.name_tfidfs` as a signal that `build` was called.
        """
        # `self.name_tfidfs` means `self.build()` has been called
        if self.name_type_weights and self.name_tfidfs:
            self.name_type_weight_vec = np.asarray([self.name_type_weights[nt] for nt in self.name_types],
                                                   dtype=np.float32)
        else:
            self.name_type_weight_vec = None
        return

    def get_nbr_names(self):
        return self.nbr_names

    def get_original_name(self, name_index: int) -> str:
        return self.original_names[name_index]

    def get_matching_concepts_batched(self, normalized_mentions: Iterable[str],
                                      concept_id: str = None, concept_type: str = None,
                                      nmax: int = None, min_score: float = None) \
            -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        `normalized_mentions` are normalized by the caller.

         If `concept_id` or `concept_type` are provided Then it applies to all the mentions.
         If `nmax` is provided Then return at most `nmax` matches.
         If `min_score` is provided Then return only matches with scores >= min_score
            (filter applied before adding match_name_type).

         Results are sorted on descending match score.
         If non-zero name_type_weights added using self.set_name_type_weights,
         Then the corresponding weights are added to the match score before sorting.

         :returns: matching_concepts_seq
                = [..., matching_concepts_seq[i], ...], one for each mention in `normalized_mentions`.
            where ...
                matching_concepts_seq[i] is:  ( array[[concept_type, concept_id], ...],
                                                array[score, ...],
                                                array[match_name_type, ...],
                                                array[name_index, ...] )
                sorted on descending score.
                `match_name_type` indicates the type of name that this match is against, controlled by params.

            Example output element:
             (array([['T022', 'C0079652'],
                     ['T022', 'C0079652'],
                     ['T022', 'C0079652'],
                     ['T022', 'C0816872'],
                     ['T022', 'C1321512']], dtype='<U8'),
              array([0.03777104, 0.03643799, 0.03555635, 0.01965859, 0.0189897 ]),
              array([1, 2, 2, 3, 4]),
              array([980106, 980107, 980108, 980381, 980900]))

            IF mentions[i] has no matches THEN
                returns tuple of empty arrays:
                        (array([], shape=(0, 2), dtype='<U1'),
                         array([], dtype=float32),
                         array([], dtype=int32),
                         array([], dtype=int32))

            name_index can be used to retrieve:
                - Original Name:    get_original_name(name_index)
        """
        start_idx, end_idx = 0, self.nbr_names

        # Restrict match to constrained concept_id or type_id
        if concept_id is not None:
            start_idx, end_idx = self.concept_name_idx.get(concept_id, [-1, -1])
        elif concept_type is not None:
            start_idx, end_idx = self.concept_type_name_idx.get(concept_type, [-1, -1])

        if start_idx >= end_idx:
            return [(np.empty((0, 2), dtype=np.str_),
                     np.empty(0, dtype=np.float32),
                     np.empty(0, dtype=np.int32),
                     np.empty(0, dtype=np.int32))
                    for _ in normalized_mentions]

        tot_pairwise_scores = None

        for tvzr, sim_wt, lex_tfidfs in zip(self.tvzrs, self.tvzrs_weights, self.name_tfidfs):

            # Compute match scores (higher is better) between each (mention, reference) pair
            # pairwise_scores.shape = (n_mentions, n_refs) where n_refs = end_idx - start_idx

            concept_tfidfs = lex_tfidfs[start_idx: end_idx]
            mention_tfidfs = tvzr.transform(normalized_mentions)
            # Both arrays are of type `scipy.sparse.csr_matrix`

            # The following actually slows it down!
            # mention_tfidfs = mention_tfidfs.astype(np.float32)
            # mention_tfidfs.sort_indices()

            # pairwise_scores is np.array of shape (n_mentions, n_refs)
            if tvzr.norm == "l2":
                pairwise_scores = linear_kernel(mention_tfidfs, concept_tfidfs)
            else:
                # `n_jobs=-1` means use all available CPUs
                pairwise_scores = pairwise_distances(mention_tfidfs, concept_tfidfs, metric='cosine', n_jobs=-1)

            pairwise_scores = sim_wt * pairwise_scores

            if tot_pairwise_scores is None:
                tot_pairwise_scores = pairwise_scores
            else:
                tot_pairwise_scores += pairwise_scores

        # Add type-name weights. Take advantage of np broadcasting
        if self.name_type_weight_vec is not None:
            tot_pairwise_scores += self.name_type_weight_vec[start_idx : end_idx].reshape((-1, 1))

        # Sort on reverse-score, followed by tid, and cuids are in order read from lexicon

        matching_concepts_seq = [self.sorted_nz(tot_pairwise_scores[row], start_idx, end_idx,
                                                nmax=nmax, min_score=min_score)
                                 for row in range(tot_pairwise_scores.shape[0])]

        # noinspection PyTypeChecker
        return matching_concepts_seq

    def sorted_nz(self, row_scores: Sequence[float],
                  start_idx: int, end_idx: int,
                  nmax: int = None, min_score: float = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Discard zero-score matches and scores below `min_score`,
        sort on score descending, and select top `nmax`.
        IF no name-type-weights THEN compound sort on (score, match_name_type)
        """

        row_scores = np.asarray(row_scores)

        if min_score is not None:

            if self.name_type_weight_vec is not None:
                # Apply filter as if before adding match_name_type to score.
                min_scores_vec = self.name_type_weight_vec[start_idx : end_idx] + min_score
                nz_i = np.nonzero(row_scores >= min_scores_vec)[0]
            else:
                nz_i = np.nonzero(row_scores >= min_score)[0]

        else:
            nz_i = np.nonzero(row_scores)[0]

        nz = row_scores[nz_i]

        if self.name_type_weight_vec is None:
            # This will do a compound sort on (score, match_name_type)
            # ASSUMES: name_type < 100 (self.MAX_NAME_TYPE_STRICT)
            nz = np.around(nz, self.ROUND_NBR_DECIMALS) * np.power(10, self.ROUND_NBR_DECIMALS + 2)
            nz += self.name_types[start_idx : end_idx][nz_i]

        # Use "mergesort" for stable sort
        s_i = np.argsort(-nz, kind="mergesort")
        if nmax:
            s_i = s_i[:nmax]

        sorted_idxs = nz_i[s_i]

        return self.concept_type_and_id[start_idx + sorted_idxs], \
            row_scores[sorted_idxs], \
            self.name_types[start_idx + sorted_idxs], \
            start_idx + sorted_idxs

# /
