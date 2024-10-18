"""
Utility functions for fuzzy TF-iDF matching primarily for the CompanyManager.
The core function to use is *fuzzy_nn_match(...)*.
"""

import itertools
import re
from typing import Any, List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# String pre-processing
def preprocess_string(s: str) -> str:
    # Remove spaces between strings with one or two letters
    s = re.sub(r"(?<=\b\w)\s*[ &]\s*(?=\w\b)", "", s)
    # Remove special characters
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
    # Remove multiple spaces
    s = re.sub(r"\s+", " ", s)
    # Lowercase
    s = s.lower()
    return s


# String matching - TF-IDF
def initialize_matching_components(
    clean: Union[pd.Series, npt.NDArray[str]],
    analyzer: str = "word",
    ngram_range: Tuple[int, int] = (1, 2),
    n_neighbors: int = 1,
    **kwargs: Any,
) -> Tuple[TfidfVectorizer, NearestNeighbors]:
    """
    Initialization of matching components that are required for the process.

    :param clean: pd.Series; series to build NN indices and TF-IDF vocabulary for
    :param analyzer: see
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    :param ngram_range: see
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    :param n_neighbors: int; number of nearest neighbors for the NN index; see

    :param kwargs: kwargs for TF-IDF vectorizer, see
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    :return: tuple; tf-idf vectorizer and NN index
    """
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        input="content",
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words="english",
        **kwargs,
    )
    clean_vectors = vectorizer.fit_transform(clean.astype("U"))

    # Fit nearest neighbors corpus
    nn = NearestNeighbors(
        n_neighbors=n_neighbors, metric=cosine_similarity
    ).fit(clean_vectors)
    return vectorizer, nn


# String matching - KNN
def tfidf_nn(
    messy: pd.Series, clean: pd.Series, n_neighbors: int = 1, **kwargs: Any
) -> Tuple[npt.NDArray[str], npt.NDArray[float]]:
    """
    TF-IDF NN search.

    :param messy: pd.Series; messy string records to match
    :param clean: pd.Series; clean string records to match against
    :param n_neighbors: int; number of nearest neighbors to identify with the TF-IDF matching
    :param kwargs: kwargs for TF-IDF vectorizer
    :return: tuple[nearest_neighbors_array (textual), distances_array]
    """
    # Fit clean data and transform messy data
    vectorizer, nbrs = initialize_matching_components(
        clean,
        n_neighbors=min(len(clean), n_neighbors),
        analyzer="word",
        **kwargs,
    )
    input_vec = vectorizer.transform(messy)

    # Determine best possible matches
    distances, indices = nbrs.kneighbors(
        input_vec, n_neighbors=min(len(clean), n_neighbors)
    )
    nearest_values = np.array(clean)[indices]
    return nearest_values, distances


# String matching - match fuzzy
def find_matches_fuzzy(
    row: str, match_candidates: npt.NDArray[str], limit: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Search of fuzzy matches.

    :param row: str; record to search matches for
    :param match_candidates: array, shape=(n_candidates, ); candidates among which search is to be conducted
    :param limit: int; number of top matches to return
    :return: list of tuples; each tuple contains the following information:
        tuple[row, top_match_i, match_confidence_i] where 0 <= i < 5
    """
    row_matches = process.extract(
        query=row,
        choices=dict(enumerate(match_candidates)),
        scorer=fuzz.WRatio,
        limit=limit,
    )
    result = [(row, match[0], match[1]) for match in row_matches]
    return result


# String matching - TF-IDF
def fuzzy_nn_match(
    messy: Union[pd.Series, npt.NDArray[str]],
    clean: Union[pd.Series, npt.NDArray[str]],
    n_neighbors: int = 100,
    limit: int = 5,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Function that creates a matching table between messy and clean records.
    Firstly, it searches for top *n_neighbors* TF-IDF matches; secondly, it adds to the match table
    the top *limit* fuzzy matches out of those.

    :param messy: pd.Series; messy string records to match
    :param clean: pd.Series; clean string records to match against
    :param n_neighbors: int; number of nearest neighbors to identify with the TF-IDF matching
    :param limit: int; number of top matches to return per messy record
    :param kwargs: kwargs for the TF-IDF vectorizer
    :return: pd.DataFrame; matching table with messy vs. clean matches with match_confidence score
    """
    # Create a preprocessed version of the data
    messy_ = pd.Series(messy).apply(preprocess_string)
    clean_ = pd.Series(clean).apply(preprocess_string)
    # Create a map between original and preprocessed data
    original_to_preprocessed_messy = {
        row_: row for row_, row in zip(messy_, messy)
    }
    original_to_preprocessed_clean = {
        row_: row for row_, row in zip(clean_, clean)
    }

    nearest_values, _ = tfidf_nn(
        messy=messy_, clean=clean_, n_neighbors=n_neighbors, **kwargs
    )

    results = [
        find_matches_fuzzy(
            row=cast(str, row), match_candidates=nearest_values[i], limit=limit
        )
        for i, row in enumerate(messy_)
    ]
    df = pd.DataFrame(
        itertools.chain.from_iterable(results),
        columns=["messy", "clean", "match_confidence"],
    )
    df.loc[:, "match_confidence"] /= 100.0
    # Map back to original data
    df.loc[:, "messy"] = df["messy"].map(original_to_preprocessed_messy)
    df.loc[:, "clean"] = df["clean"].map(original_to_preprocessed_clean)
    return df.copy()
