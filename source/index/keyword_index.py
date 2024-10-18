import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Set

import nltk
import numpy as np

from source.utils.logging import get_stream_logger
from source.utils.string_matching import fuzzy_nn_match

__here__ = Path(__file__).resolve().parent
__root__ = __here__.parents[1]
__data_dir__ = __root__ / "data"
assert __data_dir__.exists(), f"Data directory not found: {__data_dir__!r}"


class KeywordSearchIndex:
    stopwords: Set[str] = set()

    def __init__(self) -> None:
        self.logger = get_stream_logger(self.__class__.__name__)
        # Store core documents with their keywords and hierarchical structure
        self.index: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.stopwords = self.get_stop_words()

    def __new__(cls, *args: Any, **kwargs: Any) -> "KeywordSearchIndex":
        # Set stopwords as a class attribute
        if not cls.stopwords:
            cls.stopwords = cls.get_stop_words()
        return super().__new__(cls)

    @classmethod
    def get_stop_words(cls) -> Set[str]:
        # Check if stopwords are already downloaded within NLTK
        try:
            nltk.data.find("corpora/stopwords.zip")
        except LookupError:
            nltk.download("stopwords")
        return set(
            nltk.corpus.stopwords.words("english")
            + nltk.corpus.stopwords.words("german")
        )

    @classmethod
    def _extract_keywords(cls, text: str) -> Set[str]:
        # Simple keyword extraction by splitting on non-alphabetic characters
        # and removing stopwords
        words = re.findall(r"\b\w+\b", text.lower())
        keywords = [word for word in words if word not in cls.stopwords]
        return set(keywords)

    def add_core_document(
        self, project_name: str, activity_desc: str, content: str
    ) -> None:
        """Add core documents with their hierarchical structure and extracted
        keywords"""
        keywords = self._extract_keywords(
            f"{project_name} {activity_desc} {content}"
        )
        self.index[project_name][activity_desc].update(keywords)
        self.logger.info(
            f"Added core document: {project_name} - {activity_desc}: "
            f"{len(keywords)} keywords"
        )

    def remove_frequent_keywords_from_index(self, threshold: int = 5) -> None:
        """Remove such keywords that repeat across multiple core documents"""
        keyword_counts: Dict[str, int] = defaultdict(int)
        for project, activities in self.index.items():
            for activity, keywords in activities.items():
                for keyword in keywords:
                    keyword_counts[keyword] += 1
        for project, activities in self.index.items():
            for activity, keywords in activities.items():
                self.logger.debug(
                    f"Removing frequent keywords from {project} - {activity}: "
                    f"{len(keywords)} keywords before"
                )
                self.index[project][activity] = {
                    keyword
                    for keyword in keywords
                    if keyword_counts[keyword] <= threshold
                }
                self.logger.debug(
                    f"Removing frequent keywords from {project} - {activity}: "
                    f"{len(self.index[project][activity])} keywords after"
                )

    def process_query_document(self, title: str, body: str) -> Set[str]:
        """Process a query document and extract its keywords from the title
        and body"""
        query_keywords = self._extract_keywords(title + " " + body)
        return query_keywords

    def search(
        self, query_keywords: Set[str], title: Optional[str]
    ) -> Dict[str, Any]:
        """Search for matching documents in the index based on the extracted
        keywords"""
        results: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        title_tagged_results = set()
        # Apply quick fuzzy matching on project titles
        if title:
            project_match = fuzzy_nn_match(
                np.array([title]),
                np.array(list(self.index.keys())),
                n_neighbors=5,
                limit=1,
            )
            # Filter our all matches with low confidence
            project_match = project_match.loc[
                project_match["match_confidence"] >= 0.95, :
            ]
            if not project_match.empty:
                project_match = project_match.sort_values(
                    "match_confidence", ascending=False
                )
                for project in project_match["clean"].tolist():
                    if project in title_tagged_results:
                        continue
                    for activity, _ in self.index[project].items():
                        results[project][activity] += project_match[
                            "match_confidence"
                        ].iloc[0]
                title_tagged_results.add(title)

        # Search through the hierarchical index for keyword matches
        for project, activities in self.index.items():
            for activity, keywords in activities.items():
                match_score = results[project][activity] + len(
                    keywords.intersection(query_keywords)
                ) / len(keywords)
                if match_score > 0.0:
                    results[project][activity] = round(match_score, 4)

        return results

    @classmethod
    def display_search_results(cls, results: Dict[str, Any]) -> None:
        """Display the search results in a readable format"""
        for project, activities in results.items():
            print(f"Project: {project}")
            for activity, match_count in activities.items():
                print(f"\tActivity: {activity} (Matches: {match_count})")

    def save(self, filename: str) -> None:
        """Save the search index to a JSON file"""
        # Convert the defaultdict to a regular dictionary before serializing
        index_serializable = {
            project: {
                activity: list(keywords)
                for activity, keywords in activities.items()
            }
            for project, activities in self.index.items()
        }
        with open(__data_dir__ / filename, "w") as file:
            data = json.dumps(index_serializable, indent=4)
            file.write(data)

    @classmethod
    def from_file(cls, filename: str) -> "KeywordSearchIndex":
        """Load the search index from a JSON file"""
        empty_index: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        with open(__data_dir__ / filename, "r") as file:
            index_raw = json.load(file)
            for project, activities in index_raw.items():
                for activity, keywords in activities.items():
                    empty_index[project][activity] = set(keywords)
        instance = cls()
        instance.index = empty_index
        return instance


if __name__ == "__main__":
    # Example usage:
    # Create the search index
    index = KeywordSearchIndex()

    # Add core documents to the index
    index.add_core_document(
        "Gasum Loiste", "Engineering", "IAT/FAT/SAT documentation"
    )
    index.add_core_document(
        "Gasum Loiste",
        "Engineering",
        "SAF Heat Storage value converting, FAT preparation",
    )
    index.add_core_document(
        "OPTIMAX APPS&MODELS CZ 2024 WP-12081.04", "Optimax - demo", ""
    )
    # Remove frequent keywords from the index
    # (here, using a threshold of 1 just for debugging purposes)
    index.remove_frequent_keywords_from_index(threshold=1)

    # Define a query document
    query = {
        "title": "Discussing Optimax",
        "body": "Let's have a discussion about Optimax and its features",
    }
    # Process a query document
    query_processed = index.process_query_document(**query)

    # Perform search
    search_results = index.search(query_processed, title=query["title"])

    # Display search results
    print(f"Query: {query!r}")
    index.display_search_results(search_results)

    # Save the search index
    index.save("keyword_search_index.json")

    # Load the search index
    index = KeywordSearchIndex.from_file("keyword_search_index.json")
    print("Search index loaded from file:", index.index)

    # Perform search
    search_results = index.search(query_processed, title=query["title"])

    # Display search results
    print(f"Query: {query!r}")
    index.display_search_results(search_results)