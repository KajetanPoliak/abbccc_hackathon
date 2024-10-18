import numpy as np

from source.data_processing.data_processing import DataProcessor
from source.index.keyword_index import KeywordSearchIndex
from source.index.vector_index import FaissIndex
from source.vectorization.deep import apply_vectorization


class Pipeline:
    def __init__(self) -> None:
        self.dp = DataProcessor()
        self.index = FaissIndex(dim=768)

    def run(self) -> None:
        # Vectorize event data
        title_body_list = self.dp.get_email_data()
        embeddings = apply_vectorization(title_body_list=title_body_list)

        ####### KEYWORD SEARCH INDEX #######
        # Load the search index
        index = KeywordSearchIndex.from_file("keyword_search_index.json")
        print("Search index loaded from file:", index.index)

        # Define a query document
        query = {
            "title": title_body_list[0],
            "body": title_body_list[1],
        }
        # Process a query document
        query_processed = index.process_query_document(**query)

        # Perform search
        search_results = index.search(query_processed, title=query["title"])

        # Display search results
        print(f"Query: {query!r}")
        index.display_search_results(search_results)

        ##### FAISS INDEX #####
        # Load the index
        index = FaissIndex.from_file("faiss_ip.index")
        dist, idx = index.search(embeddings, k=5)
        print("Search after the load:", dist, idx)
        assert idx[0][0] == 0 and np.isclose(dist[0][0], 1.0, atol=1e-3)


if __name__ == "__main__":
    pass
