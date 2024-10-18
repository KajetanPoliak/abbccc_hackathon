import numpy as np

from source.data_processing.data_processing import DataProcessor
from source.index.keyword_index import KeywordSearchIndex
from source.index.vector_index import FaissIndex
from source.vectorization.deep import get_tokenizer_and_model


class Pipeline:
    def __init__(self) -> None:
        self.dp = DataProcessor()
        self.index = FaissIndex(dim=768)

    def run(self) -> None:
        # Vectorize event data
        title_body_list = self.dp.get_email_data()

        ####### KEYWORD SEARCH INDEX #######
        # Load the search index
        keyword_index = KeywordSearchIndex.from_file(
            "keyword_search_index.json"
        )
        print("Search index loaded from file:", keyword_index.index)
        # Define a query document
        keyword_query = {
            "title": title_body_list[0],
            "body": title_body_list[1],
        }
        # Process a query document
        keyword_query_processed = keyword_index.process_query_document(
            **keyword_query
        )
        # Perform search
        search_results = keyword_index.search(
            keyword_query_processed, title=keyword_query["title"]
        )
        # Display search results
        print(f"Query: {keyword_query!r}")
        keyword_index.display_search_results(search_results)
        keyword_index.save_results(search_results, "keyword_search_results.csv")

        ##### FAISS INDEX #####
        # Load the index
        context_index = FaissIndex.from_file("faiss_ip.index")
        context_query = context_index.process_query_document(
            title=title_body_list[0], body=title_body_list[1]
        )
        tok, mod = get_tokenizer_and_model(device="mpu")
        dist, idx, itms = context_index.search(
            document=context_query, tokenizer=tok, model=mod, k=5
        )
        print("Search after the load:", dist, idx, itms)


if __name__ == "__main__":
    pass
