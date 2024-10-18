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
        ### Initialize the pipeline ###
        # Load the keyword index
        keyword_index = KeywordSearchIndex.from_file(
            "keyword_search_index.json"
        )
        # Load the context index
        context_index = FaissIndex.from_file("faiss_ip.index")

        # Get the event data
        title_body_list = self.dp.get_email_data()
        for event in title_body_list:
            event_id = event[0]
            ####### KEYWORD SEARCH #######
            # Define a query document
            keyword_query = {
                "title": event[1],
                "body": event[2],
            }
            # Process a query document
            keyword_query_processed = keyword_index.process_query_document(
                **keyword_query
            )
            # Perform search
            search_results = keyword_index.search(
                keyword_query_processed, title=keyword_query["title"]
            )
            search_results_df = keyword_index.to_dataframe(search_results)
            search_results_df["event_id"] = event_id

            print(search_results_df)

            ##### FAISS INDEX #####
            context_query = context_index.process_query_document(
                title=event[1], body=event[2]
            )
            tok, mod = get_tokenizer_and_model(device="cpu")
            dist, idx, itms = context_index.search(
                document=context_query, tokenizer=tok, model=mod, k=5
            )
            print("Search after the load:", dist, idx, itms)


if __name__ == "__main__":
    pass
