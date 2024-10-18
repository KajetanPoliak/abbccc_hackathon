import numpy as np
import pandas as pd

from source.data_processing.data_processing import DataProcessor
from source.index.keyword_index import KeywordSearchIndex
from source.index.vector_index import FaissIndex
from source.vectorization.deep import get_tokenizer_and_model


class Pipeline:
    def __init__(self) -> None:
        self.dp = DataProcessor()
        self.index = FaissIndex(dim=768)

    def run(self) -> pd.DataFrame:
        ### Initialize the pipeline ###
        # Load the keyword index
        keyword_index = KeywordSearchIndex.from_file(
            "keyword_search_index.json"
        )
        # Load the context index
        context_index = FaissIndex.from_file("faiss_ip.index")

        # Get the event data
        title_body_list = self.dp.get_email_data()
        preds = []
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

            print(search_results_df)

            ##### FAISS INDEX #####
            context_query = context_index.process_query_document(
                title=event[1], body=event[2]
            )
            tok, mod = get_tokenizer_and_model(device="cpu")
            print("TEST")
            dist, idx, itms = context_index.search(
                document=context_query, tokenizer=tok, model=mod, k=None
            )
            print("Search after the load:", dist, idx, itms)
            context_result_df = context_index.to_dataframe(dist, idx, itms)

            result_df = pd.merge(
                search_results_df, context_result_df, on=["project", "activity"]
            )
            result_df["event_id"] = event_id
            preds.append(result_df)
        preds_df = pd.concat(preds, axis=0)
        return preds_df


if __name__ == "__main__":
    pass
