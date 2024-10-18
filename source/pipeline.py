from typing import Any, Dict, List

import pandas as pd
from tqdm.auto import tqdm

from source.data_processing.data_processing import DataProcessor
from source.index.keyword_index import KeywordSearchIndex
from source.index.vector_index import FaissIndex
from source.vectorization.deep import get_tokenizer_and_model


class Pipeline:
    def __init__(self, device: str = "cpu") -> None:
        self.dp = DataProcessor()
        self.device = device

    def run(self) -> List[Dict[str, Any]]:
        ### Initialize the pipeline ###
        # Load the keyword index
        keyword_index = KeywordSearchIndex.from_file(
            "keyword_search_index.json"
        )
        # Load the context index
        tok, mod = get_tokenizer_and_model(device=self.device)
        context_index = FaissIndex.from_file("faiss_ip.index")

        # Get the event data
        events = self.dp.load_event_data()
        for event in tqdm(events, desc="Processing events"):
            ####### KEYWORD SEARCH #######
            # Define a query document
            keyword_query = {
                "title": event["subject"],
                "body": event["body_clean"],
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

            ##### FAISS INDEX #####
            # Process the query document for the context index
            context_query = context_index.process_query_document(
                title=event["subject"], body=event["body_clean"]
            )
            # Perform search
            dist, idx, itms = context_index.search(
                document=context_query,
                tokenizer=tok,
                model=mod,
                k=None,
                device=self.device,
            )
            # Convert the search results to a DataFrame
            context_result_df = context_index.to_dataframe(dist, idx, itms)

            result_df = pd.merge(
                search_results_df, context_result_df, on=["project", "activity"]
            )

            # Aggregate the "match_count" and "context_score" columns by summing
            # them to get the final score
            result_df["match_confidence"] = (
                result_df["match_count"] + result_df["context_score"]
            )
            idx_max = result_df["match_confidence"].idxmax()
            result_df = result_df.loc[[idx_max], :]
            output = result_df.rename(
                columns={
                    "project": "project_description",
                    "activity": "project_activity",
                    "match_count": "pred_confidence_score_keyword",
                    "context_score": "pred_confidence_score_context",
                    "match_confidence": "pred_confidence_score",
                }
            ).to_dict(orient="records")
            event |= output[0]
        return events


if __name__ == "__main__":
    pass
