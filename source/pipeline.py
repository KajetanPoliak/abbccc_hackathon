from source.data_processing.data_processing import DataProcessor
from source.index.keyword_index import KeywordSearchIndex
from source.index.vector_index import FaissIndex
from source.vectorization.deep import apply_vectorization


class Pipeline:
    def __init__(self) -> None:
        self.dp = DataProcessor()
        self.index = FaissIndex(dim=768)

    def run(self) -> None:
        keywords_df = self.dp.get_project_data()

        # Vectorize event data
        # title_body_list = self.dp.get_email_data()
        # embeddings = apply_vectorization(title_body_list=title_body_list)
        # print(embeddings)


if __name__ == "__main__":
    pass
