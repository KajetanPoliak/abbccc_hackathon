import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import numpy.typing as npt
from transformers import AutoModel, AutoTokenizer

from source.utils.logging import get_stream_logger
from source.vectorization.deep import encode_documents, get_tokenizer_and_model

__here__ = Path(__file__).resolve().parent
__root__ = __here__.parents[1]
__data_dir__ = __root__ / "data"
assert __data_dir__.exists(), f"Data directory not found: {__data_dir__!r}"


class FaissIndex:
    def __init__(self, dim: int):
        self.logger = get_stream_logger(self.__class__.__name__)
        self.index = faiss.IndexFlatIP(dim)
        self.index_items: List[str] = []

    def set_embeddings(self, embeddings: npt.NDArray, items: List[str]) -> None:
        self.index.reset()
        self.logger.info(f"Setting embeddings with shape: {embeddings.shape}")
        self.index.add(embeddings)
        self.index_items = items

    def search_by_vector_query(
        self,
        query: npt.NDArray,
        k: Optional[int] = 5,
    ) -> Tuple[npt.NDArray, npt.NDArray, List[str]]:
        if k is None:
            k = len(self.index_items)
        distances, indices = self.index.search(query, k)
        return distances, indices, [self.index_items[i] for i in indices[0]]

    @classmethod
    def process_query_document(cls, title: str, body: str) -> str:
        """Process a query document and extract its embeddings from the title
        and body"""
        query = f"{title} / {body}".strip("/ ")
        return query

    def search(
        self,
        document: str,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        k: Optional[int] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray, List[str]]:
        query = encode_documents(
            [document], tokenizer, model, normalize=True, device="mps"
        )
        # Normalize the query vector
        query /= np.linalg.norm(query, axis=1).reshape(-1, 1)
        return self.search_by_vector_query(query, k)

    def save(self, filename: str) -> None:
        path = __data_dir__ / filename
        self.logger.info(f"Saving index to {path!r}")
        faiss.write_index(self.index, str(path))
        # Save the index items
        path_json = __data_dir__ / f"{filename}.json"
        self.logger.info(f"Saving index items to {path!r}")
        with open(path_json, "w") as f:
            data = json.dumps(self.index_items, indent=4)
            f.write(data)

    @classmethod
    def from_file(cls, filename: str) -> "FaissIndex":
        instance = cls(dim=0)
        instance.index = faiss.read_index(str(__data_dir__ / filename))
        with open(__data_dir__ / f"{filename}.json", "r") as f:
            instance.index_items = json.load(f)
        return instance


if __name__ == "__main__":
    index = FaissIndex(dim=768)
    _sample_vectors = np.random.randn(100, 768).astype("float32")
    _sample_vectors /= np.linalg.norm(_sample_vectors, axis=1).reshape(-1, 1)
    index.set_embeddings(
        embeddings=_sample_vectors, items=[f"item_{i}" for i in range(100)]
    )
    # Search for the most similar document for vector at index 0
    dist, idx, itms = index.search_by_vector_query(_sample_vectors[:1], k=None)
    print("First search:", dist, idx, itms)
    assert idx[0][0] == 0 and np.isclose(dist[0][0], 1.0, atol=1e-3)

    # Save the index
    index.save("faiss_ip.index")

    # Load the index
    index = FaissIndex.from_file("faiss_ip.index")
    dist, idx, itms = index.search_by_vector_query(_sample_vectors[:1], k=5)
    print("Search after the load:", dist, idx, itms)
    assert idx[0][0] == 0 and np.isclose(dist[0][0], 1.0, atol=1e-3)

    # Read project data to create an index
    import pandas as pd

    df = pd.read_csv(__data_dir__ / "trimmed_project_data.csv").fillna("")
    df = df.groupby(
        by=["Project Description", "Activity Description"],
        as_index=False,
    ).agg({"Comment": lambda item: " | ".join(item.tolist()).strip("| ")})
    tok, mod = get_tokenizer_and_model(device="mps")
    # Create a list of documents by concatenating project name,
    # activity description, and comment
    documents = df.apply(
        lambda row: f"{row['Project Description']}: "
        f"{row['Activity Description']} / "
        f"{row['Comment']}".strip("/ "),
        axis=1,
    ).tolist()
    embs = encode_documents(
        documents, tokenizer=tok, model=mod, normalize=True, device="mps"
    )
    index = FaissIndex(dim=embs.shape[1])
    # Create the index items by preserving only the project name and activity
    # descriptions
    items_parsed = [document.split("/")[0].strip() for document in documents]
    index.set_embeddings(embs, items=items_parsed)
    index.save("faiss_ip.index")

    # Example search
    title = "Discussing Optimax"
    body = "This is a discussion about Optimax"
    query = FaissIndex.process_query_document(title, body)
    dist, idx, itms = index.search(query, tok, mod, k=None)
    print("Example search:", dist, idx, itms)
