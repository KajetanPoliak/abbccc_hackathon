from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
import numpy.typing as npt
from transformers import AutoModel, AutoTokenizer

from source.vectorization.deep import encode_documents

__here__ = Path(__file__).resolve().parent
__root__ = __here__.parents[1]
__data_dir__ = __root__ / "data"
assert __data_dir__.exists(), f"Data directory not found: {__data_dir__!r}"


class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def set_embeddings(self, embeddings: npt.NDArray) -> None:
        self.index.reset()
        self.index.add(embeddings)

    def search(
        self, query: npt.NDArray, k: int = 5
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        distances, indices = self.index.search(query, k)
        return distances, indices

    def search_by_document(
        self,
        document: str,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        k: int = 1,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        query = encode_documents([document], tokenizer, model, normalize=True)
        # Normalize the query vector
        query /= np.linalg.norm(query, axis=1).reshape(-1, 1)
        return self.search(query, k)

    def save(self, filename: str) -> None:
        faiss.write_index(self.index, str(__data_dir__ / filename))

    @classmethod
    def from_file(cls, filename: str) -> "FaissIndex":
        instance = cls(dim=0)
        instance.index = faiss.read_index(str(__data_dir__ / filename))
        return instance


if __name__ == "__main__":
    index = FaissIndex(dim=768)
    _sample_vectors = np.random.randn(100, 768).astype("float32")
    _sample_vectors /= np.linalg.norm(_sample_vectors, axis=1).reshape(-1, 1)
    index.set_embeddings(embeddings=_sample_vectors)
    # Search for the most similar document for vector at index 0
    dist, idx = index.search(_sample_vectors[:1], k=5)
    print("First search:", dist, idx)
    assert idx[0][0] == 0 and np.isclose(dist[0][0], 1.0, atol=1e-3)

    # Save the index
    index.save("faiss_ip.index")

    # Load the index
    index = FaissIndex.from_file("faiss_ip.index")
    dist, idx = index.search(_sample_vectors[:1], k=5)
    print("Search after the load:", dist, idx)
    assert idx[0][0] == 0 and np.isclose(dist[0][0], 1.0, atol=1e-3)
