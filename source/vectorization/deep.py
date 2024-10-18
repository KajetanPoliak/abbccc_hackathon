from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def get_tokenizer_and_model(
    model_name: str = "sentence-transformers/paraphrase-multilingual"
    "-mpnet-base-v2",
    device: str = "cpu",
) -> Tuple[AutoTokenizer, AutoModel]:
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    return tokenizer, model


def mean_pooling(
    model_output: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean Pooling, taking attention mask into account for correct averaging

    :param model_output:
    :param attention_mask:
    :return:
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode_documents(
    documents: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 8,
    normalize: bool = False,
) -> npt.NDArray:
    # Tokenize sentences
    # noinspection PyCallingNonCallable
    encoded_input = tokenizer(
        documents, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = encoded_input.to("mps")
    # Compute token embeddings
    with torch.no_grad():
        # Process in batches to avoid memory issues
        embeddings = []
        for i in tqdm(
            range(0, len(documents), batch_size), desc="Computing embeddings"
        ):
            batch = (
                encoded_input["input_ids"][i : i + batch_size],
                encoded_input["attention_mask"][i : i + batch_size],
            )
            # noinspection PyCallingNonCallable
            output = model(*batch)
            embeddings.append(
                mean_pooling(model_output=output, attention_mask=batch[1])
            )
        document_embeddings = torch.cat(embeddings, dim=0)
    if normalize:
        # Normalize embeddings by L2 norm using torch functions
        document_embeddings /= torch.norm(document_embeddings, dim=1).unsqueeze(
            dim=1
        )
    return document_embeddings.cpu().numpy()


def apply_vectorization(title_body_list: List[str]) -> npt.NDArray:
    tok, mod = get_tokenizer_and_model(device="mpu")
    embeddings = encode_documents(
        documents=title_body_list, tokenizer=tok, model=mod, normalize=True
    )
    return embeddings


if __name__ == "__main__":
    tok, mod = get_tokenizer_and_model(device="mpu")
    example_documents = [
        "This is a test sentence",
        "This is another sentence",
    ]

    example_embeddings = encode_documents(
        documents=example_documents, tokenizer=tok, model=mod, normalize=True
    )
    # Output: [[...], [...]]
    print(example_embeddings)
    # Test that norma of the embeddings is 1
    print(np.linalg.norm(example_embeddings, axis=1))
