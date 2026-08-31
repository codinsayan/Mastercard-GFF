"""One-time builder for the committed A2 MiniLM embedding cache."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .assets import EMBEDDINGS_PATH, INTENT_PHRASES, MODEL_NAME, load_catalogue


def main() -> None:
    catalogue = load_catalogue()
    item_texts: list[str] = []
    item_categories: list[str] = []
    item_median_prices: list[float] = []
    for category, products in catalogue.items():
        for product in products:
            item_texts.append(str(product["item"]))
            item_categories.append(category)
            item_median_prices.append(float(product["median_price"]))

    intent_categories = list(INTENT_PHRASES)
    intent_texts = [INTENT_PHRASES[category] for category in intent_categories]
    contextual_item_texts = [
        f"{category.replace('_', ' ')} supplies: {item}"
        for category, item in zip(item_categories, item_texts, strict=True)
    ]
    model = SentenceTransformer(MODEL_NAME)
    raw_item_embeddings = model.encode(
        contextual_item_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    intent_embeddings = model.encode(
        intent_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    intent_by_category = dict(zip(intent_categories, intent_embeddings, strict=True))
    category_context_weight = 0.65
    item_embeddings = np.asarray(
        [
            (1.0 - category_context_weight) * vector
            + category_context_weight * intent_by_category[category]
            for category, vector in zip(
                item_categories,
                raw_item_embeddings,
                strict=True,
            )
        ],
        dtype=np.float32,
    )
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_PATH,
        model_name=np.asarray(MODEL_NAME),
        item_texts=np.asarray(item_texts),
        item_categories=np.asarray(item_categories),
        item_median_prices=np.asarray(item_median_prices, dtype=np.float64),
        item_embeddings=item_embeddings,
        intent_categories=np.asarray(intent_categories),
        intent_texts=np.asarray(intent_texts),
        intent_embeddings=intent_embeddings,
    )
    print(
        f"Wrote {len(item_texts)} item and {len(intent_texts)} intent embeddings "
        f"to {EMBEDDINGS_PATH}"
    )


if __name__ == "__main__":
    main()
