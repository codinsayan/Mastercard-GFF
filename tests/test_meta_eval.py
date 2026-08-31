import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from eval.metrics import binary_metrics, leave_one_attack_out, per_feature_auc
from models.meta import MetaLearner


def test_metrics_meta_and_leave_one_attack_out():
    rng = np.random.default_rng(4)
    fraud_attacks = [name for name in ("A1", "A2", "A3", "A4") for _ in range(10)]
    attacks = np.array(["LEGIT"] * 40 + fraud_attacks)
    labels = (attacks != "LEGIT").astype(np.int8)
    features = pd.DataFrame(
        {
            "shared": labels + rng.normal(0, 0.2, len(labels)),
            "novelty": rng.normal(0, 1, len(labels)),
        }
    )
    head_scores = pd.DataFrame({"a4_score": np.clip(features["shared"], 0, 1)})
    meta = MetaLearner()
    meta.fit(head_scores, features, labels)
    scores = meta.score(head_scores, features)
    assert binary_metrics(labels, scores)["auc_pr"] > 0.9
    assert per_feature_auc(features, labels)["shared"] > 0.9

    folds = leave_one_attack_out(
        features,
        labels,
        attacks,
        lambda: LogisticRegression(max_iter=1_000),
    )
    assert set(folds) == {"A1", "A2", "A3", "A4"}
