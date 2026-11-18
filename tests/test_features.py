"""Feature contract tests: splits, target encoding and the transformer."""

from __future__ import annotations

import numpy as np

from churnops.data.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    prepare_xy,
    split_data,
)


def test_split_sizes_and_no_overlap(churn_frame):
    train, valid, test = split_data(churn_frame)

    assert len(train) + len(valid) + len(test) == len(churn_frame)
    assert len(valid) == len(test)
    ids = [set(part["customerID"]) for part in (train, valid, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]


def test_split_is_stratified(churn_frame):
    train, valid, test = split_data(churn_frame)
    overall = (churn_frame["Churn"] == "Yes").mean()

    for part in (train, valid, test):
        assert abs((part["Churn"] == "Yes").mean() - overall) < 0.03


def test_prepare_xy_encodes_target_and_drops_leakage(churn_frame):
    features, target = prepare_xy(churn_frame)

    assert "Churn" not in features.columns
    assert "customerID" not in features.columns
    assert set(target.unique()) <= {0, 1}
    assert target.dtype.kind in "iu"


def test_preprocessor_output_is_finite(churn_frame):
    features, _ = prepare_xy(churn_frame)
    transformer = build_preprocessor()

    matrix = transformer.fit_transform(features)

    assert matrix.shape[0] == len(features)
    assert np.isfinite(matrix).all()
    expected = len(NUMERIC_FEATURES) + sum(
        features[column].nunique() for column in CATEGORICAL_FEATURES
    )
    assert matrix.shape[1] == expected


def test_preprocessor_handles_unseen_categories(churn_frame):
    features, _ = prepare_xy(churn_frame)
    transformer = build_preprocessor().fit(features)

    unseen = features.head(3).copy()
    unseen["Contract"] = "Three year"
    matrix = transformer.transform(unseen)

    assert np.isfinite(matrix).all()
