import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder

pytest.importorskip("lime")

from lime import lime_tabular
from lime.discretize import DecileDiscretizer, QuartileDiscretizer

from shapash import SmartExplainer
from shapash.backend.lime_backend import LimeBackend


@pytest.fixture
def data():
    x = pd.DataFrame(
        {"value": np.linspace(-2, 2, 30), "category": np.arange(30) % 3},
        index=np.arange(100, 130),
    )
    return x


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_lime_options_are_forwarded_for_each_task(data, task):
    if task == "regression":
        model = Ridge().fit(data, np.sin(data["value"]))
    else:
        n_classes = 2 if task == "binary" else 3
        model = LogisticRegression().fit(data, np.arange(len(data)) % n_classes)
    original_init = lime_tabular.LimeTabularExplainer.__init__
    original_explain = lime_tabular.LimeTabularExplainer.explain_instance
    init_calls = []
    explain_calls = []
    sample_counts = []

    def record_init(instance, *args, **kwargs):
        init_calls.append(kwargs.copy())
        original_init(instance, *args, **kwargs)

    def record_explain(instance, *args, **kwargs):
        explain_calls.append(kwargs.copy())
        row, predict_fn = args

        def record_predictions(values):
            sample_counts.append(len(values))
            return predict_fn(values)

        return original_explain(instance, row, record_predictions, **kwargs)

    explainer = SmartExplainer(
        model=model,
        backend="lime",
        data=data,
        categorical_features=["category"],
        random_state=42,
        num_samples=123,
        kernel_width=1.5,
        discretize_continuous=True,
        discretizer="decile",
    )
    with (
        patch.object(lime_tabular.LimeTabularExplainer, "__init__", record_init),
        patch.object(lime_tabular.LimeTabularExplainer, "explain_instance", record_explain),
    ):
        explainer.compile(x=data.iloc[:2])

    assert init_calls[0]["random_state"] == 42
    assert init_calls[0]["kernel_width"] == 1.5
    assert init_calls[0]["discretize_continuous"] is True
    assert init_calls[0]["discretizer"] == "decile"
    assert init_calls[0]["categorical_features"] == [1]
    assert len(explain_calls) == 2
    assert all(call["num_samples"] == 123 and call["num_features"] == 2 for call in explain_calls)
    assert sample_counts == [123, 123]
    if task == "multiclass":
        assert all(call["top_labels"] == 3 for call in explain_calls)
    assert isinstance(explainer.explainer.backend.explainer.discretizer, DecileDiscretizer)


def test_lime_defaults_are_not_overridden(data):
    model = Ridge().fit(data, np.sin(data["value"]))
    original_init = lime_tabular.LimeTabularExplainer.__init__
    original_explain = lime_tabular.LimeTabularExplainer.explain_instance
    init_calls = []
    explain_calls = []
    sample_counts = []

    def record_init(instance, *args, **kwargs):
        init_calls.append(kwargs.copy())
        original_init(instance, *args, **kwargs)

    def record_explain(instance, *args, **kwargs):
        explain_calls.append(kwargs.copy())
        row, predict_fn = args

        def record_predictions(values):
            sample_counts.append(len(values))
            return predict_fn(values)

        return original_explain(instance, row, record_predictions, **kwargs)

    backend = LimeBackend(model)
    with (
        patch.object(lime_tabular.LimeTabularExplainer, "__init__", record_init),
        patch.object(lime_tabular.LimeTabularExplainer, "explain_instance", record_explain),
    ):
        backend.run_explainer(data.iloc[:1])

    for key in ("random_state", "kernel_width", "discretize_continuous", "discretizer"):
        assert key not in init_calls[0]
    assert "num_samples" not in explain_calls[0]
    assert sample_counts == [5000]
    assert isinstance(backend.explainer.discretizer, QuartileDiscretizer)


def test_lime_options_preserve_onehot_groups_without_discretization():
    x = pd.DataFrame({"value": np.arange(30, dtype=float), "category": ["a", "b", "c"] * 10})
    encoder = ColumnTransformer(
        [("category", OneHotEncoder(sparse_output=False), ["category"])], remainder="passthrough"
    ).fit(x)
    encoded = pd.DataFrame(encoder.transform(x), columns=["a", "b", "c", "value"])
    model = Ridge().fit(encoded, np.sin(x["value"]))
    predict = model.predict
    observed = []

    def checked_predict(values):
        observed.append(len(values))
        groups = values[["a", "b", "c"]].to_numpy()
        assert np.isin(groups, [0, 1]).all()
        np.testing.assert_array_equal(groups.sum(axis=1), 1)
        return predict(values)

    backend = LimeBackend(
        model, preprocessing=encoder, data=encoded, random_state=42, num_samples=250, discretize_continuous=False
    )
    with patch.object(model, "predict", checked_predict):
        result = backend.run_explainer(encoded.iloc[:1])
    assert observed == [250]
    assert backend._lime_input.categorical_indices == [0]
    assert result["contributions"].shape == (1, 2)


def test_lime_seed_and_disabled_discretization_survive_export(data):
    model = Ridge().fit(data, np.sin(data["value"]))
    options = dict(
        data=data,
        categorical_features=["category"],
        random_state=42,
        num_samples=150,
        kernel_width=0.8,
        discretize_continuous=False,
    )
    first = SmartExplainer(model=model, backend="lime", **options)
    second = SmartExplainer(model=model, backend="lime", **options)
    first.compile(x=data.iloc[:2])
    second.compile(x=data.iloc[:2])
    pd.testing.assert_frame_equal(first.explainer.contributions, second.explainer.contributions)
    assert first.explainer.backend.explainer.discretizer is None

    predictor = pickle.loads(pickle.dumps(first.to_smartpredictor()))
    backend = predictor.backend
    assert backend.explainer is None
    assert backend.random_state == 42
    assert backend.num_samples == 150
    assert backend.kernel_width == 0.8
    assert backend.discretize_continuous is False
    predictor.add_input(x=data.iloc[:2])
    pd.testing.assert_frame_equal(predictor.data["contributions"], first.explainer.contributions)
    assert backend.explainer.discretizer is None


def test_lime_old_pickle_without_new_options_still_works(data):
    model = Ridge().fit(data, np.sin(data["value"]))
    backend = LimeBackend(model, data=data)
    for name in ("random_state", "num_samples", "kernel_width", "discretize_continuous", "discretizer"):
        delattr(backend, name)
    restored = pickle.loads(pickle.dumps(backend))
    result = restored.run_explainer(data.iloc[:1])
    assert result["contributions"].shape == (1, 2)
    assert isinstance(restored.explainer.discretizer, QuartileDiscretizer)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": True}, "num_samples"),
        ({"num_samples": 1.5}, "num_samples"),
        ({"kernel_width": 0}, "kernel_width"),
        ({"kernel_width": float("nan")}, "kernel_width"),
        ({"discretize_continuous": "false"}, "discretize_continuous"),
    ],
)
def test_lime_invalid_options_fail_explicitly(data, options, message):
    model = Ridge().fit(data, np.sin(data["value"]))
    with pytest.raises(ValueError, match=message):
        LimeBackend(model, **options)


def test_lime_unknown_options_are_not_silently_ignored(data):
    model = Ridge().fit(data, np.sin(data["value"]))
    with pytest.raises(TypeError, match="random_sate"):
        LimeBackend(model, random_sate=42)
