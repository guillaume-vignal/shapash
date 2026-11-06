import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

pytest.importorskip("lime")

from lime import lime_tabular

from shapash import SmartExplainer


@pytest.fixture
def data():
    rng = np.random.RandomState(7)
    return pd.DataFrame(rng.normal(size=(90, 3)), columns=["a", "b", "c"], index=np.arange(100, 190))


def test_binary_lime_baselines_reconstruct_complementary_local_surrogates(data):
    model = LogisticRegression().fit(data, np.where(data["a"] > 0, "yes", "no"))
    sample = data.iloc[:2]
    original_explain = lime_tabular.LimeTabularExplainer.explain_instance
    explanations = []

    def capture(instance, *args, **kwargs):
        explanation = original_explain(instance, *args, **kwargs)
        explanations.append(explanation)
        return explanation

    xpl = SmartExplainer(model=model, backend="lime", data=data, random_state=9, num_samples=400)
    with patch.object(lime_tabular.LimeTabularExplainer, "explain_instance", capture):
        xpl.compile(x=sample)

    result = xpl.explainer.explain_data
    assert result["output_space"] == "probability"
    assert result["explanation_type"] == "local_surrogate"
    for i, explanation in enumerate(explanations):
        positive_base = explanation.intercept[1]
        np.testing.assert_allclose(result["base_values"][i], [1 - positive_base, positive_base])
        negative, positive = (frame.iloc[i].sum() for frame in xpl.explainer.contributions)
        np.testing.assert_allclose(negative, -positive)
        np.testing.assert_allclose(positive_base + positive, explanation.local_pred[0])
        np.testing.assert_allclose(result["base_values"][i, 0] + negative, 1 - explanation.local_pred[0])


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("discretize", [True, False])
def test_lime_waterfall_uses_probability_scale_without_softmax(data, n_classes, discretize):
    target = np.arange(len(data)) % n_classes
    model = LogisticRegression(max_iter=300).fit(data, target)
    xpl = SmartExplainer(
        model=model,
        backend="lime",
        data=data,
        random_state=42,
        num_samples=300,
        discretize_continuous=discretize,
    )
    xpl.compile(x=data.iloc[:2])
    xpl.filter(max_contrib=1)
    line = [data.index[0]]

    with patch.object(xpl.plot, "_softmax_from_scores", side_effect=AssertionError("unexpected softmax")):
        _, values, bases, scores, base_scores = xpl.plot._get_waterfall_classification_coupled_outputs(
            line, xpl.explainer.data
        )
        np.testing.assert_array_equal(values, scores)
        np.testing.assert_array_equal(bases, base_scores)
        for label in model.classes_:
            fig = xpl.plot.local_plot(
                index=line[0], label=label, plot_type="waterfall", show_predict=True, show_masked=False
            )
            trace = fig.data[0]
            assert len(trace.x) == 3
            assert trace.y[-1] == "<i>Sum of displayed weights</i>"
            assert fig.layout.xaxis.title.text == "Local surrogate values"
            assert "Baseline + displayed weights" in trace.customdata[-1]
            assert "LIME sum on probability scale" in trace.customdata[-1]
            assert "Model output" not in trace.customdata[-1]
            assert "Proba:" not in trace.customdata[-1]
            assert all("Cumulative output" not in item for item in trace.customdata)
            expected = trace.x[0] + sum(trace.x[1:-1])
            assert trace.x[-1] == pytest.approx(expected)
            assert trace.customdata[-1].endswith(f"<b>{expected:.4f}</b>")
            model_probability = model.predict_proba(data.iloc[:1])[0, int(label)]
            assert f"Proba: <b>{model_probability:.4f}</b>" in fig.layout.title.text


def test_lime_regression_waterfall_labels_local_weight_sum(data):
    model = Ridge().fit(data, np.sin(data["a"]))
    xpl = SmartExplainer(model=model, backend="lime", data=data, random_state=42, num_samples=200)
    xpl.compile(x=data.iloc[:1])
    figure = xpl.plot.local_plot(index=data.index[0], plot_type="waterfall", show_predict=False)
    assert figure.data[0].y[-1] == "<i>Sum of displayed weights</i>"
    assert "Baseline + displayed weights" in figure.data[0].customdata[-1]
    assert figure.layout.xaxis.title.text == "Local surrogate values"


def test_legacy_lime_explanation_metadata_avoids_softmax(data):
    model = LogisticRegression().fit(data, np.arange(len(data)) % 2)
    xpl = SmartExplainer(model=model, backend="lime", data=data, random_state=42, num_samples=200)
    xpl.compile(x=data.iloc[:1])
    del xpl.explainer.explain_data["output_space"]
    del xpl.explainer.explain_data["explanation_type"]
    with patch.object(xpl.plot, "_softmax_from_scores", side_effect=AssertionError("unexpected softmax")):
        figure = xpl.plot.local_plot(index=data.index[0], label=0, plot_type="waterfall")
    assert figure.data[0].y[-1] == "<i>Sum of displayed weights</i>"
    assert "LIME sum on probability scale" in figure.data[0].customdata[-1]


def test_reused_lime_explanation_retains_probability_scale(data):
    model = LogisticRegression().fit(data, np.arange(len(data)) % 2)
    original = SmartExplainer(model=model, backend="lime", data=data, random_state=42, num_samples=200)
    original.compile(x=data.iloc[:1])
    saved_data = pickle.loads(pickle.dumps(original.explainer.explain_data))

    reused = SmartExplainer(model=model, backend="lime", data=data)
    reused.compile(x=data.iloc[:1], contributions=saved_data)
    with patch.object(reused.plot, "_softmax_from_scores", side_effect=AssertionError("unexpected softmax")):
        figure = reused.plot.local_plot(index=data.index[0], label=0, plot_type="waterfall")
    assert figure.data[0].x[0] == pytest.approx(saved_data["base_values"][0, 0])
    assert "LIME sum on probability scale" in figure.data[0].customdata[-1]
