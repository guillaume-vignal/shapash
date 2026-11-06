import numpy as np
import pandas as pd
import pytest

from shapash import SmartExplainer


@pytest.mark.parametrize("output_space", [None, "score", "probability"])
def test_precomputed_waterfall_distinguishes_probability_values_and_scores(output_space):
    x = pd.DataFrame({"first": [1.0, 2.0], "second": [3.0, 4.0]}, index=["a", "b"])
    contributions = [
        pd.DataFrame([[0.1, -0.1], [0.1, -0.1]], index=x.index, columns=x.columns),
        pd.DataFrame([[-0.1, 0.1], [-0.1, 0.1]], index=x.index, columns=x.columns),
    ]
    explain_data = {"contributions": contributions, "base_values": np.array([0.2, 0.8])}
    if output_space is not None:
        explain_data["output_space"] = output_space

    xpl = SmartExplainer(task="classification", classes=[0, 1])
    xpl.compile(x=x, y_pred=pd.Series([1, 1], index=x.index), contributions=explain_data)
    _, probabilities, baseline, scores, base_scores = xpl.plot._get_waterfall_classification_coupled_outputs(
        ["a"], xpl.explainer.data
    )
    figure = xpl.plot.local_plot(index="a", label=1, plot_type="waterfall", show_predict=False)

    if output_space == "probability":
        np.testing.assert_allclose(probabilities, scores)
        np.testing.assert_allclose(baseline, base_scores)
        assert figure.data[0].customdata[-1].endswith("Probability-scale sum: <b>0.8000</b>")
    else:
        np.testing.assert_allclose(probabilities, np.exp(scores) / np.exp(scores).sum())
        assert "Proba:" in figure.data[0].customdata[-1]
