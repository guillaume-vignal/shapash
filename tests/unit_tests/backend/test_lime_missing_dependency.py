import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from shapash import SmartExplainer
from shapash.backend import lime_backend


def test_missing_lime_is_reported_when_computing(monkeypatch):
    x = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    model = Ridge().fit(x, [1.0, 2.0, 3.0])
    monkeypatch.setattr(lime_backend, "is_lime_available", False)
    explainer = SmartExplainer(model=model, backend="lime")

    with pytest.raises(ImportError, match=r"pip install 'shapash\[lime\]'"):
        explainer.compile(x=x)


def test_missing_lime_does_not_block_precomputed_contributions(monkeypatch):
    x = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    model = Ridge().fit(x, [1.0, 2.0, 3.0])
    monkeypatch.setattr(lime_backend, "is_lime_available", False)
    explainer = SmartExplainer(model=model, backend="lime")

    explainer.compile(x=x, contributions=pd.DataFrame(np.ones((3, 1)), columns=x.columns))
    assert explainer.explainer.contributions.shape == x.shape


@pytest.mark.parametrize("missing", ["lime", "missing_lime_dependency"])
def test_lime_import_only_suppresses_missing_lime(missing):
    code = textwrap.dedent(
        f"""
        import builtins
        import importlib.abc
        import sys

        class BlockLime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "lime":
                    raise ModuleNotFoundError("missing dependency", name={missing!r})
                return None

        sys.meta_path.insert(0, BlockLime())
        try:
            import shapash.backend.lime_backend as backend
        except ModuleNotFoundError as error:
            assert error.name == {missing!r}
            assert {missing!r} != "lime"
        else:
            assert {missing!r} == "lime"
            assert backend.is_lime_available is False
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
