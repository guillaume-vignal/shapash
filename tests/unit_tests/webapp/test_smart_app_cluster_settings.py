"""Regression tests for the collapsible cluster settings panel."""

import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from shapash import SmartExplainer
from shapash.webapp.smart_app import SmartApp


def find_component(node, component_id):
    if isinstance(node, dict):
        if node.get("props", {}).get("id") == component_id:
            return node["props"]
        for value in node.get("props", {}).values():
            found = find_component(value, component_id)
            if found is not None:
                return found
    elif isinstance(node, list):
        for child in node:
            found = find_component(child, component_id)
            if found is not None:
                return found
    return None


class TestClusterSettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.normal(size=(20, 2)), columns=["feat", "other_feat"])
        y_pred = pd.DataFrame({"pred": rng.normal(size=20)})
        model = DecisionTreeRegressor(max_depth=2).fit(x, y_pred)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=x, contributions=x.copy(), y_pred=y_pred)
        cls.smart_app = SmartApp(xpl.explainer)
        cls.client = cls.smart_app.server.test_client()

    def test_initial_layout_keeps_toggle_outside_settings(self):
        response = self.client.get("/_dash-layout")
        self.assertEqual(response.status_code, 200)
        layout = response.get_json()
        graph_column = find_component(layout, "clusters_graph_column")
        settings = find_component(layout, "clusters_settings")
        button = find_component({"props": graph_column}, "toggle_clusters_settings")

        self.assertEqual(graph_column["width"], 9)
        self.assertEqual(settings["width"], 3)
        self.assertEqual(settings["style"], {"height": "100%"})
        self.assertEqual(button["aria-expanded"], "true")
        self.assertEqual(button["aria-controls"], "clusters_settings")
        self.assertEqual(button["type"], "button")
        self.assertIsNone(find_component({"props": settings}, "toggle_clusters_settings"))
        self.assertIs(find_component(layout, "clusters")["responsive"], True)
        for component_id in ("toggle_points_on", "toggle_points_off", "color_param_clusters", "n_clusters_slider"):
            self.assertIsNotNone(find_component({"props": settings}, component_id))

    def test_repeated_toggles_only_change_panel_visibility_and_graph_width(self):
        outputs = [
            {"id": "clusters_settings", "property": "style"},
            {"id": "clusters_graph_column", "property": "width"},
            {"id": "clusters_settings_icon", "property": "children"},
            {"id": "toggle_clusters_settings", "property": "title"},
            {"id": "toggle_clusters_settings", "property": "aria-label"},
            {"id": "toggle_clusters_settings", "property": "aria-expanded"},
        ]
        output_key = ".." + "...".join(f"{out['id']}.{out['property']}" for out in outputs) + ".."
        for clicks in (None, 0, 1, 2, 3, 4):
            with self.subTest(clicks=clicks):
                response = self.client.post(
                    "/_dash-update-component",
                    json={
                        "output": output_key,
                        "outputs": outputs,
                        "inputs": [{"id": "toggle_clusters_settings", "property": "n_clicks", "value": clicks}],
                        "state": [],
                        "changedPropIds": ["toggle_clusters_settings.n_clicks"],
                    },
                )
                self.assertEqual(response.status_code, 200)
                result = response.get_json()["response"]
                collapsed = bool(clicks and clicks % 2)
                label = "Show cluster settings" if collapsed else "Hide cluster settings"
                self.assertEqual(
                    result,
                    {
                        "clusters_settings": {"style": {"display": "none"} if collapsed else {"height": "100%"}},
                        "clusters_graph_column": {"width": 12 if collapsed else 9},
                        "clusters_settings_icon": {"children": "chevron_left" if collapsed else "chevron_right"},
                        "toggle_clusters_settings": {
                            "title": label,
                            "aria-label": label,
                            "aria-expanded": "false" if collapsed else "true",
                        },
                    },
                )


if __name__ == "__main__":
    unittest.main()
