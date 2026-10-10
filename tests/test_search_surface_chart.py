import plotly.graph_objects as go

from search_surface_chart import build_search_surface_figure


def _rows():
    return [
        {
            "Shortlist cap": cap,
            "Maximum assets": assets,
            "Annual Return": annual_return,
            "Trading days": 444,
            "Assets": assets,
            "Status": status,
        }
        for cap, assets, annual_return, status in (
            (8500, 1000, 0.88, "Feasible"),
            (8500, 1050, 0.89, "Feasible"),
            (8550, 1000, 0.90, "Feasible"),
            (8550, 1050, 0.91, "Feasible"),
            (8600, 1100, 0.99, "Solver failed"),
        )
    ]


def test_surface_uses_only_feasible_observed_grid_and_marks_decisions():
    rows = _rows()
    figure = build_search_surface_figure(
        rows,
        raw_peak=rows[3],
        robust_choice=rows[2],
    )
    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == [
        "Observed return surface",
        "Feasible observations",
        "Raw peak",
        "Robust choice",
    ]
    observations = figure.data[1]
    assert len(observations.x) == 4
    assert 8600 not in observations.x
    assert list(figure.data[2].z) == [91.0]
    assert list(figure.data[3].z) == [90.0]


def test_sparse_results_remain_points_without_inventing_a_surface():
    figure = build_search_surface_figure([_rows()[0]])
    assert [trace.name for trace in figure.data] == ["Feasible observations"]


def test_empty_results_render_an_empty_figure():
    assert len(build_search_surface_figure([]).data) == 0
