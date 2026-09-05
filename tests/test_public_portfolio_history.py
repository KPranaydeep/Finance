from public_portfolio_history import build_allocation_change_rows


def test_allocation_changes_skip_voided_versions_and_compare_active_weights():
    publications=[
        {"publication_id":"PUB-3","portfolio_version":3,"effective_status":"ACTIVE","published_at":"later"},
        {"publication_id":"PUB-2","portfolio_version":2,"effective_status":"VOIDED_DUPLICATE","published_at":"middle"},
        {"publication_id":"PUB-1","portfolio_version":1,"effective_status":"ACTIVE","published_at":"first"},
    ]
    positions=[
        {"publication_id":"PUB-1","ticker":"A","target_weight":.6},
        {"publication_id":"PUB-1","ticker":"B","target_weight":.4},
        {"publication_id":"PUB-2","ticker":"A","target_weight":.6},
        {"publication_id":"PUB-2","ticker":"B","target_weight":.4},
        {"publication_id":"PUB-3","ticker":"A","target_weight":.5},
        {"publication_id":"PUB-3","ticker":"C","target_weight":.5},
    ]
    rows=build_allocation_change_rows(publications,positions)
    assert len(rows) == 1
    assert rows[0]["Change"] == "P001 → P003"
    assert rows[0]["Added"] == "C"
    assert rows[0]["Removed"] == "B"
    assert rows[0]["Decreased"] == 1
    assert rows[0]["Target turnover"] == .5
