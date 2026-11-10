"""Automatic personal workspace selection; nickname profiles are not authentication."""


def resolve_personal_profile(conn, *, configured="", session_profile="", shared_owner="__universal__"):
    conn.execute("CREATE TABLE IF NOT EXISTS personal_workspace_preferences (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    pinned = conn.execute("SELECT value FROM personal_workspace_preferences WHERE key='profile'").fetchone()
    profile = str(configured or session_profile or (pinned[0] if pinned else "")).strip()
    if not profile:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        owners = set()
        for name in ("master_holdings","latest_analysis"):
            if name in tables:
                owners.update(str(row[0]).strip() for row in conn.execute(f"SELECT DISTINCT owner FROM {name}") if row[0])
        owners = {owner for owner in owners if owner.lower() != shared_owner.lower()}
        if len(owners) > 1:
            raise ValueError("Multiple saved profiles exist. Set PORTFOLIO_PERSONAL_PROFILE once to select the personal workspace; no profiles were merged.")
        profile = next(iter(owners)) if owners else "Personal"
    if not profile or profile.lower() == shared_owner.lower():
        raise ValueError("The shared universal portfolio cannot be used as a personal profile")
    conn.execute("INSERT INTO personal_workspace_preferences VALUES('profile',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(profile,))
    conn.commit()
    return profile
