# Global-stock benchmark

Upload the included root Python files and preserve the tests/ path. Commit and refresh Streamlit after deployment. No database migration or new secret is required.

The comparison uses Vanguard Total World Stock ETF (VT), an investable FTSE Global All Cap proxy, with Yahoo Finance adjusted closes converted using INR=X (INR per USD). Dividend adjustment approximates reinvestment; this is ETF performance rather than the official index series.

The selected performance period controls the chart. Portfolio NAV, VT and FX must all have observations on the comparison date. Both series begin at 100 on the first common date. Missing prices are not filled. Returns beneath the chart use exactly this shared period. US closes occur later than Indian closes; the chart is an end-of-day calendar-date comparison.

Benchmark downloads are cached for one hour. If a download fails, the comparison displays an unavailable message and the portfolio remains visible.

Three unit tests and syntax checks passed. Live Yahoo data and Streamlit rendering were not exercised locally.
