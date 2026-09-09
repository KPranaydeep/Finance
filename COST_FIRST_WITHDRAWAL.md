# Cost-first withdrawal

Replace the included files, preserving tests/. The solver now minimizes estimated selling charges first, then fixes that minimum cost and uses target-allocation deviation only to select among equally cheap plans. It never accepts a higher fee to improve diversification.

The model still requires the requested net cash after charges, uses available model cash first and cannot sell more shares than it holds. All fees remain estimates using the existing assumptions. A concentrated sale may produce greater allocation drift.

If the second tie-breaking optimization times out, the first proven minimum-cost plan is retained and reconciled. If the first optimization does not establish an optimum, no plan is presented as optimal.
