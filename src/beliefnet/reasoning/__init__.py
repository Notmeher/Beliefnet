"""Sound reasoning layer: exact feasibility + certificates (unsat core, MCS).

Unlike the differentiable ``StructuredEquilibriumCore`` (which produces a soft
residual score), this package decides joint satisfiability *exactly*, so a
positive contradiction verdict is a proof rather than a score. The two are
complementary: the energy gives a smooth training signal and calibrated
confidence; the exact checker gives the certificate and the soundness
guarantee.
"""
