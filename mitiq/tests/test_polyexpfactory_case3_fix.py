"""
Regression test for PolyExpFactory.extrapolate() CASE 3 bug fix.

⚠️ DISCLAIMER: This test was generated with AI assistance (GitHub Copilot).

Tests that zero-noise limit error is correctly computed for ExpFactory
with a known asymptote and avoid_log=False (logarithmic linearization path).

Issue: Variable name mismatch and incorrect covariance matrix indexing
prevented zne_error calculation in CASE 3 of PolyExpFactory.extrapolate().
"""

import numpy as np
import pytest
from mitiq.zne.inference import ExpFactory


def test_exp_factory_with_asymptote_avoid_log_false_full_output():
    """Test that ExpFactory computes zne_error in CASE 3 (asymptote + log transform).
    
    Before fix: zne_error would be None due to variable name mismatch (param_cov vs params_cov)
               and incorrect covariance matrix indexing.
    After fix: zne_error is numeric and non-None.
    """
    # Setup: Create ExpFactory with asymptote and avoid_log=False
    scale_factors = [1.0, 2.0, 3.0, 4.0]
    asymptote = 0.5
    order = 1  # Used in PolyExpFactory via ExpFactory
    
    factory = ExpFactory(
        scale_factors=scale_factors,
        asymptote=asymptote,
        avoid_log=False
    )
    
    # Push realistic (noise-scaled) expectation values
    exp_values = [0.8, 0.65, 0.58, 0.54]
    for scale_factor, exp_value in zip(scale_factors, exp_values):
        factory.push({"scale_factor": scale_factor}, exp_value)
    
    # Reduce and extract full output
    factory.reduce()
    
    # Assert zne_limit is returned
    zne_limit = factory.get_zero_noise_limit()
    assert zne_limit is not None
    assert isinstance(zne_limit, (float, np.floating))
    assert not np.isnan(zne_limit)
    
    # Assert zne_error is computed (not None) - THIS IS THE KEY FIX TEST
    zne_error = factory.get_zero_noise_limit_error()
    assert zne_error is not None, (
        "zne_error should be computed for ExpFactory with asymptote and avoid_log=False. "
        "This suggests the CASE 3 covariance calculation bug is not fixed."
    )
    assert isinstance(zne_error, (float, np.floating))
    assert zne_error > 0, "zne_error should be positive"
    assert not np.isnan(zne_error)


def test_exp_factory_with_asymptote_avoid_log_false_full_output_tuple():
    """Test that full_output tuple includes non-None zne_error for CASE 3."""
    scale_factors = [1.0, 2.0, 3.0, 4.0]
    asymptote = 0.5
    exp_values = [0.8, 0.65, 0.58, 0.54]
    
    # Use static extrapolate method with full_output=True
    zne_limit, zne_error, opt_params, params_cov, zne_curve = ExpFactory.extrapolate(
        scale_factors,
        exp_values,
        asymptote=asymptote,
        avoid_log=False,
        full_output=True
    )
    
    # Assert all components are returned
    assert zne_limit is not None
    assert zne_error is not None, (
        "zne_error should be non-None for CASE 3. "
        "Check that params_cov variable is correctly assigned and indexed."
    )
    assert opt_params is not None
    assert params_cov is not None
    assert zne_curve is not None
    
    # Assert error is numeric
    assert isinstance(zne_error, (float, np.floating))
    assert zne_error > 0


def test_exp_factory_avoid_log_true_unaffected():
    """Verify that CASE 2 (avoid_log=True) is unaffected by the fix."""
    scale_factors = [1.0, 2.0, 3.0, 4.0]
    asymptote = 0.5
    exp_values = [0.8, 0.65, 0.58, 0.54]
    
    zne_limit, zne_error, opt_params, params_cov, zne_curve = ExpFactory.extrapolate(
        scale_factors,
        exp_values,
        asymptote=asymptote,
        avoid_log=True,  # CASE 2, not affected by CASE 3 bug
        full_output=True
    )
    
    # CASE 2 should also work
    assert zne_limit is not None
    assert zne_error is not None
    assert isinstance(zne_error, (float, np.floating))


if __name__ == "__main__":
    test_exp_factory_with_asymptote_avoid_log_false_full_output()
    test_exp_factory_with_asymptote_avoid_log_false_full_output_tuple()
    test_exp_factory_avoid_log_true_unaffected()
    print("✓ All regression tests passed!")
