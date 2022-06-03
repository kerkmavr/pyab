import pytest

from fixed_horizon import ChiSquaredFixedHorizonTest


def test_ChiSquarepychadFixedHorizonTest_get_sample_size():

    chi_squared = ChiSquaredFixedHorizonTest()
    samples_per_variant = chi_squared.get_sample_size(
        base_rate=0.26, mde=0.05, alpha=0.05, power=0.8
    )
    assert samples_per_variant == 18153

    # test w/ higher power
    samples_per_variant_higher_power = chi_squared.get_sample_size(
        base_rate=0.26, mde=0.05, alpha=0.05, power=0.9
    )
    assert samples_per_variant_higher_power > samples_per_variant

    # test w/ higher minimum detectable effect
    samples_per_variant_higher_mde = chi_squared.get_sample_size(
        base_rate=0.26, mde=0.1, alpha=0.05, power=0.9
    )
    assert samples_per_variant_higher_mde < samples_per_variant_higher_power

    # test w/ higher significance level / lower confidence level
    samples_per_variant_higher_aplha = chi_squared.get_sample_size(
        base_rate=0.26, mde=0.1, alpha=0.1, power=0.9
    )
    assert samples_per_variant_higher_aplha < samples_per_variant_higher_mde

    # invalid base rate values
    with pytest.raises(ValueError):
        chi_squared.get_sample_size(base_rate=-23, mde=0.05)
        chi_squared.get_sample_size(base_rate=23, mde=0.05)

    # invalid mde values
    with pytest.raises(ValueError):
        chi_squared.get_sample_size(base_rate=0.2, mde=0)
        chi_squared.get_sample_size(base_rate=0.2, mde=-10)

    # invalid alpha values
    with pytest.raises(ValueError):
        chi_squared.get_sample_size(base_rate=0.2, mde=0.05, alpha=0.12)
        chi_squared.get_sample_size(base_rate=0.2, mde=0.05, alpha=0.001)

    # invalid power values
    with pytest.raises(ValueError):
        chi_squared.get_sample_size(base_rate=0.2, mde=0.05, power=0.55)
        chi_squared.get_sample_size(base_rate=0.2, mde=0.05, power=0.99)
