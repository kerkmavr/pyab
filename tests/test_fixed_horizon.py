import pytest

from fixed_horizon import ChiSquaredFixedHorizonTest


def test_ChiSquaredFixedHorizonTest_get_sample_size():

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


def test_ChiSquaredFixedHorizonTest_get_test_length_in_days():
    chi_squared = ChiSquaredFixedHorizonTest()
    _, number_of_days = chi_squared.get_test_length_in_days(
        base_rate=0.2, mde=0.05, n_samples_per_day=1000
    )
    _, number_of_days_more_samples = chi_squared.get_test_length_in_days(
        base_rate=0.2, mde=0.05, n_samples_per_day=1500
    )
    assert number_of_days > number_of_days_more_samples

    _, number_of_days_fewer_samples = chi_squared.get_test_length_in_days(
        base_rate=0.2, mde=0.05, n_samples_per_day=500
    )
    assert number_of_days < number_of_days_fewer_samples

    # invalid n_samples_per_day
    with pytest.raises(ValueError):
        chi_squared.get_test_length_in_days(
            base_rate=0.2, mde=0.05, n_samples_per_day=0
        )
        chi_squared.get_test_length_in_days(
            base_rate=0.2, mde=0.05, n_samples_per_day=-100
        )
