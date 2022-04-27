from __future__ import annotations

import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.proportion
import tqdm

from pyab.core.base import ABTestABC

if typing.TYPE_CHECKING:
    from typing import Any, Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class ChiSquaredFixedHorizonTest(ABTestABC):

    @staticmethod
    def get_sample_size(base_rate: float, mde: float, alpha: float = 0.05, beta: float = 0.8) -> int:
        """
        Returns the number of samples required (per variant) before stopping th AB test.

        Args:
            base_prob_conv: the conversion rate for the "control" variant (the one currently in production)
            mde: The Minimum Detectable Effect is the smallest effect that will be detected (1-\\beta)\% of the time.
            alpha: a float representing the percentage of the time a difference will be detected, assuming one does NOT exist
            beta: a float representing the percentage of the time the minimum effect size will be detected, assuming it exists

        Returns:
            an integer, the number of samples per variation required before stopping th AB test.
        """
        factor = (scipy.stats.norm.ppf(1 - alpha / 2) + scipy.stats.norm.ppf(beta)) ** 2
        variance = base_rate * (1 - base_rate) + (base_rate * (1 + mde) * (1 - base_rate * (1 + mde)))
        err = (base_rate * mde) ** 2
        sample_size = int(factor * variance / err)
        logger.info(f"MDE: {mde} (relative), {mde * base_rate} (absolute)")
        logger.info(f"NHST samples required (per variant): {sample_size}")
        return sample_size

    def get_test_length_in_days(
        self,
        *,
        base_rate: float,
        mde: float,
        n_samples_per_day: float,
        alpha: float = 0.05,
        beta: float = 0.8,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Returns the number of days you are expected to run your AB test for in order to ensure a statistical significance
        of \(\\alpha\) and a power of \(\\beta\).

        Args:
            base_prob_conv: the conversion rate for the "control" variant (the one currently in production)
            mde: The Minimum Detectable Effect is the smallest effect that will be detected \( (1-\\beta) \) % of the time.
            n_samples_per_day: an integer, the amount of collected samples per day (total)
            alpha: a float representing the percentage of the time a difference will be detected, assuming one does NOT exist
            beta: a float representing the percentage of the time the minimum effect size will be detected, assuming it exists

        Returns:
            an integer, the number of samples per variation required before stopping th AB test.
        """
        sample_size = self.get_sample_size(base_rate=base_rate, mde=mde, alpha=alpha, beta=beta, **kwargs)
        wait_time_in_days = int(sample_size * 2 / n_samples_per_day) + 1
        logger.info(f"MDE: {mde} (relative), {mde * base_rate} (absolute)")
        logger.info(f"NHST samples required (per variant): {sample_size} -> {wait_time_in_days} days")
        return sample_size, wait_time_in_days

    @staticmethod
    def get_pvalues(
        conversions_a: np.ndarray,
        conversions_b: np.ndarray,
        counts_a: Optional[np.ndarray] = None,
        counts_b: Optional[np.ndarray] = None,
        n_samples_per_day: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the daily pvalues resulting from a chi squared test for a given list of experiments.

        Args:
            conversions_a: `a numpy.ndarray` of shape `(number of experiments, number of days)`. The element at position (i,j) in the array is
                           the number of conversions on the j-th day for the i-th experiment for variant A.
            conversions_b: `a numpy.ndarray` of shape `(number of experiments, number of days)`. The element at position (i,j) in the array is
                           the number of conversions on the j-th day for the i-th experiment for variant B.
            counts_a: `a numpy.ndarray` of shape `(number of experiments, number of days)`. The element at position (i,j) in the array is
                      the number of samples collected on the j-th day for the i-th experiment for variant A.
                      If this value is not provided then `n_samples_per_day // 2` sample are assumed to be collected each day for variant A.
            counts_b: `a numpy.ndarray` of shape `(number of experiments, number of days)`. The element at position (i,j) in the array is
                      the number of samples collected on the j-th day for the i-th experiment for variant B.
                      If this value is not provided then `n_samples_per_day // 2` sample are assumed to be collected each day for variant B.
            n_samples_per_day: an integer, the number of samples collected per day (total). This value has to be provided only if `counts_a` and/or `counts_b` are not passed as parameters.

        Returns:
            `a numpy.ndarray` of shape(number of experiments, number of days). The element at position (i,j) in the array is
            the pvalue of the test performed on the j-th day for the i-th experiment.
        """
        n_experiments, n_days = conversions_a.shape

        if counts_a is None:
            counts_a = np.cumsum(np.ones((n_experiments, n_days)) * n_samples_per_day, axis=1)
        else:
            counts_a = np.cumsum(counts_a, axis=1)
        if counts_b is None:
            counts_b = np.cumsum(np.ones((n_experiments, n_days)) * n_samples_per_day, axis=1)
        else:
            counts_b = np.cumsum(counts_b, axis=1)

        conversions_a = np.cumsum(conversions_a, axis=1)
        conversions_b = np.cumsum(conversions_b, axis=1)

        logger.info("Computing pvalues...")
        pv_per_experiment = []
        for e in range(n_experiments):
            pv_per_day = []
            for d in range(n_days):
                stat, pvalue, tab = statsmodels.stats.proportion.proportions_chisquare(
                    [conversions_a[e, d], conversions_b[e, d]],
                    [counts_a[e, d], counts_b[e, d]],
                )
                pv_per_day.append(pvalue)
            pv_per_experiment.append(pv_per_day)
        return np.array(pv_per_experiment, dtype=np.float32)

    def run_test(self) -> np.ndarray:
        """
        Run an AB testing simulation for `n_days` for `n_experiments` (number of different experiments).
        The main KPI of the test is any kind of Conversion Rate (CR).

        `n_days` and `n_experiments` are defined by the user within `self.data_loader`.

        Returns:
            `a numpy.ndarray` of shape(n_experiments, n_days). The element at position (i,j) in the array is
            the pvalue of the test performed on the j-th day for the i-th experiment.
        """
        (conversions_a, counts_a), (
            conversions_b,
            counts_b,
        ) = self.data_loader.load_data()
        pvalues = self.get_pvalues(conversions_a, conversions_b, counts_a=counts_a, counts_b=counts_b)
        return pvalues

    def plot(
        self,
        arr: np.ndarray,
        hlines: Optional[Tuple[List[float], float]] = ([0.05], 0),
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        n_experiments, n_days = arr.shape

        _ = plt.figure()
        for e in range(n_experiments):
            plt.plot(range(n_days), arr[e])
            # plt.scatter(wait_time_in_days, arr[e][wait_time_in_days], marker='X', color='red')
        if hlines is not None:
            hx, hy = hlines
            plt.hlines(hx, hy, n_days - 1, linestyle="--", color="black")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

    def get_test_type(self) -> str:
        return "Chisq"

    def get_value_a_and_value_b(self) -> Tuple[float, float]:
        (conversions_a, counts_a), (conversions_b, counts_b) = self.data_loader.load_data()
        cr_a = conversions_a.sum() / counts_a.sum()
        cr_b = conversions_b.sum() / counts_b.sum()
        return cr_a, cr_b

    def produce_table_results(
        self, product: str, main_kpi: bool, force_output: bool, params_for_get_test_length_in_days: Dict[str, Any]
    ) -> pd.DataFrame:
        date_range = self.data_loader.get_date_range()
        samples, min_duration = self.get_test_length_in_days(**params_for_get_test_length_in_days)
        cr_a, cr_b = self.get_value_a_and_value_b()

        test_duration = len(self.data_loader.get_ordered_dates())

        if test_duration < min_duration and not force_output:
            p_value = None
            significant = None
        else:
            p_values = self.run_test()
            p_value = p_values[-1][-1]  # TODO check this we could want to enable multiple expnts
            significant = p_value < params_for_get_test_length_in_days["alpha"]

        dfs = []

        for test_symbol, cr in [("P", cr_a), ("T", cr_b)]:
            result_dict = {
                "EXPERIMENT_NAME": [self.name],
                "PRODUCT": [product],
                "APPROACH": ["Frequentist"],
                "START_DATE": [date_range[0]],
                "END_DATE": [date_range[1]],
                "DATE": [date_range[1]],
                "MIN_DURATION": [min_duration],
                "TEST_TYPE": [self.get_test_type()],
                "MAIN_KPI": [main_kpi],
                "KPI": [self.data_loader.get_kpi()],
                "PROFILE": test_symbol,
                "VALUE": [cr],
                "PVALUE": [p_value],
                "EXPECTED_LOSS": [None],
                "LOWER_BOUNDARY": [None],
                "UPPER_BOUNDARY": [None],
                "EXPECTED_LOSS_THRESHOLD": [None],
                "SIGNIFICANCE_FLAG": [significant],
            }
            assert set(self.table_results_columns) == set(result_dict.keys())
            dfs.append(pd.DataFrame(result_dict))

        return pd.concat(dfs)


class MannWhitneyFixedHorizonTest(ChiSquaredFixedHorizonTest):
    """
        The following options are available for the alternative hypothesis (default is None):

          * None: computes p-value half the size of the 'two-sided' p-value and
            a different U statistic. The default behavior is not the same as
            using 'less' or 'greater'; it only exists for backward compatibility
            and is deprecated.
          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided"""

    def __init__(self, *, alternative_hypothesis: Optional[str] = None, **kwargs):
        assert alternative_hypothesis in [None, "two-sided", "less", "greater"]
        self.alternative_hypothesis = alternative_hypothesis
        super(MannWhitneyFixedHorizonTest, self).__init__(**kwargs)

    def get_pvalues(
        self,
        conversions_a: np.ndarray,
        conversions_b: np.ndarray,
        revenues_a: np.ndarray,
        revenues_b: np.ndarray,
    ) -> np.ndarray:
        """

        Args:
            conversions_a: a binary `numpy.ndarray` of shape `(n_experiments, n_days, n_samples_per_day_per_varaint)`.
                           It contains the conversion (or non conversion) for each user on each day and experiment for varaint A.
            conversions_b: a binary `numpy.ndarray` of shape `(n_experiments, n_days, n_samples_per_day_per_varaint)`.
                           It contains the conversion (or non conversion) for each user on each day and experiment for varaint B.
            revenues_a: a `numpy.ndarray` of shape `(n_experiments, n_days, n_samples_per_day_per_variant)`.
                        It contains the revenues for each possible sale for variant A.
                        When multiplied by `conversions_a` it gives you back the real revenues per sale
                        (the non conversions are assigned a 0 revenue) for variant A.
            revenues_b: a `numpy.ndarray` of shape `(n_experiments, n_days, n_samples_per_day_per_varaint)`.
                        It contains the revenues for each possible sale for variant B.
                        When multiplied by `conversions_b` it gives you back the real revenues per sale
                        (the non conversions are assigned a 0 revenue) for variant B.

        Returns:
            `a numpy.ndarray` of shape(n_experiments, n_days). The element at position (i,j) in the array is
            the pvalue of the test performed on the j-th day for the i-th experiment.
        """
        n_experiments, n_days, n_samples_per_day = conversions_a.shape

        logger.info("Computing pvalues...")
        pvalues = []
        for i in tqdm.tqdm(range(1, n_days + 1)):
            a = (conversions_a * revenues_a)[:, : i + 1, :].reshape(n_experiments, -1)
            b = (conversions_b * revenues_b)[:, : i + 1, :].reshape(n_experiments, -1)
            res = [scipy.stats.mannwhitneyu(a[j][~np.isnan(a[j])], b[j][~np.isnan(b[j])], alternative=self.alternative_hypothesis).pvalue for j in range(len(a))]
            pvalues.append(res)
        pvalues = np.array(pvalues).T
        return pvalues

    def run_test(self) -> np.ndarray:
        """
        Run an AB testing simulation for `n_days` for `n_experiments` (number of different experiments).
        The main KPI of the test is Revenue Per User (RPU).

        `n_days` and `n_experiments` are defined by the user within `self.data_loader`.

        Returns:
            `a numpy.ndarray` of shape(n_experiments, n_days). The element at position (i,j) in the array is
            the pvalue of the test performed on the j-th day for the i-th experiment.
        """
        (conversions_a, revenues_a), (
            conversions_b,
            revenues_b,
        ) = self.data_loader.load_data()
        pvalues = self.get_pvalues(
            conversions_a,
            conversions_b,
            revenues_a,
            revenues_b,
        )
        return pvalues

    def get_test_type(self) -> str:
        return "MannWhitney"

    def get_value_a_and_value_b(self) -> Tuple[float, float]:
        (conversions_a, revenues_a), (conversions_b, revenues_b) = self.data_loader.load_data()
        average_revenue_a = np.nansum((conversions_a * revenues_a)) / sum(~np.isnan(conversions_a.flatten()))
        average_revenue_b = np.nansum((conversions_b * revenues_b)) / sum(~np.isnan(conversions_b.flatten()))
        return average_revenue_a, average_revenue_b

    # TODO refactor this, it's horrible
    @staticmethod
    def get_sample_size(
        base_rate: float, base_std_dev: float, mde: float, alpha: float = 0.05, beta: float = 0.8
    ) -> int:
        """
        Returns the number of samples required (per variant) before stopping th AB test.
        WARNING: this is a gaussian assumption for a t-test type test, is not fully reliable for Mann-Whitney

        Args:
            base_rpu: the revenue per user "control" variant (the one currently in production)
            base_std_dev: the revenue's standard deviation for the "control" variant (the one currently in production)
            mde: The Minimum Detectable Effect is the smallest effect that will be detected (1-\\beta)\% of the time.
            alpha: a float representing the percentage of the time a difference will be detected, assuming one does NOT exist
            beta: a float representing the percentage of the time the minimum effect size will be detected, assuming it exists

        Returns:
            an integer, the number of samples per variation required before stopping th AB test.
        """
        factor = (scipy.stats.norm.ppf(1 - alpha / 2) + scipy.stats.norm.ppf(beta)) ** 2
        err = (base_rate * mde) ** 2
        variance = base_std_dev ** 2
        sample_size = int(factor * variance / err)
        logger.info(f"MDE: {mde} (relative), {mde * base_rate} (absolute)")
        logger.info(f"NHST samples required (per variant): {sample_size}")
        return sample_size
