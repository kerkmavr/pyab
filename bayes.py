from __future__ import annotations

import abc
import datetime
import logging
import typing

import arviz as az
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

from pyab.core.base import ABTestABC

if typing.TYPE_CHECKING:
    from typing import Callable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
tfd = tfp.distributions
tfb = tfp.bijectors


class BayesABTestABC(ABTestABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _compute_posterior_numerical(self, *args, **kwargs):
        pass

    def _compute_posterior_analytical(self, *args, **kwargs):
        raise RuntimeError(
            "No analytical solution is available. Use the numerical one."
        )

    @abc.abstractmethod
    def _compute_expected_loss(self, *args, **kwargs):
        pass

    @staticmethod
    def plot_expected_loss(
        arr: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        experiments_idxs: Sequence[int],
        threshold: float = 5e-6,
        plot_interval: bool = True,
        label_1="A",
        label_2="B",
        colour_1="blue",
        colour_2="orange",
        saving_filepath=None,
    ) -> None:
        _ = plt.figure()
        linestyles = ["-", "--", "-.-"]
        for e in experiments_idxs:
            for i, (ela, elb, ila, ilb) in enumerate(arr):
                plt.plot(
                    range(len(ela[e])), ela[e], color=colour_1, linestyle=linestyles[i]
                )
                plt.plot(
                    range(len(ela[e])), elb[e], color=colour_2, linestyle=linestyles[i]
                )
                if plot_interval:
                    plt.plot(
                        range(len(ila[e, :, 0])),
                        ila[e, :, 0],
                        color=colour_1,
                        linestyle=":",
                    )
                    plt.plot(
                        range(len(ila[e, :, 1])),
                        ila[e, :, 1],
                        color=colour_1,
                        linestyle=":",
                    )
                    plt.plot(
                        range(len(ilb[e, :, 0])),
                        ilb[e, :, 0],
                        color=colour_2,
                        linestyle=":",
                    )
                    plt.plot(
                        range(len(ilb[e, :, 1])),
                        ilb[e, :, 1],
                        color=colour_2,
                        linestyle=":",
                    )

        plt.hlines([threshold], 0, len(ela[e]), linestyle="--")
        plt.xlabel("n days")
        plt.ylabel("exp loss")
        blue_patch = matplotlib.patches.Patch(color="blue", label=f"E[L({label_1})]")
        orange_patch = matplotlib.patches.Patch(
            color="orange", label=f"E[L({label_2})]"
        )
        plt.legend(handles=[blue_patch, orange_patch])

        if saving_filepath is not None:
            plt.savefig(saving_filepath)
        else:
            plt.show()

    def plot_posterior(self, samples: np.ndarray, xlabel: str = "", title: str = ""):
        sns.distplot(samples)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.show()

    def summarize_results(
        self,
        threashold: float,
        exp_loss_a: np.ndarray,
        exp_loss_b: np.ndarray,
        p_b_over_a: np.ndarray,
        initial_days: int = 0,
        probbaility_threashold_for_being_better: float = 0.0,
    ) -> Tuple[float, float, np.ndarray]:
        a = np.array(
            list(
                map(
                    lambda x: x.argmax() if True in x else np.inf,
                    (exp_loss_a[:, initial_days:] < threashold)
                    & (
                        (1.0 - p_b_over_a[:, initial_days:])
                        >= probbaility_threashold_for_being_better
                    ),
                )
            )
        )
        b = np.array(
            list(
                map(
                    lambda x: x.argmax() if True in x else np.inf,
                    (exp_loss_b[:, initial_days:] < threashold)
                    & (
                        p_b_over_a[:, initial_days:]
                        >= probbaility_threashold_for_being_better
                    ),
                )
            )
        )

        mask = ~((a == np.inf) & (b == np.inf))
        decision_taken_at_day = np.minimum(a[mask], b[mask]) + initial_days
        inconclusive = (np.sum(~mask) + (a[mask] == b[mask]).sum()) / len(a)
        fpr = (a[mask] < b[mask]).sum() / len(a)
        logger.info(f"Errors: {fpr}, Inconclusive experiments: {inconclusive}")
        return fpr, inconclusive, decision_taken_at_day

    def get_all_values_a_and_value_b(self):
        (conversions_a, counts_a), (
            conversions_b,
            counts_b,
        ) = self.data_loader.load_data()
        cr_a = conversions_a / counts_a
        cr_b = conversions_b / counts_b
        return cr_a, cr_b

    def produce_table_results(
        self, product, days_for_significance=3, **kwargs
    ) -> pd.DataFrame:
        date_range = self.data_loader.get_date_range()
        ordered_dates = self.data_loader.get_ordered_dates()
        cr_a, cr_b = self.get_all_values_a_and_value_b()

        (
            expected_loss_a,
            expected_loss_b,
            interval_loss_a,
            interval_loss_b,
            p_b_over_a,
        ) = self.run_test(exact=True)

        dfs = []

        for test_symbol, el, il, cr in [
            ("P", expected_loss_a, interval_loss_a, cr_a),
            ("T", expected_loss_b, interval_loss_b, cr_b),
        ]:
            result_dict = {
                "DATE": ordered_dates,
                "VALUE": cr[0],
                "EXPECTED_LOSS": el[0],
                "LOWER_BOUNDARY": il[0, :, 0],
                "UPPER_BOUNDARY": il[0, :, 1],
            }

            df = pd.DataFrame(result_dict)
            df["PRODUCT"] = product
            df["EXPERIMENT_NAME"] = self.name
            df["TEST_TYPE"] = self.get_test_type()
            df["APPROACH"] = "Bayesian"
            df["MIN_DURATION"] = None
            df["PVALUE"] = None
            df["KPI"] = self.data_loader.get_kpi()
            df["PROFILE"] = test_symbol
            df["MAIN_KPI"] = None
            df["START_DATE"] = date_range[0]
            df["END_DATE"] = date_range[1]

            # df["LOWER_BOUNDARY"] = df["EXPECTED_LOSS"] - 0.1 * df["EXPECTED_LOSS"]
            # df["UPPER_BOUNDARY"] = df["EXPECTED_LOSS"] + 0.1 * df["EXPECTED_LOSS"]
            df["EXPECTED_LOSS_THRESHOLD"] = kwargs["threshold"]
            if len(df) > days_for_significance:
                df["SIGNIFICANCE_FLAG"] = df["EXPECTED_LOSS"] < kwargs["threshold"]
            else:
                df["SIGNIFICANCE_FLAG"] = False

            assert set(self.table_results_columns) == set(df.keys())
            dfs.append(df)
        return pd.concat(dfs)


class BayesianInferenceMixin:
    @tf.function(autograph=False, experimental_compile=True)
    def _run_mcmc(
        self,
        init_state,
        target_log_prob_fn: Callable,
        mcmc_samples: int,
        n_chains: int = 1,
    ):
        step_size = np.random.rand(n_chains, 1) * 0.5 + 1.0

        num_burnin_steps = int(mcmc_samples * 0.3)

        bijectors_list = [tfb.Identity()]

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn, step_size=[step_size]
                ),
                bijector=bijectors_list,
            ),
            target_accept_prob=0.8,
            num_adaptation_steps=int(0.8 * num_burnin_steps),
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(step_size=new_step_size)
            ),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        def trace_fn(_, pkr):
            return (
                pkr.inner_results.inner_results.target_log_prob,
                pkr.inner_results.inner_results.leapfrogs_taken,
                pkr.inner_results.inner_results.has_divergence,
                pkr.inner_results.inner_results.energy,
                pkr.inner_results.inner_results.log_accept_ratio,
            )

        res = tfp.mcmc.sample_chain(
            num_results=mcmc_samples,
            num_burnin_steps=int(mcmc_samples * 0.3),
            current_state=[init_state],
            kernel=kernel,
            trace_fn=trace_fn,
        )
        return res


class ConversionRateBayesABTest(BayesianInferenceMixin, BayesABTestABC):
    def __init__(
        self,
        *,
        prior_a: Optional[tfd.Distribution] = None,
        prior_b: Optional[tfd.Distribution] = None,
        **kwargs,
    ):
        super(ConversionRateBayesABTest, self).__init__(**kwargs)
        self.prior_a = prior_a or tfd.Beta(1, 1)
        self.prior_b = prior_b or tfd.Beta(1, 1)

    def get_test_type(self) -> str:
        return "BayesConversionRate"

    def _compute_posterior_numerical(
        self, conversions, counts, prior: tfd.Distribution, mcmc_samples: int = 10_000
    ):
        logger.info("Computing the posterior for the conversion rates (MCMC)...")
        counts = np.cumsum(counts, axis=1)
        conversions = np.cumsum(conversions, axis=1)

        model = tfd.JointDistributionSequential(
            [
                # P(p) = prior
                prior,
                # likelihood: P(X|p) = Binom(n,p)
                lambda p: tfd.Sample(
                    tfd.Binomial(
                        tf.convert_to_tensor(counts.ravel(), dtype=tf.float32), probs=p
                    ),
                ),
            ]
        )
        n_chains = 3
        init_state = np.tile(
            model.sample()[0].numpy(), (n_chains, np.prod(counts.shape))
        )
        target_log_prob_fn = lambda p: model.log_prob(p, conversions.ravel())
        posterior_samples = self._run_mcmc(
            init_state, target_log_prob_fn, mcmc_samples=mcmc_samples, n_chains=n_chains
        )
        return model, posterior_samples[0][0].numpy().mean(axis=1).reshape(
            -1, *counts.shape
        )

    def _compute_posterior_analytical(
        self, conversions, counts, prior, mcmc_samples: int = 10_000
    ):
        logger.info("Computing the posterior for the conversion rates (analytical)...")
        counts = np.cumsum(counts, axis=1)
        conversions = np.cumsum(conversions, axis=1)
        successes = prior.concentration1 + conversions
        failures = prior.concentration0 + counts - prior.concentration1 - conversions
        posterior = tfd.Independent(tfd.Beta(successes, failures))
        posterior_samples = posterior.sample(mcmc_samples).numpy()
        return posterior, posterior_samples

    def _compute_expected_loss(
        self, posterior_a_samples, posterior_b_samples
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        posterior_diff_samples = posterior_b_samples - posterior_a_samples
        mcmc_samples, n_experiments, n_days = posterior_diff_samples.shape
        (
            expected_loss_a,
            interval_loss_a,
            expected_loss_b,
            interval_loss_b,
            prob_b_vs_a,
        ) = (
            np.zeros((n_experiments, n_days)),
            np.zeros((n_experiments, n_days, 2)),
            np.zeros((n_experiments, n_days)),
            np.zeros((n_experiments, n_days, 2)),
            np.zeros((n_experiments, n_days)),
        )
        logger.info("Computing the expected losses...")
        for e in tqdm.tqdm(range(n_experiments)):
            for d in range(n_days):
                d_pdf = posterior_diff_samples[:, e, d]
                pdf_at_bins, bins = np.histogram(d_pdf, bins=100, density=True)
                bins = 0.5 * (bins[:-1] + bins[1:])
                loss_a = np.maximum(bins, 0) * pdf_at_bins
                loss_b = np.maximum(-bins, 0) * pdf_at_bins

                expected_loss_a[e, d] = np.trapz(loss_a, bins)

                # (1) This way of doing it we ignore all the cases where the loss is negative
                # # I've put this here for completeness but I don't think we want it due to it's biasing effect
                # interval_loss_a[e, d] = az.hdi(d_pdf[d_pdf>0], 0.95)

                # (2) This way of doing it we include the cases where the loss is negative in the pdf:
                interval_loss_a[e, d] = az.hdi(d_pdf * np.heaviside(d_pdf, 0), 0.95)

                expected_loss_b[e, d] = np.trapz(loss_b, bins)

                # (1) This way of doing it we ignore all the cases where the loss is negative
                # # I've put this here for completeness but I don't think we want it due to it's biasing effect
                # interval_loss_b[e, d] = az.hdi(-d_pdf[d_pdf<0], 0.95)

                # (2) This way of doing it we include the cases where the loss is negative in the pdf:
                interval_loss_b[e, d] = az.hdi(
                    -d_pdf * np.heaviside(-d_pdf, 0), 0.95
                )  # this is

                prob_b_vs_a[e, d] = (
                    posterior_b_samples[:, e, d] > posterior_a_samples[:, e, d]
                ).astype(int).sum() / mcmc_samples
        return (
            expected_loss_a,
            expected_loss_b,
            interval_loss_a,
            interval_loss_b,
            prob_b_vs_a,
        )

    def run_test(
        self, exact: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        posterior_samples_fn = self._compute_posterior_numerical
        if exact is True:
            posterior_samples_fn = self._compute_posterior_analytical

        (conversions_a, counts_a), (
            conversions_b,
            counts_b,
        ) = self.data_loader.load_data()
        posterior, posterior_a_samples = posterior_samples_fn(
            conversions_a, counts_a, self.prior_a
        )
        posterior, posterior_b_samples = posterior_samples_fn(
            conversions_b, counts_b, self.prior_b
        )
        return self._compute_expected_loss(posterior_a_samples, posterior_b_samples)

    def _compare_numerical_and_analyitical_solution(self, plot: bool = True):
        (conversions_a, counts_a), (
            conversions_b,
            counts_b,
        ) = self.data_loader.load_data()
        _, posterior_a_samples = self._compute_posterior_numerical(
            conversions_a, counts_a, self.prior_a
        )
        _, posterior_b_samples = self._compute_posterior_numerical(
            conversions_b, counts_b, self.prior_b
        )
        losses_numerical = self._compute_expected_loss(
            posterior_a_samples, posterior_b_samples
        )
        _, pa = self._compute_posterior_analytical(
            conversions_a, counts_a, self.prior_a, 10_000
        )
        _, pb = self._compute_posterior_analytical(
            conversions_b, counts_b, self.prior_b, 10_000
        )
        losses_analytical = self._compute_expected_loss(pa, pb)
        if plot:
            self.plot_expected_loss(
                [losses_numerical[:2], losses_analytical[:2]], experiments_idxs=[0]
            )
        return (losses_numerical[0], losses_numerical[1], losses_numerical[-1]), (
            losses_analytical[0],
            losses_analytical[1],
            losses_analytical[-1],
        )


class RevenuePerUserBayesABTest(BayesianInferenceMixin, BayesABTestABC):
    def __init__(
        self,
        *,
        prior_pa: Optional[tfd.Distribution] = None,
        prior_pb: Optional[tfd.Distribution] = None,
        prior_revenues_on_sale: Optional[tfd.Distribution] = None,
        **kwargs,
    ):
        super(RevenuePerUserBayesABTest, self).__init__(**kwargs)
        self.prior_pa = prior_pa or tfd.Beta(1, 1)
        self.prior_pb = prior_pb or tfd.Beta(1, 1)
        self.prior_revenues_on_sale = prior_revenues_on_sale or tfd.Gamma(1, 0.01)

    def _compute_posterior_numerical(self, conversions, revenues, priors, mcmc_samples):
        raise NotImplementedError

    def get_test_type(self) -> str:
        return "BayesRevenuePerUser"

    def _compute_posterior_analytical(
        self, conversions, revenues, priors, mcmc_samples
    ):
        logger.info("Computing the posterior for the conversion rates (analytical) ...")
        counts = (~np.isnan(conversions)).astype(int).sum(axis=-1).cumsum(axis=-1)
        c = np.nansum(conversions, axis=-1).cumsum(axis=-1)
        successes = priors["p"].concentration1 + c
        failures = priors["p"].concentration0 + counts - priors["p"].concentration1 - c
        posterior_conversion = tfd.Beta(successes, failures)
        posterior_samples_conversion = posterior_conversion.sample(mcmc_samples).numpy()

        logger.info("Computing the posterior for the revenues per sale (analytical)...")
        rpu = np.nansum(conversions * revenues, axis=-1).cumsum(axis=-1) / c
        if len(rpu[rpu < 0.0]):
            logger.warning(
                f"There are some full days with negative revenues: {rpu[rpu < 0.]}. Will set them to 0.0"
            )
            rpu[rpu < 0.0] = 0.0

        concentration = priors["theta"].concentration + c
        rate = priors["theta"].rate / (1 + priors["theta"].rate * c * rpu)
        posterior_rev = tfd.Gamma(concentration, 1 / rate)
        posterior_samples_rev = posterior_rev.sample(mcmc_samples).numpy()
        return (
            posterior_conversion,
            posterior_rev,
        ), posterior_samples_conversion / posterior_samples_rev

    def _compute_expected_loss(
        self, posterior_a_samples: np.ndarray, posterior_b_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Computing the expected losses...")
        b_vs_a = np.maximum(posterior_b_samples > posterior_a_samples, 0).astype(int)

        loss_b = np.maximum(
            np.abs(posterior_a_samples) - np.abs(posterior_b_samples), 0
        )
        loss_a = np.maximum(
            np.abs(posterior_b_samples) - np.abs(posterior_a_samples), 0
        )

        b_vs_a_loss_a = b_vs_a * loss_a
        expected_loss_a = np.mean(b_vs_a_loss_a, axis=0)
        # interval_loss_a = az.hdi(b_vs_a_loss_a, 0.95).reshape((1,-1,2))
        interval_loss_a = np.moveaxis(
            np.apply_along_axis(lambda x: az.hdi(x, 0.95), axis=0, arr=b_vs_a_loss_a),
            0,
            -1,
        )

        b_vs_a_loss_b = b_vs_a * loss_b
        expected_loss_b = np.mean((1.0 - b_vs_a) * loss_b, axis=0)
        # interval_loss_b = az.hdi(b_vs_a * loss_b, 0.95).reshape((1,-1,2))
        interval_loss_b = np.moveaxis(
            np.apply_along_axis(lambda x: az.hdi(x, 0.95), axis=0, arr=b_vs_a_loss_b),
            0,
            -1,
        )
        prob_b_vs_a = np.mean(b_vs_a, axis=0)
        return (
            expected_loss_a,
            expected_loss_b,
            interval_loss_a,
            interval_loss_b,
            prob_b_vs_a,
        )

    def run_test(
        self, exact: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        posterior_samples_fn = self._compute_posterior_numerical
        if exact is True:
            posterior_samples_fn = self._compute_posterior_analytical

        (conversions_a, revenues_a), (
            conversions_b,
            revenues_b,
        ) = self.data_loader.load_data()
        _, posterior_a_samples = posterior_samples_fn(
            conversions_a,
            revenues_a,
            {"p": self.prior_pa, "theta": self.prior_revenues_on_sale},
            mcmc_samples=10_000,
        )
        _, posterior_b_samples = posterior_samples_fn(
            conversions_b,
            revenues_b,
            {"p": self.prior_pb, "theta": self.prior_revenues_on_sale},
            mcmc_samples=10_000,
        )
        return self._compute_expected_loss(posterior_a_samples, posterior_b_samples)
