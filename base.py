from __future__ import annotations

import abc
import logging
import typing

import pandas as pd

if typing.TYPE_CHECKING:
    # from pyab.data.synthetic import DataGeneratorABC - this is a data generator to test the code
    from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ABTestABC(metaclass=abc.ABCMeta):

    table_results_columns = [
        "EXPERIMENT_NAME",
        "PRODUCT",
        "APPROACH",
        "START_DATE",
        "END_DATE",
        "DATE",
        "MIN_DURATION",
        "TEST_TYPE",
        "MAIN_KPI",
        "KPI",
        "PROFILE",
        "VALUE",
        "PVALUE",
        "EXPECTED_LOSS",
        "LOWER_BOUNDARY",
        "UPPER_BOUNDARY",
        "EXPECTED_LOSS_THRESHOLD",
        "SIGNIFICANCE_FLAG",
    ]

    def __init__(self, data_loader: Optional = None, name: Optional[str] = None):
        self.data_loader = data_loader
        self.name = name

    @abc.abstractmethod
    def run_test(self):
        pass

    @abc.abstractmethod
    def produce_table_results(self, **kwargs) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_test_type(self) -> str:
        pass
