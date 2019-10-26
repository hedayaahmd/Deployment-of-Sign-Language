import pytest
import os

from CNNModel.config import config


@pytest.fixture
def buy_dir():
    test_data_dir = os.path.join(config.DATASET_DIR, 'test_data')
    buy_dir = os.path.join(test_data_dir, 'buy')

    return buy_dir


@pytest.fixture
def angry_dir():
    test_data_dir = os.path.join(config.DATASET_DIR, 'test_data')
    angry_dir = os.path.join(test_data_dir, 'angry')

    return angry_dir


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, str) and isinstance(right, str) and op == "==":
        return [
            "Comparing Classes instances:",
            "   vals: {} != {}".format(left, right),
        ]
