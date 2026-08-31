from unittest.mock import patch

import pytest

from run_stage3_model import _validate_stage3_config


def test_validate_stage3_config_passes_when_purge_covers_horizon():
    with patch("run_stage3_model.WALK_FORWARD_PURGE_DAYS", 20), \
         patch("run_stage3_model.PREDICTION_HORIZON_DAYS", 20):
        _validate_stage3_config()  # should not raise


def test_validate_stage3_config_raises_when_purge_smaller_than_horizon():
    with patch("run_stage3_model.WALK_FORWARD_PURGE_DAYS", 10), \
         patch("run_stage3_model.PREDICTION_HORIZON_DAYS", 20):
        with pytest.raises(ValueError, match="WALK_FORWARD_PURGE_DAYS"):
            _validate_stage3_config()
