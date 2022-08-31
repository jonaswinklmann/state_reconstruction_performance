import json
import os

from libics import env


###############################################################################
# Default configuration
###############################################################################


_CONFIG = {
    # File I/O
    "projector_cache_dir": os.path.join(
        env.DIR_LIBICS, "state_estimation", "projector_cache"
    ),
    "state_estimator_config_dir": os.path.join(
        env.DIR_LIBICS, "state_estimation", "state_estimator"
    ),
    # gen.trafo_gen
    "trafo_gen.phase_ref_image": (256, 256),
    "trafo_gen.phase_ref_site": (85, 85),
    "trafo_gen.trafo_site_to_image": {}
}

# Find configuration file path
_CONFIG_PATH = os.path.join(
    env.DIR_LIBICS, "state_estimation", "config.json"
)
if "state_estimation.config" in env.DIRS:
    _CONFIG_PATH = env.DIRS["state_estimation.config"]
config_dir = os.path.dirname(_CONFIG_PATH)
# Ensure configuration directory exists
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
# If file available, load configuration
if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH, "r") as _f:
        _CONFIG.update(json.load(_f))
# If unavailable, create configuration
else:
    with open(_CONFIG_PATH, "w") as _f:
        json.dump(_CONFIG, _f, indent=4)


###############################################################################
# Configuration I/O
###############################################################################


def get_config(key=None):
    """
    Gets a configuration value.

    Parameters
    ----------
    key : `str` or `None`
        Configuration key.
        If `None`, returns the full configuration dictionary.
    """
    if key is None:
        return _CONFIG
    else:
        return _CONFIG[key]


def set_config(**kwargs):
    """
    Sets a configuration value and saves it to the configuration file.

    Parameters
    ----------
    **kwargs : `Any`
        Keyword argument/value corresponds to configuration key/value.
    """
    _CONFIG.update(kwargs)
    with open(_CONFIG_PATH, "w") as _f:
        json.dump(_CONFIG, _f, indent=4)
