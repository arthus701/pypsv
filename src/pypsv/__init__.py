import warnings

from pypsv import fieldmodels
from pypsv import parameters
from pypsv import utils
from pypsv import calibration_curves
from pypsv.psv_curve import PSVCurve

# Monkey-patch the line away from warnings, as it is rather irritating.
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda msg, cat, fname, lineno, line=None: \
    formatwarning_orig(msg, cat, fname, lineno, line='')


__all__ = [
    'fieldmodels', 'PSVCurve', 'parameters', 'utils', 'calibration_curves',
]

__version__ = "0.0.1"
