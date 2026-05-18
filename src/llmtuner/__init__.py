# Level: eval, train > data, model > extras, hparams

# Patch vllm for compatibility with trl 0.28.0
# The cluster image has vllm 'dev' version which:
# 1. Has invalid version string 'dev' (not PEP 440 compliant)
# 2. Missing newer APIs like StructuredOutputsParams
try:
    import vllm
    # Fix version string
    if hasattr(vllm, '__version__') and vllm.__version__ == 'dev':
        vllm.__version__ = '0.11.0'
    # Mock missing classes that trl 0.28.0 expects
    import vllm.sampling_params as sp
    if not hasattr(sp, 'StructuredOutputsParams'):
        class StructuredOutputsParams:
            """Mock class for vllm compatibility with trl 0.28.0"""
            pass
        sp.StructuredOutputsParams = StructuredOutputsParams
except ImportError:
    pass

from .eval import Evaluator
from .train import export_model, run_exp


__version__ = "0.5.3"
__all__ = ["Evaluator", "export_model", "run_exp"]
