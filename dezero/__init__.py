is_simple_core = False

if is_simple_core:
    from dezero.core_simple import (
        Config,
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        setup_variable,
        using_config,
    )
else:
    from dezero.core import (
        Config,
        Function,
        Parameter,
        Variable,
        as_array,
        as_variable,
        no_grad,
        setup_variable,
        test_mode,
        using_config,
    )

import dezero.functions  # NOQA
from dezero.layers import Layer  # NOQA
from dezero.models import Model  # NOQA
import dezero.datasets  # NOQA
from dezero.dataloaders import DataLoader  # NOQA
import dezero.optimizers  # NOQA


setup_variable()


__all__ = [
    "Function",
    "Parameter",
    "Variable",
    "as_array",
    "as_variable",
    "no_grad",
    "using_config",
    "Config",
    "test_mode",
]
