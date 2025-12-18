is_simple_core = False

if is_simple_core:
    from dezero.core_simple import (
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
        Function,
        Parameter,
        Variable,
        as_array,
        as_variable,
        no_grad,
        setup_variable,
        using_config,
    )

import dezero.functions  # NOQA

setup_variable()


__all__ = [
    "Function",
    "Parameter",
    "Variable",
    "as_array",
    "as_variable",
    "no_grad",
    "using_config",
]
