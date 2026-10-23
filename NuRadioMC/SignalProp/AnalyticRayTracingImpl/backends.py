"""Separate Python and Numba helper namespaces for analytic ray tracers."""

from functools import lru_cache
from importlib import import_module
from types import FunctionType, SimpleNamespace


class _Backend(SimpleNamespace):
    """Serialize the selection rather than functions or dispatchers."""

    def __reduce__(self):
        return _restore_backend, (self._module_names, self._compile_numba)


def _restore_backend(module_names, compile_numba):
    modules = tuple(import_module(name) for name in module_names)
    return get_backend(modules, compile_numba)


def numba_candidate(**options):
    """Mark a Python helper for optional compilation without replacing it."""
    def decorate(function):
        function._numba_options = options
        return function
    return decorate


@lru_cache(maxsize=None)
def get_backend(modules, compile_numba):
    """Return helpers whose entire call graph uses the selected backend.

    Parameters
    ----------
    modules : tuple of module
        Helper modules, including Python solvers that call their functions.
    compile_numba : bool
        Compile marked helpers, keeping the source modules unchanged.

    Notes
    -----
    Numba resolves helper calls through a function's globals. Merely retaining
    ``dispatcher.py_func`` does not isolate nested calls. For the compiled
    backend, copies of the functions share private dictionaries, with all
    references to helpers (including imports between modules) bound to the
    compiled backend. Python solvers are copied too, but remain Python.
    Function bodies, defaults and source locations are preserved; physics code
    is neither duplicated nor generated. The Python backend uses the originals.
    """
    functions = {
        value: module
        for module in modules
        for value in vars(module).values()
        if (isinstance(value, FunctionType)
            and value.__module__ == module.__name__)
    }
    if not compile_numba:
        return _Backend(_module_names=tuple(m.__name__ for m in modules),
                        _compile_numba=False,
                        **{fn.__name__: fn for fn in functions})

    from numba import njit

    namespaces = {module: dict(vars(module)) for module in modules}
    replacements = {}
    for function, module in functions.items():
        copy = FunctionType(
            function.__code__, namespaces[module], function.__name__,
            function.__defaults__, function.__closure__)
        copy.__kwdefaults__ = function.__kwdefaults__
        copy.__annotations__ = function.__annotations__
        copy.__dict__.update(function.__dict__)
        copy.__module__ = function.__module__
        copy.__qualname__ = function.__qualname__
        options = getattr(function, '_numba_options', None)
        replacements[function] = (
            njit(**options)(copy) if options is not None else copy)

    # Bind after creating every dispatcher, before any lazy compilation occurs.
    for namespace in namespaces.values():
        for name, value in namespace.items():
            if isinstance(value, FunctionType) and value in replacements:
                namespace[name] = replacements[value]

    return _Backend(_module_names=tuple(m.__name__ for m in modules),
                    _compile_numba=True,
                    **{fn.__name__: value for fn, value in replacements.items()})
