"""Regression tests for independent Python/Numba analytic tracer instances.

Run with ``python -m unittest discover -s NuRadioMC/test/SignalProp
-p test_backend_isolation.py``. No C++ build or external data is needed.
"""

import os
import pickle
import subprocess
import sys
import unittest
from types import FunctionType
from unittest.mock import patch

import numpy as np

# These tests exercise the Python implementations, not the import-time C++ build.
with patch.object(subprocess, 'call', return_value=1):
    from NuRadioMC.SignalProp import analyticraytracing as art
from NuRadioMC.SignalProp.AnalyticRayTracingImpl import single_layer_analytic_raytracer as single
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing import corefunctions
from NuRadioMC.utilities import medium

try:
    from numba.core.registry import CPUDispatcher
    import numba
    NUMBA_ENABLED = not numba.config.DISABLE_JIT
except ImportError:
    NUMBA_ENABLED = False


def make_tracer(model, enabled):
    return art.ray_tracing(model(), use_cpp=False, compile_numba=enabled)


def solve(tracer, radius=100.):
    tracer.set_start_and_end_point(np.array([0., 0., -500.]),
                                   np.array([radius, 0., -100.]))
    tracer.find_solutions()
    return np.array([
        [tracer.get_results()[i]['C0'], tracer.get_travel_time(i), tracer.get_path_length(i),
         *tracer.get_launch_vector(i), *tracer.get_receive_vector(i)]
        for i in range(tracer.get_number_of_solutions())
    ])


class BackendIsolationTest(unittest.TestCase):
    def test_default_backend_selection(self):
        for model in (medium.southpole_2015, medium.southpole_2015_layered):
            tracer = make_tracer(model, None)
            self.assertEqual(tracer.compile_numba, single.numba_available)
            self.assertEqual(len(solve(tracer)), 2)

    @unittest.skipUnless(NUMBA_ENABLED, 'Numba JIT unavailable or disabled')
    def test_mixed_instances(self):
        for model, module, helper in (
            (medium.southpole_2015, single, '_get_delta_y'),
            (medium.southpole_2015_layered, corefunctions, 'get_delta_y'),
        ):
            originals = {name: fn for name, fn in vars(module).items()
                         if isinstance(fn, FunctionType) and fn.__module__ == module.__name__}
            for order in ((False, True, False), (True, False, True)):
                with self.subTest(model=model.__name__, order=order):
                    tracers = [make_tracer(model, enabled) for enabled in order]
                    results = []
                    for tracer, enabled in zip(tracers, order):
                        python_calls = []

                        def profile(frame, event, arg):
                            if event == 'call' and frame.f_code is originals[helper].__code__:
                                python_calls.append(frame.f_code)

                        previous_profile = sys.getprofile()
                        try:
                            sys.setprofile(profile)
                            results.append(solve(tracer))
                        finally:
                            sys.setprofile(previous_profile)
                        self.assertEqual(tracer.compile_numba, enabled)
                        self.assertEqual(bool(python_calls), not enabled)
                        selected = getattr(tracer._r2d._backend, helper)
                        self.assertEqual(isinstance(selected, CPUDispatcher), enabled)
                        self.assertEqual(len(results[-1]), 2)
                    for result in results[1:]:
                        np.testing.assert_allclose(result, results[0], rtol=1e-10, atol=1e-9)

                    # Reuse an older Python instance after another JIT solve.
                    python = tracers[order.index(False)]
                    compiled = tracers[order.index(True)]
                    solve(compiled, radius=500.)
                    signatures = {name: tuple(value.signatures)
                                  for name, value in vars(compiled._r2d._backend).items()
                                  if isinstance(value, CPUDispatcher)}
                    solve(python, radius=500.)
                    self.assertEqual(signatures, {
                        name: tuple(value.signatures)
                        for name, value in vars(compiled._r2d._backend).items()
                        if isinstance(value, CPUDispatcher)
                    })
                    for name, fn in originals.items():
                        self.assertIs(getattr(module, name), fn)

    def test_python_backend_and_serialization(self):
        for model in (medium.southpole_2015, medium.southpole_2015_layered):
            for enabled in (False, True):
                with self.subTest(model=model.__name__, enabled=enabled):
                    tracer = make_tracer(model, enabled)
                    expected = solve(tracer)
                    restored = pickle.loads(pickle.dumps(tracer))
                    np.testing.assert_allclose(solve(restored), expected, rtol=1e-12)
                    self.assertEqual(restored.compile_numba, tracer.compile_numba)

    @unittest.skipUnless(NUMBA_ENABLED, 'Numba JIT unavailable or disabled')
    def test_preparation_failure_falls_back_per_instance(self):
        for model, module in (
            (medium.southpole_2015, single),
            (medium.southpole_2015_layered,
             sys.modules[art.multi_layer_ray_tracing_2D.__module__]),
        ):
            compiled = make_tracer(model, True)
            original_get_backend = module.get_backend

            def fail_compilation(modules, enabled):
                if enabled:
                    raise RuntimeError('simulated dispatcher preparation failure')
                return original_get_backend(modules, enabled)

            with patch.object(module, 'get_backend', side_effect=fail_compilation):
                fallback = make_tracer(model, True)
            self.assertFalse(fallback.compile_numba)
            self.assertTrue(compiled.compile_numba)
            np.testing.assert_allclose(solve(fallback), solve(compiled), rtol=1e-10)

    def test_no_numba(self):
        # Use a fresh process so imported Numba modules cannot hide the fallback.
        script = '''
import importlib.abc
import sys
import unittest
class NoNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numba' or fullname.startswith('numba.'):
            raise ImportError('Numba deliberately unavailable')
sys.meta_path.insert(0, NoNumba())
from test_backend_isolation import make_tracer, solve, medium
for model in (medium.southpole_2015, medium.southpole_2015_layered):
    for enabled in (False, True, None):
        tracer = make_tracer(model, enabled)
        assert tracer.compile_numba is False
        assert len(solve(tracer)) == 2
assert not any(name == 'numba' or name.startswith('numba.') for name in sys.modules)
'''
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([os.path.dirname(__file__), *sys.path])
        result = subprocess.run([sys.executable, '-c', script], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
