"""Compatibility tests for the oscillator--excitation--time-solver boundary."""

import unittest

import jax.numpy as jnp
import numpy as np
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from poscidyn.excitation.direct import DirectHarmonicExcitation
from poscidyn.excitation.free_vibration import FreeVibration
from poscidyn.excitation.parametric import ParametricHarmonicExcitation
from poscidyn.oscillator.nonlinear import NonlinearOscillator
from poscidyn.multistart.linear_response import LinearResponse
from poscidyn.response_measure.rms import RMS
from poscidyn.solver.time_integration.time_integration import TimeIntegration


def make_oscillator(n_modes=2):
    return NonlinearOscillator(
        omega_0=jnp.ones(n_modes),
        Q=jnp.full(n_modes, 10.0),
        a=jnp.zeros((n_modes, n_modes, n_modes)),
        b=jnp.zeros((n_modes, n_modes, n_modes, n_modes)),
    )


class ExcitationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.y = jnp.array([2.0, -3.0, 0.0, 0.0])
        self.args = {"omega": jnp.array([2.0, 2.0]), "lambda": jnp.array([9.0, 9.0])}

    def test_free_vibration_returns_zero_force_for_every_mode(self):
        force = FreeVibration().f_e(0.3, self.y, {})
        np.testing.assert_array_equal(force, jnp.zeros(2))

    def test_direct_harmonic_force_has_one_value_per_mode(self):
        excitation = DirectHarmonicExcitation(
            f_d=jnp.array([2.0, 3.0]), lambdas=jnp.array([0.5, 2.0])
        )
        force = excitation.f_e(0.0, self.y, self.args)
        np.testing.assert_allclose(force, jnp.array([1.0, 6.0]))

    def test_parametric_force_modulates_each_modal_displacement(self):
        excitation = ParametricHarmonicExcitation(
            f_p=jnp.array([2.0, 3.0]), lambdas=jnp.array([0.5, 2.0])
        )
        force = excitation.f_e(0.0, self.y, self.args)
        np.testing.assert_allclose(force, jnp.array([2.0, -18.0]))

    def test_each_builtin_excitation_is_compatible_with_time_response(self):
        excitations = (
            (FreeVibration(), {"t": 0.1}),
            (
                DirectHarmonicExcitation(
                    f_d=jnp.array([0.1, 0.2]), omega=1.0
                ),
                {"t": 0.1},
            ),
            (
                ParametricHarmonicExcitation(
                    f_p=jnp.array([0.1, 0.2]), omega=1.0
                ),
                {"t": 0.1},
            ),
        )
        for excitation, kwargs in excitations:
            with self.subTest(excitation=type(excitation).__name__):
                response = TimeIntegration(
                    make_oscillator(), excitation, n_time_steps=4
                ).time_response(jnp.zeros(2), jnp.zeros(2), **kwargs)
                self.assertEqual(response.displacement.shape, (4, 2))
                self.assertTrue(np.isfinite(np.asarray(response.displacement)).all())

    def test_periodic_excitations_are_compatible_with_frequency_sweep(self):
        excitations = (
            DirectHarmonicExcitation(f_d=jnp.array([0.1])),
            ParametricHarmonicExcitation(f_p=jnp.array([0.1])),
        )
        for excitation in excitations:
            with self.subTest(excitation=type(excitation).__name__):
                solver = TimeIntegration(
                    make_oscillator(n_modes=1),
                    excitation,
                    response_measure=RMS(),
                    multistart=LinearResponse(n_init_cond=1),
                    n_time_steps=8,
                    periods_to_retain=1,
                    max_steps=4096,
                )
                sweep = solver.frequency_sweep(jnp.array([0.9, 1.1]))
                self.assertEqual(sweep.frequency.shape, (2,))
                self.assertTrue(
                    np.isfinite(np.asarray(sweep.forward.total.amplitude)).all()
                )


if __name__ == "__main__":
    unittest.main()
