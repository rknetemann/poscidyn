import unittest

import jax.numpy as jnp

from poscidyn.excitation.direct import DirectHarmonicExcitation
from poscidyn.excitation.free_vibration import FreeVibration
from poscidyn.oscillator.nonlinear import NonlinearOscillator
from poscidyn.solver.time_integration import TimeIntegration


def make_oscillator():
    return NonlinearOscillator(
        omega_0=jnp.array([1.0]),
        Q=jnp.array([10.0]),
        a=jnp.zeros((1, 1, 1)),
        b=jnp.zeros((1, 1, 1, 1)),
    )


class TimeResponseValidationTests(unittest.TestCase):
    def test_periodic_excitation_requires_omega_even_with_duration(self):
        solver = TimeIntegration(
            make_oscillator(),
            DirectHarmonicExcitation(f_d=jnp.array([0.1])),
        )

        with self.assertRaisesRegex(ValueError, "requires an omega"):
            solver.time_response(jnp.array([0.0]), jnp.array([0.0]), t=1.0)

    def test_time_response_rejects_multiple_excitation_frequencies(self):
        solver = TimeIntegration(
            make_oscillator(),
            DirectHarmonicExcitation(
                f_d=jnp.array([0.1]), omega=jnp.array([1.0, 2.0])
            ),
        )

        with self.assertRaisesRegex(ValueError, "exactly one excitation frequency"):
            solver.time_response(jnp.array([0.0]), jnp.array([0.0]), t=1.0)

    def test_non_periodic_excitation_requires_duration(self):
        solver = TimeIntegration(make_oscillator(), FreeVibration())

        with self.assertRaisesRegex(ValueError, "requires t"):
            solver.time_response(jnp.array([0.0]), jnp.array([0.0]))

    def test_explicit_duration_requires_time_steps(self):
        solver = TimeIntegration(make_oscillator(), n_time_steps=None)

        with self.assertRaisesRegex(ValueError, "n_time_steps"):
            solver.time_response(jnp.array([0.0]), jnp.array([0.0]), t=1.0)

    def test_explicit_duration_must_be_positive(self):
        solver = TimeIntegration(make_oscillator())

        with self.assertRaisesRegex(ValueError, "positive scalar duration"):
            solver.time_response(jnp.array([0.0]), jnp.array([0.0]), t=0.0)

    def test_steady_state_estimate_receives_angular_frequency(self):
        class RecordingOscillator(NonlinearOscillator):
            def t_steady_state(self, driving_frequency, ss_tol):
                self.driving_frequency = driving_frequency
                return jnp.array(0.0)

        oscillator = RecordingOscillator(
            omega_0=jnp.array([1.0]),
            Q=jnp.array([10.0]),
            a=jnp.zeros((1, 1, 1)),
            b=jnp.zeros((1, 1, 1, 1)),
        )
        solver = TimeIntegration(
            oscillator,
            DirectHarmonicExcitation(f_d=jnp.array([0.1]), omega=jnp.array([2.0])),
            n_time_steps=2,
            periods_to_retain=1,
        )

        solver.time_response(jnp.array([0.0]), jnp.array([0.0]))

        self.assertTrue(jnp.allclose(oscillator.driving_frequency, jnp.array([2.0])))


if __name__ == "__main__":
    unittest.main()
