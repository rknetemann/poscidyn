import jax.numpy as jnp
import numpy as np
import sys
import unittest

import poscidyn

_PLOT_ON_MAIN = "--plot" in sys.argv
if _PLOT_ON_MAIN:
    sys.argv.remove("--plot")

try:
    import matplotlib

    if not _PLOT_ON_MAIN:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class DummySolver:
    def __init__(self):
        self.model = None
        self.received = None

    def time_response(self, driving_frequency, driving_amplitude, initial_displacement, initial_velocity, **kwargs):
        self.received = {
            "driving_frequency": driving_frequency,
            "driving_amplitude": driving_amplitude,
            "initial_displacement": initial_displacement,
            "initial_velocity": initial_velocity,
            "kwargs": kwargs,
        }
        n_modes = self.model.n_modes
        ts = jnp.linspace(0.0, 1.0, 4)
        xs = jnp.zeros((4, n_modes))
        vs = jnp.zeros((4, n_modes))
        ys = jnp.concatenate([xs, vs], axis=1)
        return ts, ys


def make_model(n_modes=1):
    return poscidyn.Nonlinear(
        omega_0=np.ones((n_modes,)),
        Q=np.full((n_modes,), 100.0),
        a=np.zeros((n_modes, n_modes, n_modes)),
        b=np.zeros((n_modes, n_modes, n_modes, n_modes)),
    )


def make_excitation(drive_frequency=1.0, drive_amplitude=0.1, modal_forces=None):
    if modal_forces is None:
        modal_forces = np.array([1.0])
    return poscidyn.DirectExcitation(
        drive_frequencies=np.array([drive_frequency]),
        drive_amplitudes=np.array([drive_amplitude]),
        modal_forces=np.asarray(modal_forces),
    )


def plot_time_response(ts, xs, vs, title="Time response validation"):
    if plt is None:
        raise RuntimeError("matplotlib is not available.")

    ts = np.asarray(ts)
    xs = np.asarray(xs)
    vs = np.asarray(vs)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for mode_idx in range(xs.shape[1]):
        axes[0].plot(ts, xs[:, mode_idx], label=f"Mode {mode_idx + 1}")
        axes[1].plot(ts, vs[:, mode_idx], label=f"Mode {mode_idx + 1}")

    axes[0].set_ylabel("Displacement")
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("Time")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


class TimeResponseTests(unittest.TestCase):
    def test_time_response_accepts_single_case_one_tone_excitation(self):
        model = make_model(n_modes=2)
        excitation = poscidyn.DirectExcitation(
            drive_frequencies=np.array([1.0]),
            drive_amplitudes=np.array([0.1]),
            modal_forces=np.array([1.0, 0.25]),
        )
        solver = DummySolver()

        ts, xs, vs = poscidyn.time_response(
            model=model,
            excitation=excitation,
            initial_displacement=np.array([0.0, 0.0]),
            initial_velocity=np.array([0.0, 0.0]),
            solver=solver,
            only_save_steady_state=True,
        )

        self.assertEqual(ts.shape, (4,))
        self.assertEqual(xs.shape, (4, 2))
        self.assertEqual(vs.shape, (4, 2))
        self.assertEqual(solver.received["driving_frequency"].shape, (1,))
        self.assertEqual(solver.received["driving_amplitude"].shape, (2,))
        np.testing.assert_allclose(
            np.asarray(solver.received["driving_amplitude"]),
            np.array([0.1, 0.025]),
        )
        self.assertTrue(solver.received["kwargs"]["only_save_steady_state"])

    @unittest.skipUnless(plt is not None, "matplotlib not installed")
    def test_time_response_plot_smoke(self):
        model = poscidyn.Nonlinear(
            omega_0=np.array([1.0]),
            Q=np.array([5.0]),
            a=np.zeros((1, 1, 1)),
            b=np.zeros((1, 1, 1, 1)),
        )
        excitation = make_excitation(
            drive_frequency=1.0,
            drive_amplitude=0.05,
            modal_forces=np.array([1.0]),
        )
        solver = poscidyn.TimeIntegration(
            n_time_steps=64,
            max_steps=4096,
            rtol=1e-3,
            atol=1e-6,
            t_steady_state_factor=0.5,
        )

        ts, xs, vs = poscidyn.time_response(
            model=model,
            excitation=excitation,
            initial_displacement=np.array([0.0]),
            initial_velocity=np.array([0.0]),
            solver=solver,
            only_save_steady_state=True,
        )

        xs_np = np.asarray(xs)
        vs_np = np.asarray(vs)
        self.assertTrue(np.isfinite(xs_np).all())
        self.assertTrue(np.isfinite(vs_np).all())
        self.assertGreater(np.max(np.abs(xs_np)), 0.0)
        self.assertGreater(np.max(np.abs(vs_np)), 0.0)

        fig, axes = plot_time_response(
            ts=ts,
            xs=xs,
            vs=vs,
            title="Time response smoke test",
        )
        self.assertEqual(len(axes[0].lines), model.n_modes)
        self.assertEqual(len(axes[1].lines), model.n_modes)
        fig.canvas.draw()
        if _PLOT_ON_MAIN:
            plt.show()
        plt.close(fig)

    def test_time_response_rejects_multiple_frequencies(self):
        model = make_model()
        excitation = poscidyn.DirectExcitation(
            drive_frequencies=np.array([0.9, 1.0]),
            drive_amplitudes=np.array([0.1]),
            modal_forces=np.array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "exactly one drive frequency"):
            poscidyn.time_response(
                model=model,
                excitation=excitation,
                initial_displacement=np.array([0.0]),
                initial_velocity=np.array([0.0]),
                solver=DummySolver(),
            )

    def test_time_response_rejects_multiple_amplitudes(self):
        model = make_model()
        excitation = poscidyn.DirectExcitation(
            drive_frequencies=np.array([1.0]),
            drive_amplitudes=np.array([0.1, 0.2]),
            modal_forces=np.array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "exactly one drive amplitude"):
            poscidyn.time_response(
                model=model,
                excitation=excitation,
                initial_displacement=np.array([0.0]),
                initial_velocity=np.array([0.0]),
                solver=DummySolver(),
            )

    def test_time_response_rejects_modal_force_mismatch(self):
        model = make_model(n_modes=2)
        excitation = poscidyn.DirectExcitation(
            drive_frequencies=np.array([1.0]),
            drive_amplitudes=np.array([0.1]),
            modal_forces=np.array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "Number of modes"):
            poscidyn.time_response(
                model=model,
                excitation=excitation,
                initial_displacement=np.array([0.0, 0.0]),
                initial_velocity=np.array([0.0, 0.0]),
                solver=DummySolver(),
            )


if __name__ == "__main__":
    unittest.main()
