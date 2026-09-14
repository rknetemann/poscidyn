"""End-to-end smoke check for the public result objects.

Run directly to execute the simulations and display the plots.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib.pyplot as plt
import numpy as np
import poscidyn


def build_solver():
    b = np.zeros((1,1,1,1))
    b[0,0,0,0] = 0.1
    oscillator = poscidyn.NonlinearOscillator(
        omega_0=np.array([1.0]), Q=np.array([10.0]), b=b, n_modes=1
    )
    excitation = poscidyn.DirectHarmonicExcitation(
        f_d=np.array([0.25]), omega=np.array([1.0]), lambdas=np.array([1.0])
    )
    multistart = poscidyn.LinearResponse(n_init_cond=8)
    return poscidyn.TimeIntegration(
        oscillator=oscillator, excitation=excitation,
        response_measure=poscidyn.Demodulation(multiples=(1.0,), window=None),
        n_time_steps=32, periods_to_retain=1, max_steps=4096,
        rtol=1e-5, atol=1e-7, throw=False, multistart=multistart, t_steady_state_factor = 1.2
    )


def run_smoke_test():
    solver = build_solver()
    time = solver.time_response(x0=np.array([0.0]), v0=np.array([0.0]), only_save_steady_state=False)
    assert time.time.ndim == 1
    assert time.displacement.shape == time.velocity.shape
    assert time.displacement.shape[0] == time.time.shape[0]
    assert np.all(np.isfinite(np.asarray(time.displacement)))

    frequencies = np.linspace(0.7, 1.3, 250)
    sweep = solver.frequency_sweep(frequencies)
    # JAX may downcast the coordinate when x64 is disabled.
    assert np.allclose(np.asarray(sweep.frequency), frequencies)
    assert sweep.forward.modal.amplitude is not None
    assert sweep.backward.total.amplitude is not None
    assert np.all(np.isfinite(np.asarray(sweep.forward.total.amplitude)))
    assert 0 <= sweep.stats["success_rate"] <= 1
    return time, sweep


def plot_results(time, sweep):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(time.time, time.displacement[:, 0], label="displacement")
    axes[0].plot(time.time, time.velocity[:, 0], label="velocity")
    axes[0].set(title="Time response", xlabel="Time", ylabel="Response")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    frequency = np.asarray(sweep.frequency)

    def branch_amplitude(branch):
        amplitude = np.asarray(branch.total.amplitude)
        # A one-parameter smoke test must produce one response value per
        # frequency.  Remove only singleton axes; never let Matplotlib
        # interpret remaining columns as separate curves.
        amplitude = np.squeeze(amplitude)
        if amplitude.ndim != 1 or amplitude.shape != frequency.shape:
            raise ValueError(
                "Expected one total-amplitude curve per sweep direction; "
                f"got shape {amplitude.shape} for frequency shape {frequency.shape}."
            )
        return amplitude

    # Keep colour tied to sweep direction: one colour per branch.
    axes[1].plot(
        frequency,
        branch_amplitude(sweep.forward),
        color="tab:blue",
        label="forward",
    )
    axes[1].plot(
        frequency,
        branch_amplitude(sweep.backward),
        color="tab:orange",
        label="backward",
    )
    axes[1].set(title="Frequency sweep", xlabel="Excitation frequency", ylabel="Amplitude")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    time_response, frequency_sweep = run_smoke_test()
    print("Time response:", time_response.displacement.shape)
    print("Frequency sweep:", frequency_sweep.forward.total.amplitude.shape)
    print("Sweep success rate:", frequency_sweep.stats["success_rate"])
    plot_results(time_response, frequency_sweep)
    plt.show()
