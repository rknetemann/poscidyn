# Configure time integration

`TimeIntegration` is the supported Poscidyn solver. It integrates the first
order state system with Diffrax's adaptive `Tsit5` method, estimates a
transient duration, and retains a final time window for analysis.

## Important settings

- `rtol`, `atol`: relative and absolute tolerances for adaptive integration.
- `max_steps`: hard upper limit on internal integration steps per trajectory.
- `n_time_steps`: saved samples per retained period block. Set it explicitly
  when you need predictable resolution or use JIT/vmap around the call.
- `periods_to_retain`: number of final drive periods retained for a periodic
  response. The default is 4.
- `t_steady_state_factor`: safety factor multiplying the oscillator's
  linear-model transient estimate.
- `max_order_superharmonics`: used only when Poscidyn estimates
  `n_time_steps` automatically.
- `throw`: if `True`, integration failures are raised by Diffrax instead of
  being represented in sweep statistics.

## A defensible tuning sequence

Begin with a representative `time_response` and inspect the transient. Then
set `n_time_steps` high enough to resolve the highest response content relevant
to the study. Finally, tighten tolerances or raise `max_steps` only after
checking that numerical failures or visible sampling artefacts justify it.

For frequency sweeps, non-finite trajectories are counted in `result.stats`.
Do not use an attractive selected curve to hide a poor completion rate.

The [time-response quickstart](../../../quickstart/time-response.md) shows the
minimal construction; [limitations](../../limitations.md) explains the
transient and computational assumptions.
