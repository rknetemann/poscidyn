# Choosing a solver

For the public Poscidyn workflow, use `TimeIntegration`.

## Recommended default

`TimeIntegration` is the solver that currently matches the documented `time_response(...)` and `frequency_sweep(...)` APIs. It:

- integrates the nonlinear equations directly,
- works with response measures such as `Demodulation`, `Min`, `Max`, and `L2`,
- fits the multistart plus artificial-sweep workflow used throughout the package.

Start there unless you have a specific reason not to.

## Other solver classes

The package also contains `MultipleShootingSolver` and `CollocationSolver`, but they should currently be treated as experimental.

- Their APIs are not aligned with the main `frequency_sweep(...)` helper.
- Their documentation is not yet complete.
- They are better viewed as ongoing development work than drop-in replacements for `TimeIntegration`.

## Rule of thumb

- Use `TimeIntegration` for real work today.
- Explore shooting or collocation only if you are developing Poscidyn itself or extending the solver layer.
