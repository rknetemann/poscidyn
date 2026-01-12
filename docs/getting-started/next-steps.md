Now that you have installed **Poscidyn**, understand its core concepts, and reviewed the limitations, here’s a suggested path to get productive quickly.

## Confirm fit and guardrails
- Revisit [For who is Poscidyn?](../for-who-is-poscidyn.md) and [Where Poscidyn shines](../where-poscidyn-shines.md) to ensure the package matches your workload and performance expectations.
- Keep the [Limitations](limitations.md) page handy when interpreting results, especially for synthetic sweeps and multistart strategies.

## Start with a quick run
- Keep the mental model from [Understanding Poscidyn](understanding-poscidyn.md) in mind: pick an oscillator model, choose an excitation model, and decide whether you want a time response or a frequency sweep.
- Walk through the **Basic usage** pages in order:
  - [Oscillator models](../usage/basic-usage/oscillator-models.md)
  - [Excitation models](../usage/basic-usage/excitation-models.md)
  - [Time response](../usage/basic-usage/time-response.md)
  - [Frequency sweep](../usage/basic-usage/frequency-sweep.md)
- Validate your setup with a concrete example, e.g. the Duffing oscillator [time response](../examples/time-response/duffing-oscillator.md) or [frequency sweep](../examples/frequency-sweep/duffing-oscillator.md).

## Dial in configurations
- Move to **Advanced usage** to tune performance and robustness:
  - [Choosing a solver](../usage/advanced-usage/choosing-a-solver.md)
  - [Configuring multistart strategies](../usage/advanced-usage/configuring-multistart-strategies.md)
  - [Configuring sweep methods](../usage/advanced-usage/configuring-sweep-methods.md)
- For heavy workloads, consult [Where Poscidyn shines](../where-poscidyn-shines.md) for JAX tips and batching guidance.

## Extend and iterate
- If built-in components are not enough, see [Extending Poscidyn](../usage/extending-poscidyn.md) to add custom models or excitation types.
- Revisit the [Limitations](limitations.md) and [Future work](../future-work.md) pages when exploring edge cases or planning contributions.
