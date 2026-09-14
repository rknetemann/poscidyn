# Extending Poscidyn

Poscidyn is intended to become easier to extend, but extension interfaces are
not yet a fully stable public contract. Treat custom components as development
work and validate them against the same examples and numerical checks used for
built-in components.

## Current component boundaries

- An oscillator provides internal dynamics through `f_i(t, y, args, **kwargs)`
  and must expose the modal dimension and a settling-time estimate used by time
  integration.
- An excitation provides `f_e(t, y, args, **kwargs)`. Periodic excitations
  additionally carry `omega` and `lambdas`.
- A response measure is callable as `measure(xs, ts, drive_omega)` and returns
  modal and total blocks containing amplitude, phase, and response-frequency
  values.
- A multistart strategy supplies a grid of initial states for a model and
  requested frequencies.
- A synthetic-sweep strategy selects paths from the resulting candidate data.

## Safe extension workflow

1. Start from a supported quickstart and replace one component only.
2. Write a small deterministic test for dimensions, finite outputs, and known
   limiting behaviour.
3. Compare a custom component with a direct time response before trusting a
   frequency sweep.
4. Document units, shape conventions, and limitations alongside the component.

Future releases will make these interfaces more explicit and stable. The
[roadmap](../../future-work.md) records that direction without implying
backward compatibility for current internals.
