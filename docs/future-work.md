# Status and roadmap

Poscidyn is under active development. This page makes the project's direction
visible without presenting unfinished work as a release commitment.

## Available and supported

The documented solver-instance workflow supports `NonlinearOscillator`,
`DirectHarmonicExcitation`, `TimeIntegration`, `LinearResponse`,
`NearestNeighbour`, and the built-in response measures. These are the
components used by the quickstarts, guides, and reference.

## Experimental work

Prototype collocation and multiple-shooting solvers exist in the source tree.
They are **not** part of the supported public contract: their interfaces are
not aligned with `TimeIntegration`, their validation is incomplete, and they
should not be used as drop-in production solvers. They are retained as research
and development work rather than documented as normal user features.

The same distinction applies to partially implemented model or excitation
prototypes. A class appearing in the repository does not by itself make it a
supported feature.

## Intended directions

The following are design directions, not promises of order or delivery date.

- **Periodic-solution methods:** collocation and shooting methods for finding
  periodic solutions more directly.
- **Continuation and hybrid methods:** localized continuation combined with
  batched multistart calculations, aiming to reduce memory pressure while
  retaining parallel throughput.
- **Hybrid dynamics:** support for systems with events, switching, impacts, or
  other non-smooth behaviour.
- **Excitation families:** robust parametric support, multi-tone drives, and
  additional custom excitation interfaces.
- **Model and analysis ecosystem:** more canonical oscillators, stability and
  bifurcation analysis, and focused plotting tools.
- **Extension ergonomics:** clearer, stable interfaces that let users add
  models and methods without depending on solver internals.

## Documentation policy as the project grows

New capabilities will be labelled in one of three ways: **supported**,
**experimental**, or **planned**. A feature becomes supported only after it has
a stable public interface, a tested example, reference documentation, and an
explicit statement of important limitations.

This policy keeps the first-use path small while allowing the documentation to
remain a useful map of the research frontier.
