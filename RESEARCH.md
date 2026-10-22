# Research notebook

## Objective

Develop a substantively new neural-network-guided tensor-network
contraction-order algorithm that is competitive with strong existing methods
on general graph and hypergraph networks. Establish when, why, and under
what assumptions the learned component helps.

Minimize total contraction work, without a space-complexity constraint or
read-write-complexity weighting. Diagnostics, routine model variants, and
faster implementations are supporting work, not the objective.

## Comparison criteria

Evaluate best contraction work versus elapsed planning time at comparable
budgets, including training, inference, and unsuccessful search. Compare
against strong non-learned methods, including TreeSA, across relevant
networks and seeds. Separate algorithmic gains (learned guidance) from
implementation speedups and extra computation.

## Starting state

Fresh project. No inherited algorithmic choices; the codebase is empty
beyond this notebook. First task is to establish a baseline pipeline and a
protected evaluation suite before any learned component is introduced.

## Resources

Use local resources reasonably. Do not incur paid external compute or change
protected evaluations without permission.

## Understanding and evidence

**Current verdict:** nothing established yet. No baselines, no evaluation
suite, no learned component.
