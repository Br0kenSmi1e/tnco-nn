"""Generate the protected evaluation suite (fixed instances, fixed seeds).

Run once:  .venv/bin/python eval/make_suite.py
Do not modify generated instances without permission (protected evaluation).

Tracks:
  - exact: small instances (n <= 20) where ExhaustiveSearch gives the
    optimum; used to measure optimality gaps.
  - main:  moderate instances (n up to ~200) for the primary comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.instances import erdos_renyi, random_regular, square_lattice

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    instances = []

    # exact track (n <= 20)
    for seed in range(3):
        instances.append(("exact", random_regular(20, 3, seed=seed)))
    for seed in range(2):
        instances.append(("exact", erdos_renyi(20, 0.1, seed=seed)))
    instances.append(("exact", square_lattice(4, 5)))

    # main track
    for seed in range(2):
        instances.append(("main", random_regular(50, 3, seed=seed)))
        instances.append(("main", random_regular(100, 3, seed=seed)))
        instances.append(("main", random_regular(100, 4, seed=seed)))
        instances.append(("main", random_regular(200, 3, seed=seed)))
        instances.append(("main", erdos_renyi(100, 0.05, seed=seed)))
    instances.append(("main", square_lattice(6, 6)))
    instances.append(("main", square_lattice(8, 8)))
    # varied-extent instances
    instances.append(("main", random_regular(100, 3, chi=16, seed=0)))
    instances.append(("main", square_lattice(6, 6, chi=16)))

    for track, inst in instances:
        path = os.path.join(OUT, f"{track}__{inst.name}.json")
        inst.save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
