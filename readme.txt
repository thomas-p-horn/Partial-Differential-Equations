Cahn-Hilliard Phase Separation & Boundary Value Problems
Author: Thomas Horn
================================================================================

OVERVIEW
--------
This project contains two simulation tools:

  1. bvp.py          - Solves boundary value problems (Poisson equation) for
                       electrostatic and magnetostatic fields using Jacobi
                       or Gauss-Seidel algorithms.

  2. cahn-hilliard.py - Simulates phase separation dynamics using the
                        Cahn-Hilliard equation on a 2D lattice.


DEPENDENCIES
------------
  numpy, matplotlib, scipy, numba, pandas


--------------------------------------------------------------------------------
 bvp.py
--------------------------------------------------------------------------------

Solves the Poisson equation for two physical setups:

  Electrostatic  -  A point charge at the centre of a 3D cubic lattice.
                    Solves for the electrostatic potential phi and electric
                    field E.

  Magnetic       -  A point current at the centre of a 2D lattice, equivalent
                    to a wire along the z axis.
                    Solves for the magnetic vector potential Az and magnetic
                    field B.

Two solvers are available:

  jacobi         -  Standard Jacobi iteration (vectorised with NumPy).
  gauss-seidel   -  Gauss-Seidel iteration with Successive Over-Relaxation
                    (SOR), accelerated with Numba JIT compilation.

USAGE
-----
  python bvp.py [options]

OPTIONS
-------
  -p, --problem     Problem to solve: 'electrostatic', 'magnetic', or 'sor'
                    (Default: electrostatic)

  -a, --algorithm   Solver algorithm: 'jacobi' or 'gauss-seidel'
                    (Default: jacobi)

  -l, --length      Lattice size L (produces an LxLxL grid for electrostatic,
                    LxL for magnetic). (Default: 100)

  -t, --tolerance   Convergence tolerance. (Default: 1e-5)

  -m, --max         Maximum number of iterations. (Default: 1000)

  -o, --omega       SOR relaxation parameter. Set 1 < omega < 2 for
                    over-relaxation. (Default: 1.0, i.e. no over-relaxation)

  --problem sor     Sweeps omega from 1.0 to 1.99 and plots the number of
                    steps to convergence vs omega.

EXAMPLES
--------
  python bvp.py -p electrostatic -a gauss-seidel -l 50 -o 1.8
  python bvp.py -p magnetic -a jacobi -l 100
  python bvp.py -p sor -l 50 -t 1e-4

OUTPUT
------
  All output files are saved to the 'bvp data/' directory:

    Electrostatic Potential.csv / .png
    Electrostatic Potential 1D.png    (with 1/r fit)
    Electric Field.csv / .png
    Electric Field 1D.png             (with 1/r^2 fit)
    Magnetic Potential.csv / .png
    Magnetic Potential 1D.png         (with exp(-r) fit)
    Magnetic Field.csv / .png
    Magnetic Field 1D.png             (with 1/r fit)
    Omega.png                         (SOR convergence sweep)


--------------------------------------------------------------------------------
 cahn-hilliard.py
--------------------------------------------------------------------------------

Simulates phase separation on a 2D LxL lattice using the Cahn-Hilliard
equation with periodic boundary conditions. The order parameter phi represents
local composition, evolving under a free energy functional with parameters
a (bulk) and k (interfacial stiffness).

USAGE
-----
  python cahn-hilliard.py [options]

OPTIONS
-------
  --action          What to run: 'animate', 'measure', or 'draw'
                    (Default: animate)

  -l, --length      Lattice size L. (Default: 100)

  -a                Bulk free energy parameter. (Default: 1.0)
  -k                Interfacial energy parameter. (Default: 1.0)
  -M                Mobility parameter. (Default: 1.0)
  -dt               Time step size. (Default: 2e-4)
  -dx               Lattice spacing. (Default: 1.0)

  -p0, --phi0       Mean initial value of phi. (Default: 0.0)
  -sp0              Standard deviation of initial phi distribution.
                    (Default: 0.1)

  -t, --tolerance   Convergence tolerance: std dev of free energy over 500
                    steps below which the system is considered converged.
                    (Default: 0.005)

ACTIONS
-------
  animate    Runs the simulation with a live matplotlib display. Close the
             figure window to stop.

  measure    Evolves the system for up to 500,000 steps, recording the mean
             free energy at each step. Stops early when the free energy
             converges (see --tolerance). Saves a CSV and plots the result.

  draw       Plots the free energy over time from a previously saved CSV
             (requires a prior 'measure' run for the same phi0 value).

EXAMPLES
--------
  python cahn-hilliard.py --action animate -p0 0.0
  python cahn-hilliard.py --action measure -p0 0.5 -l 64
  python cahn-hilliard.py --action draw -p0 0.5

OUTPUT
------
  All output files are saved to the 'cahn-hilliard data/' directory:

    Free Energy {phi0}.csv
    Free Energy {phi0}.png

================================================================================