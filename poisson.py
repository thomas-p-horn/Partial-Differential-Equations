import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import argparse
import time

class PoissonSolver3D:
    def __init__(self, N=50, tolerance=1e-5, max_steps=10000, omega=1.0):
        self.N = N
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.omega = omega

        self.phi = np.zeros((N, N, N))
        self.rho = np.zeros((N, N, N))

    def set_point_charge(self, position=None, magnitude=1.0):
        if position is None:
            position = (self.N // 2, self.N // 2, self.N // 2)
        self.rho[position] = magnitude

    def residual(self, phi_old):
        return np.max(np.abs(self.phi - phi_old))

    def solve(self, method):
        if method == "jacobi":
            return self.solve_jacobi()
        elif method == "gauss-seidel":
            return self.solve_gauss_seidel()
        else:
            raise ValueError("Unknown method")

    # @njit
    def solve_jacobi(self):

        for i in range(self.max_steps):
            phi_old = self.phi.copy()

            self.phi[1:-1,1:-1,1:-1] = (
                phi_old[2:,1:-1,1:-1] +
                phi_old[:-2,1:-1,1:-1] +
                phi_old[1:-1,2:,1:-1] +
                phi_old[1:-1,:-2,1:-1] +
                phi_old[1:-1,1:-1,2:] +
                phi_old[1:-1,1:-1,:-2] +
                self.rho[1:-1,1:-1,1:-1]
            ) / 6.0

            if self.residual(phi_old) < self.tolerance:
                print(f"Converged in {i} iterations")
                break

        return self.phi

    def solve_gauss_seidel(self):
        for i in range(self.max_steps):
            phi_old = self.phi.copy()

            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    for k in range(1, self.N-1):
                        new_val = (
                            self.phi[i+1,j,k] + self.phi[i-1,j,k] +
                            self.phi[i,j+1,k] + self.phi[i,j-1,k] +
                            self.phi[i,j,k+1] + self.phi[i,j,k-1] +
                            self.rho[i,j,k]
                        ) / 6.0

                        # Successive Over-Relaxation (SOR)
                        self.phi[i,j,k] += self.omega * (new_val - self.phi[i,j,k])

            if self.residual(phi_old) < self.tolerance:
                print(f"Converged in {i} iterations")
                break

        return self.phi

    def compute_electric_field(self):
        Ex = np.zeros_like(self.phi)
        Ey = np.zeros_like(self.phi)
        Ez = np.zeros_like(self.phi)

        Ex[1:-1,:,:] = -(self.phi[2:,:,:] - self.phi[:-2,:,:]) / 2
        Ey[:,1:-1,:] = -(self.phi[:,2:,:] - self.phi[:,:-2,:]) / 2
        Ez[:,:,1:-1] = -(self.phi[:,:,2:] - self.phi[:,:,:-2]) / 2

        return Ex, Ey, Ez

    def plot_midplane(self):
        mid = self.N // 2

        plt.figure()
        plt.title("Potential (midplane slice)")
        plt.imshow(self.phi[:,:,mid], origin='lower')
        plt.colorbar()
        plt.show()

    def plot_field(self):
        Ex, Ey, _ = self.compute_electric_field()
        mid = self.N // 2

        plt.figure()
        plt.title("Electric Field (midplane)")
        plt.quiver(Ex[:,:,mid], Ey[:,:,mid])
        plt.show()


if __name__ == "__main__":
    solver = PoissonSolver3D(
        N=50,
        tolerance=1e-5,
        max_steps=5000,
        omega=1.  # try 1.0–1.9 for SOR tuning
    )

    solver.set_point_charge()

    t1 = time.time()
    solver.solve("jacobi")
    t2 = time.time()
    print(f"Solve time: {t2-t1:.3f}s")

    solver.plot_midplane()
    solver.plot_field()