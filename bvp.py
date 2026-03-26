import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import argparse
import time
from scipy.optimize import curve_fit

def double_exp(x, a, b, mid):
    y = np.zeros_like(x)
    lower = (x <= mid)
    upper = (x > mid)
    y[lower] = a * np.exp(b * (x[lower] - mid))
    y[upper] = a * np.exp(-b * (x[upper] - mid))
    return y

def inv_x(x, a, mid):
    return a / np.abs(x - mid)

def inv_x2(x, a, mid):
    return a / np.abs(x - mid)**2


@njit
def gauss_seidel_3D(phi, rho, omega, max_steps, tolerance):
    L = phi.shape[0]
    for n in range(max_steps):
        phi_old = phi.copy()

        for i in range(1, L-1):
            for j in range(1, L-1):
                for k in range(1, L-1):
                    new_val = (
                        phi[i+1,j,k] + phi[i-1,j,k] +
                        phi[i,j+1,k] + phi[i,j-1,k] +
                        phi[i,j,k+1] + phi[i,j,k-1] +
                        rho[i,j,k]
                    ) / 6.0

                    # Over relaxation
                    phi[i,j,k] = (1 - omega) * phi_old[i,j,k] + omega * new_val

        if np.max(np.abs(phi - phi_old)) <= tolerance:
            print(f"Converged in {n} steps")
            break
    
    return phi, n

@njit
def gauss_seidel_2D(A, J, omega, max_steps, tolerance):
    L = A.shape[0]
    for n in range(max_steps):
        A_old = A.copy()

        for i in range(1, L-1):
            for j in range(1, L-1):
                new_val = (
                    A[i+1,j] + A[i-1,j] +
                    A[i,j+1] + A[i,j-1] +
                    J[i,j]
                ) / 4.0

                # Successive Over-Relaxation (SOR)
                A[i,j] = (1 - omega) * A_old[i,j] + omega * new_val

        if np.max(np.abs(A - A_old)) <= tolerance:
            print(f"Converged in {n} steps")
            break
    
    return A, n



class PoissonSolver:
    def __init__(self, L, tolerance, max_steps, omega):
        self.L = L
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.omega = omega

        self.phi = np.zeros((L, L, L))
        self.set_charge()

    def set_charge(self, position=None, charge=1.):
        self.rho= np.zeros((self.L, self.L, self.L))
        
        if position is None:
            mid = self.L // 2
            position = (mid, mid, mid)
        self.rho[position] = charge
        
    def residual(self, phi_old):
        return np.max(np.abs(self.phi - phi_old))

    def solve_jacobi(self):

        for n in range(self.max_steps):
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
                print(f"Converged in {n} steps")
                break

        return self.phi

    def solve_gauss_seidel(self):
        self.phi, n = gauss_seidel_3D(self.phi, self.rho, self.omega, self.max_steps, self.tolerance)
        return self.phi, n
    
    def electric_field(self):
        Ex = np.zeros_like(self.phi)
        Ey = np.zeros_like(self.phi)
        Ez = np.zeros_like(self.phi)

        # Grad
        Ex[1:-1,:,:] = -(self.phi[2:,:,:] - self.phi[:-2,:,:]) / 2
        Ey[:,1:-1,:] = -(self.phi[:,2:,:] - self.phi[:,:-2,:]) / 2
        Ez[:,:,1:-1] = -(self.phi[:,:,2:] - self.phi[:,:,:-2]) / 2

        return Ex, Ey, Ez

    def plot_midplane(self):
        mid = self.L // 2

        np.savetxt('bvp data/Electrostatic Potential.csv', self.phi[:, :, mid])

        plt.figure()
        plt.title(r"Electrostatic Potential $\phi$ (midplane slice)")
        plt.xlabel(r"x")
        plt.ylabel(r"y")
        plt.imshow(self.phi[:,:,mid], origin='lower', cmap='hot')
        plt.colorbar()
        plt.savefig("bvp data/Electrostatic Potential.png", dpi=300, bbox_inches='tight')
        plt.close()


        x = np.arange(len(self.phi[:, mid, mid]), dtype=np.float64)
        y_data = self.phi[:, mid, mid]
        mask = np.arange(len(x)) != mid
        popt, _ = curve_fit(inv_x, x[mask], y_data[mask], p0=[1., float(mid)])

        x_plot = x[mask]
        y_plot = inv_x(x_plot, *popt)

        plt.figure()
        plt.title(r"Electrostatic Potential $\phi$ (x direction)")
        plt.xlabel(r"x")
        plt.ylabel(r"$\phi$")
        plt.plot(x, self.phi[:, mid, mid], label='Data', c='darkslateblue')
        plt.plot(x_plot, y_plot, label='1/r Fit', c='red')
        plt.legend()
        plt.savefig('bvp data/Electrostatic Potential 1D.png', dpi=300, bbox_inches='tight')
        plt.close()



    def plot_field(self):
        Ex, Ey, _ = self.electric_field()
        mid = self.L // 2

        ix, iy = np.meshgrid(np.arange(self.L), np.arange(self.L), indexing='ij')
        
        data = np.column_stack([
            ix.ravel(), iy.ravel(),
            self.phi[:, :, mid].ravel(),
            Ex[:, :, mid].ravel(),
            Ey[:, :, mid].ravel()
        ])
        
        np.savetxt('bvp data/Electric Field.csv', data, delimiter=',',
                header='x,y,phi,Ex,Ey', comments='')

        plt.figure()
        plt.title("Electric Field (midplane)")
        plt.quiver(Ex[:,:,mid], Ey[:,:,mid])
        plt.savefig("bvp data/Electric Field.png", dpi=300, bbox_inches='tight')
        plt.close()


        E_mag = np.sqrt(Ex[:, :, mid]**2 + Ey[:, :, mid]**2)

        x = np.arange(len(E_mag[:, mid]), dtype=np.float64)
        y_data = E_mag[:, mid]
        mask = np.arange(len(x)) != mid
        popt, _ = curve_fit(inv_x2, x[mask], y_data[mask], p0=[1., float(mid)])

        x_plot = x[mask]
        y_plot = inv_x2(x_plot, *popt)

        plt.figure()
        plt.title(r"Electric Field Strength $|E|$ (x direction)")
        plt.xlabel(r"x")
        plt.ylabel(r"$|E|$")
        plt.plot(x, y_data, label='Data', c='darkslateblue')
        plt.plot(x_plot, y_plot, label=f'1/x$^2$ Fit', c='red')
        plt.legend()
        plt.savefig('bvp data/Electric Field 1D.png', dpi=300, bbox_inches='tight')
        plt.close()



class MagneticSolver():

    def __init__(self, L, tolerance=1e-5, max_steps=10000, omega=1.0):
        self.L = L
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.omega = omega
        
        self.Az = np.zeros((L, L))
        self.set_current()

    def set_current(self, current=1.):
        self.Jz = np.zeros((self.L, self.L))
        mid = self.L // 2
        self.Jz[mid, mid] = current
    
    def residual(self, phi_old):
        return np.max(np.abs(self.Az - phi_old))

    def solve_jacobi(self):

        for n in range(self.max_steps):
            Az_old = self.Az.copy()

            self.Az[1:-1,1:-1] = (
                Az_old[2:,1:-1] +
                Az_old[:-2,1:-1] +
                Az_old[1:-1,2:] +
                Az_old[1:-1,:-2] +
                self.Jz[1:-1,1:-1]
            ) / 4.0

            if self.residual(Az_old) < self.tolerance:
                print(f"Converged in {n} steps")
                break

        return self.Az

    def solve_gauss_seidel(self):
        self.Az, n = gauss_seidel_2D(self.Az, self.Jz, self.omega, self.max_steps, self.tolerance)
        return self.Az, n

    def magnetic_field(self):
        Bx = np.zeros_like(self.Az)
        By = np.zeros_like(self.Az)

        # Curl (All of A is in A_z, other cross term derivatives vanish)
        Bx[1:-1, :] = (self.Az[2:, :] - self.Az[:-2, :]) / 2
        By[:, 1:-1] = -(self.Az[:, 2:] - self.Az[:, :-2]) / 2

        return Bx, By

    def plot_midplane(self):
        mid = self.L // 2

        np.savetxt('bvp data/Magnetic Potential.csv', self.Az[:, :])

        plt.figure()
        plt.title(r"Magnetic Potential $A_z$ (midplane slice)")
        plt.xlabel(r"x")
        plt.ylabel(r"y")
        plt.imshow(self.Az[:,:], origin='lower', cmap='hot')
        plt.colorbar()
        plt.savefig("bvp data/Magnetic Potential.png", dpi=300, bbox_inches='tight')
        plt.close()

        x = np.arange(len(self.Az[:, mid]), dtype=np.float64)
        y_data = self.Az[:, mid]
        mask = np.arange(len(x)) != mid
        popt, _ = curve_fit(double_exp, x[mask], y_data[mask], p0=[1., 1., float(mid)])

        x_plot = x[mask]
        y_plot = double_exp(x_plot, *popt)

        plt.figure()
        plt.title(r"Magnetic Potential $A_z$ (x direction)")
        plt.xlabel(r"x")
        plt.ylabel(r"$A_z$")
        plt.plot(x, y_data, label='Data', c='darkslateblue')
        plt.plot(x_plot, y_plot, label='exp(-r) Fit', c='red')
        plt.legend()
        plt.savefig('bvp data/Magnetic Potential 1D.png', dpi=300, bbox_inches='tight')
        plt.close()



    def plot_field(self):
        Bx, By = self.magnetic_field()

        ix, iy = np.meshgrid(np.arange(self.L), np.arange(self.L), indexing='ij')
        
        data = np.column_stack([
            ix.ravel(), iy.ravel(),
            self.Az[:, :].ravel(),
            Bx[:, :].ravel(),
            By[:, :].ravel()
        ])
        
        np.savetxt('bvp data/Magnetic Field.csv', data, delimiter=',',
                header='x,y,phi,Bx,By', comments='')

        plt.figure()
        plt.title(r"Magnetic Field $B$ (midplane)")
        plt.xlabel(r"x")
        plt.ylabel(r"y")
        plt.quiver(Bx[:,:], By[:,:])
        plt.savefig("bvp data/Magnetic Field.png", dpi=300, bbox_inches='tight')
        plt.close()


        mid = self.L // 2

        B_mag = np.sqrt(Bx[:, :]**2 + By[:, :]**2)

        x = np.arange(len(B_mag[:, mid]), dtype=np.float64)
        y_data = B_mag[:, mid]
        mask = np.arange(len(x)) != mid
        popt, _ = curve_fit(inv_x, x[mask], y_data[mask], p0=[1., float(mid)])

        x_plot = x[mask]
        y_plot = inv_x(x_plot, *popt)

        plt.figure()
        plt.title(r"Magnetic Field Strength $|B|$ (x direction)")
        plt.xlabel(r"x")
        plt.ylabel(r"$|B|$")
        plt.plot(x, y_data, label='Data', c='darkslateblue')
        plt.plot(x_plot, y_plot, label=f'1/x Fit', c='red')
        plt.legend()
        plt.savefig('bvp data/Magnetic Field 1D.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Runs a solver for the Poisson equation or for the magnetic field equation")
    parser.add_argument("-p", "--problem", help="Which problem to solve. 'electrostatic' or 'magnetic' or 'SOR'. Default='electrostatic'.", type=str, default='electrostatic')
    parser.add_argument("-a", "--algorithm", help="Algorithm used to solve problem. 'jacobi' or 'gauss-seidel'. Default='jacobi'.", type=str, default='jacobi')
    parser.add_argument("-l", "--length", help="LxLxL size of lattice. Default=100.", type=int, default=100)
    parser.add_argument("-t", "--tolerance", help="Tolerance requirement to terminate solver. (Solver runs until highest value for difference between each step is below tolerance.) Default=1e-5.", type=float, default=1e-5)
    parser.add_argument("-m", "--max", help="Maximum simulation steps. Default=1000.", type=int, default=1000)
    parser.add_argument("-o", "--omega", help="Over-relaxation parameter. Set between 1<omega<2 for over-relaxation. Default=1.0 (no over-relaxation).", type=float, default=1.)


    args = parser.parse_args()
    problem = args.problem
    algorithm = args.algorithm
    L = args.length
    tolerance = args.tolerance
    max_steps = args.max
    omega = args.omega

    if problem.lower() == 'electrostatic':
        solver = PoissonSolver(
            L,
            tolerance,
            max_steps,
            omega
        )
    elif problem.lower() == 'magnetic':
        solver = MagneticSolver(
            L,
            tolerance,
            max_steps,
            omega
        )
    elif problem.lower() == 'sor':
        n_list = []
        omega_list = np.linspace(1, 1.99, 50)
        for omega in omega_list:
            solver = PoissonSolver(
                L,
                tolerance,
                max_steps,
                omega
            )
            _, n = solver.solve_gauss_seidel()
            n_list.append(n)

        plt.plot(omega_list, n_list, c='darkslateblue')
        plt.xlabel(r'$\omega$')
        plt.ylabel('Number of steps to solve')
        plt.savefig('bvp data/Omega.png', dpi=300, bbox_inches='tight')
        quit()

    else:
        raise ValueError("Problem not recognised. Please input 'poisson' or 'magnetic' or 'sor'.")

    t1 = time.time()
    if algorithm.lower() == 'jacobi':
        solver.solve_jacobi()
    elif algorithm.lower() == 'gauss-seidel':
        solver.solve_gauss_seidel()
    else:
        raise ValueError("Algorithm not recognised. Please input 'jacobi' or 'gauss-seidel'.")

    t2 = time.time()
    print(f"Solve time: {t2-t1:.3f}s")

    solver.plot_midplane()
    solver.plot_field()