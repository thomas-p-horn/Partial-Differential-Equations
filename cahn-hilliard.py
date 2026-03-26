import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

running = True # This is to handle exiting the code when the figure is closed
def on_close(event):
    global running
    running = False

    
def laplacian(lattice, dx):
    return (np.roll(lattice, -1, axis=0) + np.roll(lattice,  1, axis=0) + np.roll(lattice, -1, axis=1) + np.roll(lattice,  1, axis=1) - 4*lattice) / dx**2

def grad_2(lattice):
    dx = (np.roll(lattice, 1, 0) - np.roll(lattice, -1, 0)) / 2
    dy = (np.roll(lattice, 1, 1) - np.roll(lattice, -1, 1)) / 2
    return dx**2 + dy**2




class cahn_hilliard:
    def __init__(self, L, a, k, M, dt, dx, phi0, sig_phi0, tolerance):
        self.L = L
        self.a = a
        self.k = k
        self.M = M
        self.dt = dt
        self.dx = dx
        self.phi0 = phi0
        self.tolerance = tolerance
        
        self.phi = np.random.normal(phi0, sig_phi0, (L, L))

    def cahn_hilliard_step(self):
        mu = -self.a * self.phi * (1 - self.phi**2) - self.k * laplacian(self.phi, self.dx)
        self.phi = self.phi + self.M * self.dt * laplacian(mu, self.dx)

    def animate(self):

        plt.ion()
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("close_event", on_close) # Handles closing figure
        img = ax.imshow(self.phi, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_title("Cahn-Hilliard")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        step = 0
        while running:
            step += 1
            self.cahn_hilliard_step()
            if step % 100 == 0:
                img.set_data(self.phi)
                plt.pause(0.001)
        plt.ioff()


    def free_energy(self):
        return -(self.a/2) * self.phi**2 + (self.a/4) * self.phi**4 + (self.k/2) * grad_2(self.phi)



    def plot_free_energy(self):
        df = pd.read_csv(f'cahn-hilliard data/Free Energy {self.phi0}.csv')
        t_vals = df['Time']
        F = df['Free Energy']

        plt.plot(t_vals, F, c='darkslateblue')
        plt.title(rf'$F(t) ~ ~ ~\phi_0 = {self.phi0}$')
        plt.xlabel('Time step')
        plt.ylabel('Total Free Energy')
        plt.savefig(f'cahn-hilliard data/Free Energy {self.phi0}.png', dpi=300, bbox_inches='tight')


    def record_free_energy(self):

        F = []
        t_vals = []

        for t in range(1, 500_000):
            self.cahn_hilliard_step()
            f = np.sum(self.free_energy()) / self.dx**2
            F.append(f)
            t_vals.append(t)

            if (t % 500) == 0:
                std = np.std(F[-500:])
                print(std)
                if std <= self.tolerance:
                    print(f'The total free energy converged for t={t}')
                    break

        df = pd.DataFrame({
            "Time": t_vals,
            "Free Energy": F
        })

        df.to_csv(f'cahn-hilliard data/Free Energy {self.phi0}.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Runs the Cahn-Hilliard equation, modelling phase separation.")
    parser.add_argument("--action", help="What to do with model: 'animate' or 'measure' (calculate free energy) or 'draw' (plot free energy, requires .csv files from 'measure'). Default='animate'", type=str, default='animate')
    parser.add_argument("-l", "--length", help="LxL size of lattice. Default=100", type=int, default=100)
    parser.add_argument("-a", help="Default=2.0", type=float, default=2.)
    parser.add_argument("-k", help="Default=1.0", type=float, default=1.)
    parser.add_argument("-M", help="Default=2.0", type=float, default=2.)
    parser.add_argument("-dt", help="Default=2e-4", type=float, default=2e-4)
    parser.add_argument("-dx", help="Default=1.0", type=float, default=1.)
    parser.add_argument("-p0", "--phi0", help="Mean value of initialisation. Default=0.0", type=float, default=0.)
    parser.add_argument("-sp0", "--sigma_phi0", help="Distribution width of initialisation. Default=0.5", type=float, default=0.5)
    parser.add_argument("-t", "--tolerance", help="Standard deviation of 500 iterations of the system below which the free energy is found to have converged. Default=0.1", type=float, default=0.5)

    args = parser.parse_args()
    action = args.action

    diffusion = cahn_hilliard(
        args.length,
        args.a,
        args.k,
        args.M,
        args.dt,
        args.dx,
        args.phi0,
        args.sigma_phi0,
        args.tolerance
    )


    if action == 'animate':
        diffusion.animate()

    elif action == 'measure':
        diffusion.record_free_energy()
        diffusion.plot_free_energy()

    elif action == 'draw':
        diffusion.plot_free_energy()


    else:
        print("--action not recognised")
        quit()