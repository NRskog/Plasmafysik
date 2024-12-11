
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def main():
    part1()

def part1():


    N = [128, 2]
    a = 0
    b = 2* np.pi

    omega = []
    h = []

    t_end = 100
    dt = 0.01
    N_timesteps = int(t_end / dt)


    for i in range(2):
        omega.append(np.linspace(a,b, N[i], endpoint= False))
        h.append(omega[-1][1] - omega[-1][0])

    Ex_init = np.repeat(np.random.rand(N[0])[:, None], N[1], axis=1)
    Ey_init = np.repeat(np.random.rand(N[0])[:, None], N[1], axis=1)
    Bz_init = np.zeros((N[0], N[1]))
    Ex_vec, Ey_vec, Bz_vec = TE_maxwell_solver(N_timesteps, dt, h, Ex_init, Ey_init, Bz_init)

    Ex_vec = np.array(Ex_vec, dtype=float)
    Ey_vec = np.array(Ey_vec, dtype=float)
    Bz_vec = np.array(Bz_vec, dtype=float)


    Nx = N[0] # we want to plit i x-direction
    Nt = N_timesteps + 1
    dx = h[0]

    data_Ex = Ex_vec[:, :, 1]  # Elektriska fältet E_x vid andra y-punkten
    data_Ey = Ey_vec[:, :, 1]  # Elektriska fältet E_y vid andra y-punkten
    data_Bz = Bz_vec[:, :, 1]  # Magnetiska fältet B_z vid andra y-punkten



        # Create mesh for Fourier space
    kvec = 2 * np.pi * np.fft.fftfreq(Nx, dx)[:Nx//2]
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt)[:Nt//2]
    K, W = np.meshgrid(kvec, omega)

    # FFT the data
    dispersion = (2./Nt) * (2./Nx) * np.abs(np.fft.fftn(data_Bz))[:Nt//2, :Nx//2]

    # Plot the results
    plt.figure()
    plt.contourf(K, W, dispersion**2 / (dispersion**2).max(),
                 cmap='plasma', norm=colors.LogNorm(), levels=np.logspace(-15, -1, 27))
    plt.colorbar(ticks=[1e-12, 1e-9, 1e-6, 1e-3], format='%.0e')
    plt.plot(kvec, kvec, '--', label='analytical')
    plt.title("Maxwell in vacuum")
    plt.xlabel('$k$')
    plt.ylabel('omega')
    plt.legend()
    plt.xlim(0, kvec[-1])
    plt.ylim(0, kvec[-1])
    plt.show()

    #MW should be solved, here so HINT DTF TO PLOT





def TE_maxwell_solver(N_timesteps, dt, h, E_0_x, E_0_y, B_0_z):
    Ex_vec = E_0_x.copy()
    Ey_vec = E_0_y.copy()
    Bz_vec = B_0_z.copy()

    ex_list = []
    ey_list = []
    bz_list = []
    ex_list.append(Ex_vec.copy())
    ey_list.append(Ey_vec.copy())
    bz_list.append(Bz_vec.copy())
    
    #Yee
    for _ in range(N_timesteps):
        TE_update_E(dt, h, Ex_vec, Ey_vec, Bz_vec)
        TE_update_B(dt, h, Ex_vec, Ey_vec, Bz_vec)
        ex_list.append(Ex_vec.copy())
        ey_list.append(Ey_vec.copy())
        bz_list.append(Bz_vec.copy())
    
    return ex_list, ey_list, bz_list



def TE_update_B(dt,h, Ex_vec, Ey_vec, Bz_vec):
    Ex_vec += dt / h[1] * (Bz_vec - np.roll(Bz_vec, axis = 1, shift = 1) )
    Ey_vec -= dt / h[0] * (Bz_vec - np.roll(Bz_vec, axis = 0, shift = 1) )


def TE_update_E(dt, h, Ex_vec, Ey_vec, Bz_vec):
    Bz_vec -= dt * (
        (1 / h[0]) * (np.roll(Ey_vec, axis=0, shift=-1) - Ey_vec) -
        (1 / h[1]) * (np.roll(Ex_vec, axis=1, shift=-1) - Ex_vec)
    )





    
if __name__ == "__main__":
    main()





