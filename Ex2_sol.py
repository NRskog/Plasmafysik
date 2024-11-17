import numpy as np
import matplotlib.pyplot as plt



def solve_poisson_eq_dft(rho_h, h):
    """ 
    Solving Poisson Eq. using Discrete Fourier Transform (DFT)
    
    used parameters:
    rho_h : array. RHS of equation, containing initial values (Later at least)
    h : float?. Space between grid points

    """

    # C*phi_h = rho_h <=> P *lambda *P^(*) *  phi_h = rho_h <=>phi_h = P *lambda_inverse *P^(*) * rho_h
    # P = 1/sqrt(N) * F_n_conj and F_N = ((e^(2i*pi*k*j/N)    )   )  SYM MATRIX , for 0 <= i <= N-1
    #                          and F_N_conj = ((e^(-2i*pi*k*j/N)    )   ), Be smart and just transpose above    
    ## fft.fft and even scipy does this automatically, no need to worry

    rho_hat = np.fft.fft(rho_h)
    
    #N = rho vektors längd (lösning)
    N = len(rho_h)

    # lambda^-1 has the diagonalav containing C eigenvalues inverted.
    lambda_inverse = np.zeros(N, dtype=complex)

    for k in range(N):
        #Eigenvalues from lectures lambda_o = 0, lambda_k = 4/h^2 * sin^2(k*pi/N), h = L/N, L is b - a
        if k != 0:
            lambda_k = (4 / h**2) * (np.sin((k*np.pi) / N ))**2  ##Keep N for now
            lambda_inverse[k] = 1 / lambda_k
        else:
            lambda_inverse[k] = 0  ##Should only be for k = 0, could maybe delete this row
    # We have all we need here to solve for phi_h, but solut will be in fourier space

    phi_h_hat = lambda_inverse * rho_hat
    phi_h = np.fft.ifft(phi_h_hat).real

    return(phi_h)


def calculate_error(phi_sol, phi_analytical, h):
    """Calculate error from difference between the two solutions

    """
      
    error_vector = abs(phi_analytical - phi_sol)  # Vill plotta senare
   
    L2_norm = np.sqrt(h * np.sum(error_vector ** 2))

    return L2_norm,  error_vector

def list_of_errors(N_points):
    """
    Creates a list with error from L2 norm, with different N values
    """
    L2_errors = []
    
    

    for N in N_points:
        h = (b - a) / N
        x = np.linspace(a, b, N, endpoint=False)

        phi_sol_num  = solve_poisson_eq_dft(rho_function(x), h)
        phi_analytical = phi_exact(x)
        #x_points_cont = np.linspace(a, b, N )
        #phi_sol_analytical = phi_sol_analytical_lambda(x_points_cont)

        # Beräknar fel
        L2, _ = calculate_error(phi_sol_num,   phi_analytical, h)
        L2_errors.append(L2)

    return  L2_errors

def make_plots(N_for_error, list_of_error, phi_sol, phi_exact, x_discrete, x_cont):
    """Figure for both error decay (convergence) and also analytocal vs numerical solution for poisson eq."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    
    ax[0].plot(N_for_error, list_of_error, label="L2", color="blue")
    ax[0].set_title("Error norm (L2)")
    ax[0].set_xlabel("N Points")
    ax[0].set_ylabel("Error")
    ax[0].legend()

    h_values = [(b - a) / N for N in N_for_error]
    ax[1].loglog(h_values, list_of_error, label="L2 (log-log)", color="green")
    ax[1].set_title("Error norm (L2) - loglog")
    ax[1].set_xlabel("Grid spacing h")
    ax[1].set_ylabel("Error")
    ax[1].legend()
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)


 
    ax[2].plot(x_discrete, phi_sol, color="red", label="Numerical solution", marker="o")
    ax[2].plot(x_cont, phi_exact, color="blue", label="Analytical solution")
    ax[2].set_title(f"Numerical vs Analytical solution with N = {N}")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("Solution")
    ax[2].legend()

    plt.tight_layout()
    plt.show()




#######
a = 0
b = 2*np.pi
N = 5
N_values = [10, 20, 40, 80, 160, 320, 1000]

h = (b - a)/ N
L = b - a # :/

x = np.linspace(a,b,N, endpoint= False)
x_cont = np.linspace(a,b,1000) #For analytical plot

phi_exact = lambda x_cont : np.sin(2*x_cont) + np.cos(x_cont)
phi_analyt = phi_exact(x_cont)
rho_function = lambda x_points : 4 * np.sin(2*x_points) + np.cos(x_points)
phi_sol = solve_poisson_eq_dft(rho_function(x), h)
#print(solve_poisson_eq_dft(given_rho(x), h))

L2_list = list_of_errors(N_values)
print(L2_list[3] / L2_list[4], "\n", L2_list) ## around 1/4 or 4

make_plots(N_values, L2_list, phi_sol, phi_analyt, x, x_cont)


