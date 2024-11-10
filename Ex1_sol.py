import numpy as np
import matplotlib.pyplot as plt

# Parametrar
a = 0
b = 2 * np.pi
alpha = a
beta = b

def create_matrix(N, h):
    """Skapar matrisen A för Poisson-ekvationen."""
    return (1 / (h ** 2)) * (np.diag(2 * np.ones(N)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1))

def rho(x): 
    """Beräknar värdet för rho(x)."""
    return 2 * np.sin(x) + x * np.cos(x)

def Calculate_eigenvalue(matrix):
    """Beräknar egenvärden för matrisen och hittar minsta egenvärde."""
    eigenvalues = np.linalg.eigvals(matrix)
    min_eigenvalue = min(eigenvalues)
    return(min_eigenvalue)
    #print(eigenvalues, min_eigenvalue)

def solve_poisson(N, h, A):
    """Löser Poisson-ekvationen numeriskt."""
    x_interior = np.linspace(a + h, b - h, N)
    p = rho(x_interior)
    p[0] += alpha / (h ** 2)
    p[-1] += beta / (h ** 2)

    # Lösningen utan gränsvärden
    phi_sol_no_boundary = np.linalg.solve(A, p)
    phi_sol_num = np.concatenate(([alpha], phi_sol_no_boundary, [beta]))

    return phi_sol_num, x_interior

#def analytical_sol(x):
#    """Räknar ut den analytiska lösningen."""
#    return x * np.cos(x)

def calculate_error(phi_num, phi_analytical, h):
    """Beräknar fel mellan numerisk och analytisk lösning."""
    error_vector = abs(phi_analytical - phi_num)  # Vill plotta senare
   
    L1_norm = np.sum(error_vector * h)
    L2_norm = np.sqrt(h * np.sum(error_vector ** 2))
    L_inf_norm = max(error_vector)

    return L1_norm, L2_norm, L_inf_norm, error_vector

def list_of_errors(N_points):
    """Skapar listor över olika felnormer för olika värden av N."""
    L1_errors = []
    L2_errors = []
    L_inf_errors = []

    for N in N_points:
        h = (b - a) / (N + 1)
        A = create_matrix(N, h)
        phi_sol_num, x_interior = solve_poisson(N, h, A)
        x_points_cont = np.linspace(a, b, N + 2)
        phi_sol_analytical = phi_sol_analytical_lambda(x_points_cont)

        # Beräknar fel
        L1, L2, L_inf, _ = calculate_error(phi_sol_num, phi_sol_analytical, h)
        L1_errors.append(L1)
        L2_errors.append(L2)
        L_inf_errors.append(L_inf)

    return L1_errors, L2_errors, L_inf_errors

#def make_plots(numeric_sol, analytical_sol, x_interior, x_continous):  
#    """Plottar den numeriska och analytiska lösningen."""
#    plt.plot(x_interior, numeric_sol, color="red", label="Numerical solution")
#    plt.plot(x_continous, analytical_sol, color="blue", label="Analytical solution")
#    plt.legend()
#    plt.show()

def plot_error(L1, L2, L_inf, points, numeric_sol, analytical_sol, x_continous):
    """Skapar en figur med subplots för att visa felnormer och jämföra numerisk och analytisk lösning."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    ax[0].plot(points, np.c_[L1, L2, L_inf], label = ["L1", "L2", "L_inf"])
    ax[0].set_title("Error norms")
    ax[0].set_xlabel("N Points")
    ax[0].set_ylabel("Error")
 

    ax[1].plot( x_continous, np.c_[numeric_sol, analytical_sol], label = ["Numerical solution", "Analytical solution" ]  )
    ax[1].set_title("Numerical vs Analytical sol")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Solution")

    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.show()
## Tester nedan



N = 25
h = (b - a) / (N + 1)
A = create_matrix(N, h)
phi_sol_num, grid_space = solve_poisson(N, h, A)
x_points_cont = np.linspace(a, b, N + 2)
phi_sol_analytical_lambda = lambda x_points: (x_points) * np.cos(x_points)  # Används som analytisk lösning
phi_sol_analytical = phi_sol_analytical_lambda(x_points_cont)

N_values = [10, 20, 40, 80, 160, 320]
L1, L2, L_inf = list_of_errors(N_values)

print(Calculate_eigenvalue(A))
print(L1, L2, L_inf)

plot_error(L1, L2, L_inf, N_values, phi_sol_num, phi_sol_analytical, x_points_cont)
