import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from .TRG import TRG


def compute_H_exact(betas):
    """
    Compute the exact value of the Helmoltz free energy per site of the 2D Ising model
    """

    H_exact = []
    for beta in betas:

        def f(x):

            k = 2 * np.sinh(2 * beta) / (np.cosh(2 * beta) ** 2)
            return (1 / (2 * np.pi)) * np.log(
                0.5 * (1 + np.sqrt(1 - (k**2) * np.sin(x) ** 2))
            )

        a = 0
        b = np.pi

        result_quad, error = quad(f, a, b)

        H = np.log(2 * np.cosh(2 * beta)) + result_quad
        H_exact.append(H)

    return H_exact


def compute_U_exact(betas, J):
    """
    Compute the exact value of the internal energy per site of the 2D Ising model

    Parameters
    ----------
    betas : list
        List of inverse temperatures
    J : float
        Coupling constant
    """

    def g(x):
        k = 1 / (np.sinh(2 * beta) ** 2)
        return 1 / (np.sqrt(1 - 4 * k * ((1 + k) ** (-2)) * np.sin(x) ** 2))

    U_exact = []

    for beta in betas:

        a = 0
        b = np.pi / 2
        Integral, error = quad(g, a, b)

        U = (
            -J
            * (np.cosh(2 * beta * J) / np.sinh(2 * beta * J))
            * (1 + (2 / np.pi) * (-1 + 2 * np.tanh(2 * beta * J) ** 2) * Integral)
        )
        U_exact.append(U)

    return U_exact


def compute_Z_num(N, betas, truncation):
    """
    Compute the internal energy per site of the 2D Ising model using TRG

    Parameters
    ----------
    beta : float
        Inverse temperature
    J : float
        Coupling constant
    Dcut : int
        Maximum bond dimension
    no_iter : int
        Number of TRG iterations
    """

    Z_num = []

    for beta in betas:
        trg = TRG(N, beta, truncation)
        Z = trg.solve()
        Z_num.append(Z)

    return Z_num


def analysis_H(N, betas, truncation):
    """
    Analyze the Helmoltz free energy per site of the 2D Ising model
    """

    H_exact = compute_H_exact(betas)
    H_num = np.log(compute_Z_num(N, betas, truncation))

    plt.plot(betas, H_exact, label="Exact")
    plt.plot(betas, H_num, label="Numerical")
    plt.xlabel(r"$\beta J$")
    plt.ylabel(r"$-\beta F$")
    plt.title(r"Helmoltz free energy density $-\beta \, F$")
    plt.legend()
    plt.savefig("figures/helmoltz_free_energy.pdf", dpi=4000)
    plt.show()


def analysis_U(N, betas, truncation,delta):
    """
    Analyze the internal energy per site of the 2D Ising model
    """

    U_exact = compute_U_exact(betas, 1)
    U_num = -np.gradient(np.log(compute_Z_num(N, betas, truncation)),delta)

    '''plt.plot(betas, U_exact, label="Exact")
    plt.plot(betas, U_num, label="Numerical")
    plt.xlabel(r"$\beta J$")
    plt.ylabel(r"$-\beta U$")
    plt.title(r"Internal energy density $-\beta \, U$")
    plt.legend()
    plt.savefig("figures/internal_free_energy.pdf", dpi=4000)
    plt.show()'''

    
    return U_exact, U_num

# def analysis_C(N, betas, truncation,delta):
#     """
#     Analyze the specific heat of the 2D Ising model
#     """
#     kappa = 2 * N ** 2 * np.sinh(2 * betas) / (np.cosh(2 * betas) ** 2)
    
#     C_num = kappa * (betas ** 2) * np.gradient(np.gradient(np.log(compute_Z_num(N, betas, truncation)),delta))
    
#     U_exact = compute_U_exact(betas, 1)
        
#     y=-np.gradient(U_exact,delta)
#     plt.plot(betas, y,color='black', label=r'$-\beta \,f_{\infty}$')

#     plt.plot(betas, C_num, label="Numerical")
#     plt.xlabel(r"$\beta$")
#     plt.ylabel(r"$C(\beta)$")
#     # plt.title(r"Internal energy density $-\beta \, U$")
#     plt.legend()
#     #plt.savefig("figures/internal_free_energy.pdf", dpi=4000)
#     plt.show()

#     return y, C_num




############################# new derivatives 

