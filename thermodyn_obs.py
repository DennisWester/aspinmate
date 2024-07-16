#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from spin_class import *
from hamiltonian_class import *

#Calculates the entropy of the spin system at a given temperature and magnetic field
def entropy(hamiltonian, T, B=[0, 0, 0]):
    """Calculates the entropy of the spin system at a given temperature and magnetic field

    Args:
        hamiltonian (SpinHamiltonian): An instance of the SpinHamiltonian class. The Hamiltonian of the system
        T (float): Temperature in Kelvin
        B (list, optional): Magnetic field. Defaults to [0, 0, 0].
    """
    hamiltonian.change_zeeman(B)
    eigenvals = np.linalg.eigvalsh(hamiltonian.hamilmat)
    e_shift = eigenvals[0]
    Z = np.sum(np.exp((e_shift - eigenvals)/T))
    expectH = - np.sum(eigenvals * np.exp((e_shift - eigenvals)/T)) / Z
    return (np.log(Z) - e_shift/T) - expectH / T

def entropy_gibbs(hamiltonian, T, B=[0, 0, 0]):
    """Calculates the entropy of the spin system at a given temperature and magnetic field via the Gibbs enthalpy

    Args:
        hamiltonian (SpinHamiltonian): An instance of the SpinHamiltonian class. The Hamiltonian of the system
        T (float): Temperature in Kelvin
        B (list, optional): Magnetic field. Defaults to [0, 0, 0].
    """
    hamiltonian.change_zeeman(B)
    eigenvals = np.linalg.eigvalsh(hamiltonian.hamilmat)
    e_shift = min(eigenvals)
    Tlow = T - 1e-3
    Thigh = T + 1e-3
    Z_low = np.sum(np.exp(-(eigenvals - e_shift)/Tlow))
    Z_high = np.sum(np.exp(-(eigenvals - e_shift)/Thigh))
    gibbs_low = - Tlow * np.log(Z_low)
    gibbs_high = - Thigh * np.log(Z_high)
    return - (gibbs_high - gibbs_low) / (Thigh - Tlow)

def entropy_diff(hamiltonian, mag_diff, T):
    """Calculates the negative difference (-ΔS) in entropy between mag_diff and 0 in z-direction

    Args:
        hamiltonian (SpinHamiltonian): Hamiltonian of the system. Instance of the SpinHamiltonian class
        mag_diff (float): The magnetic field at which the entropy is calculated and 0
        T (float): Temperature in Kelvin

    Returns:
        float: the entropy difference between mag_diff and 0 in z-direction at temperature T for the given spin system
    """
    return entropy(hamiltonian, T) - entropy(hamiltonian, T, mag_diff)

def analytic_test_entropy_s1_2(T, B):
    """Calculates the entropy of the spin system at a given temperature and magnetic field via the analytic formula

    Args:
        T (float): Temperature in Kelvin
        B (list, optional): Magnetic field. Defaults to [0, 0, 0].
    """
    muB_kB = 0.6717141002
    first_term = muB_kB * B[2]/T * np.tanh(muB_kB * B[2]/T)
    second_term = np.log(2*np.cosh(muB_kB * B[2]/T))
    return -first_term + second_term

def calc_z_expects(eigvecs, ham):
    """Calculating the spin expectation values in z-direction of the system

    Args:
        eigvecs (nd.array): The eigenvectors of the system aranged in a 2d array
    """    
    expects = np.zeros(ham.dim)
    h = np.arange(ham.dim)
    for sh in range(ham.spin_sys.spinnum):
        veca = ham.num_to_vec(h)
        for d in range(ham.dim):
            expects[d] += np.sum(ham.spin_sys.s_z_vect(sh, veca) * np.abs(eigvecs[h,d])**2)
    return expects


    
def magnetization_in_z(eigvecs, eigvals, hamiltonian, temperatures, gfactor):
    """Calculating the magnetization in z direction of the system

    Args:
        eigvecs (nd.array): The eigenvectors of the system
        eigvals (nd.array): The eigenvalues of the system
        temperatures (nd.array): The temperatures of the system
        gfactor (float): The g-factor of the system
    """ 
    sz_expects = calc_z_expects(eigvecs, hamiltonian)
    e_shift = eigvals[0]
    mag = np.zeros(len(temperatures))
    for tidx, t in enumerate(temperatures):
        Z = np.sum(np.exp((e_shift - eigvals)/t))
        mag[tidx] = np.sum(sz_expects * np.exp((e_shift - eigvals)/t))
        mag[tidx] *= - gfactor / Z

    return mag

def exp_suscept(eigvecs, eigvals, hamiltonian, temperatures, gfactor, mag_field):
    """Calculating the magnetic susceptibility of the system

    Args:
        eigvecs (nd.array): The eigenvectors of the system
        eigvals (nd.array): The eigenvalues of the system
        temperatures (nd.array): The temperatures of the system
        gfactor (float): The g-factor of the system
        mag_fields (float): The magnetic fields of the system
    """    
    suscept = np.zeros(len(temperatures))
    suscept = magnetization_in_z(eigvecs, eigvals, hamiltonian, temperatures, gfactor)
    suscept *= temperatures / mag_field
    # suscept *= 0.558494 #conversion factor from mu_B to cm³mol⁻¹K
    return suscept


def entropy_from_eig(eigenvals, T):
    e_shift = eigenvals[0]
    Z = np.sum(np.exp((e_shift - eigenvals)/T))
    expectH = - np.sum(eigenvals * np.exp((e_shift - eigenvals)/T)) / Z
    return (np.log(Z) - e_shift/T) - expectH / T

def entropy_diff_from_eig(eigenvals0, eigenvals1, T):
    return entropy_from_eig(eigenvals0, T) - entropy_from_eig(eigenvals1, T)