#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.animation as ani
import matplotlib as mpl
from test_class import TestClass
from spin_class import *
from hamiltonian_class import *
from thermodyn_obs import *



def test_packages():
    test_obj = TestClass()
    test_obj.test_spin1()

    test_obj.test_dip()
    test_obj.test_zfs()

    return True

def calc_prop(percentage, eigenvals, temperature):
    """Calculating the function which has <percentage> probability of occupancy at temperature T 

    Args:
        percentage (float): The threshold for the probability of occupancy. limits: (0, 1]
        eigenvals (nd.array): The eigenvalues of the system
        temperature (float): The temperature of the system
    """  
    e_shift = np.min(eigenvals)
    Z = np.sum(np.exp((e_shift - eigenvals)/temperature))
    ret = - temperature * (np.log(Z) + np.log(percentage) - e_shift/temperature)
    return ret


def animate(h, fig, axs, hamiltonian, barr, gfactor, 
            states_to_keep, cutoff, temperature, to_change, title_arr, title_unit, 
            temp_arr, mag_temps, suscept_temps, occ_label, change_lin,
            zfs_arr, idx_change):
    if to_change == 0:
        change_val = change_lin[h, idx_change]
        heis_int = change_lin[h, :]
        hamiltonian.change_heis(heis_int)
    elif to_change == 1:
        change_val = change_lin[h, idx_change]
        zfs_arr[0] = change_lin[h, :]
        hamiltonian.change_zfs(zfs_arr)
    elif to_change == 2:
        hamiltonian.change_zfs(zfs_arr)
        change_val = change_lin[h, idx_change]
        zfs_arr[2] = change_lin[h, :]
    elif to_change == 3:
        hamiltonian.change_zfs(zfs_arr)
        change_val = change_lin[h, idx_change]
        zfs_arr[3] = change_lin[h, :]
    eigvals = np.zeros((hamiltonian.dim, len(barr)))
    eigvecs = np.zeros((hamiltonian.dim, hamiltonian.dim, len(barr)), dtype=np.complex128)
    prob_cutoff = np.zeros(len(barr)*2)
    frame = build_frame(fig, axs, eigvals, eigvecs, hamiltonian, prob_cutoff, change_val, barr, gfactor, states_to_keep,
                        cutoff, temperature, to_change, title_arr, title_unit, temp_arr, mag_temps, suscept_temps, occ_label)
    return frame

def build_frame(fig, axs, eigvals, eigvecs, hamiltonian, prob_cutoff, value, barr, gfactor, 
                states_to_keep, cutoff, temperature, to_change, title_arr, title_unit, 
                temp_arr, mag_temps, suscept_temps, occ_label):
    plt.gca().clear()
    axs[0, 0].clear()
    axs[0, 1].clear()
    axs[1, 0].clear()
    axs[1, 1].clear()
    zeeman_eigvals = np.zeros((hamiltonian.dim, len(barr)*2))
    zeeman_barr = np.zeros(len(barr)*2)
    zeeman_barr = np.array([-np.flip(barr), barr])
    zeeman_barr = zeeman_barr.flatten()
    for b in range(len(barr)):
        bfield = np.array([0, 0, barr[b]])
        hamiltonian.change_zeeman(bfield)
        eigenvals, eigenvecs = np.linalg.eigh(hamiltonian.hamilmat)
        eigvals[:, b] = eigenvals
        eigvecs[:, :, b] = eigenvecs
        prob_cutoff[len(barr)-b-1] = prob_cutoff[len(barr)+b] = calc_prop(cutoff, eigenvals, temperature)
        if barr[b] == 0:
            eigvals_0b = eigenvals
        zeeman_eigvals[:, len(barr)-b-1] = zeeman_eigvals[:, len(barr)+b] = eigenvals
    for i in range(states_to_keep):
        axs[0, 0].plot(zeeman_barr, zeeman_eigvals[i, :])
    axs[0, 0].plot(zeeman_barr, prob_cutoff, 'k--', label=occ_label)

    #Entropy difference plot
    for b in range(len(barr)):
        if barr[b] >= 0.5 and np.abs(barr[b] - round(barr[b], 0)) < 0.01:
            ent_diff = np.zeros(len(temp_arr))
            for t in range(len(temp_arr)):
                ent_diff[t] = entropy_diff_from_eig(eigvals_0b, eigvals[:, b], temp_arr[t])
            axs[0, 1].plot(temp_arr, ent_diff, label="$B_z = $" + str(round(barr[b], 0)) + "T")
    #Magnetization plot
    mag = np.zeros((len(mag_temps), len(barr)))
    for b in range(len(barr)):
        mag[:, b] = magnetization_in_z(eigvecs[:, :, b], eigvals[:, b], hamiltonian, mag_temps, gfactor)
    for t in range(len(mag_temps)):
        axs[1, 0].plot(barr, mag[t], label="$T = $" + str(mag_temps[t]) + "K")

    #susceptibility plot
    suscept = np.zeros((len(suscept_temps), 2))
    for b in range(len(barr)):
        if barr[b] == 0.1:
            suscept[:, 0] = exp_suscept(eigvecs[:, :, b], eigvals[:, b], hamiltonian, suscept_temps, gfactor, barr[b])
            axs[1, 1].plot(suscept_temps, suscept[:, 0], label="$B_z = $" + str(round(barr[b], 1)) + "T")
        if barr[b] == 1:
            suscept[:, 1] = exp_suscept(eigvecs[:, :, b], eigvals[:, b], hamiltonian, suscept_temps, gfactor, barr[b])
            axs[1, 1].plot(suscept_temps, suscept[:, 1], label="$B_z = $" + str(round(barr[b], 0)) + "T")
    
    valstring = str(round(value, 2))

    axs[0, 0].set(xlabel="$\mathregular{B\ (T)}$", ylabel="$\mathregular{Energy\ (k_B\ K)}$")
    axs[0, 0].set_title(title_arr[to_change] + " = " + valstring + title_unit[to_change])
    axs[0, 0].legend()

    axs[0, 1].set(xlabel="$\mathregular{T\ (K)}$", ylabel="$\mathregular{-\Delta S\ (k_B)}$")
    axs[0, 1].set_title("Entropy difference to $B=0$")
    axs[0, 1].legend(loc='upper right')

    axs[1, 0].set(xlabel="$\mathregular{B\ (T)}$", ylabel="$\mathregular{M_z\ (\mu_B)}$")
    axs[1, 0].legend()

    axs[1, 1].set(xlabel="$\mathregular{T\ (K)}$", ylabel="$\mathregular{MT/B (\mu_B)}$")
    axs[1, 1].legend()
    frame = fig
    return frame


def init():
    #do nothing
    pass

def save_parameters(spin_system, zfs_arr, heis_arr, to_change, change_lin, descname, sys_id, distance):
    """Save the parameters for the interactive video

    Args:
        spin_system (SpinSystem): The Spin System object
        zfs_arr (nd.array): The array with the zero field couplings (D, E, theta, phi)
        heis_arr (nd.array): The Heisenberg interaction. 
        to_change (int): Identifier for the parameter which has to be changed
        change_lin (nd.array): The array with the changing values
        descname (str): The filename for the description of the parameters
        sys_id (int): The identifier for the system
        distance (float): The Characteristic length in the system for the dipolar interaction
    """    
    spinstring = ""
    for i in range(len(spin_system.spins)):
        if (spin_system.spins[i] % 2) == 0:
            spinstring += str(spin_system.spins[i]//2) + "  "
        else:
            spinstring += str(spin_system.spins[i]) + "/2  "
    sys_names = ["Spin ring", "Spin chain", "Spin Tetrahedron", "Spin Butterfly"]
    dist_names = ["Radius", "nearest Neighbor distance", "Neighbor distance", "short diagonal distance"]
    change_names=["Heisenberg interaction J", "ZFS strength D", "ZFS direction \u03D1", "ZFS direction \u03C6"]
    with open(descname, 'w') as f:
        f.write("This is the description of your gif with all the parameters you have chosen\n")
        f.write("You have: " + str(spin_system.spinnum) + " spins with quantum numbers: " + spinstring + "\n")
        f.write("The system is a " + sys_names[sys_id] + "\n")
        f.write("Hilbert space dimension: " + str(spin_system.hilbertdim) + "\n")
        if spin_system.dipolar:
            f.write("Dipolar interaction is included with a " + dist_names[sys_id] + " of: " + str(distance) + "\n")
            if sys_id == 3:
                f.write("The butterfly has a long diagonal distance of: " + str(distance*1.3) + "\n")
        else:
            f.write("No dipolar interaction\n")
        ###### Heisenberg
        f.write("Heisenberg interaction\n")
        f.write("Values: " + str(spin_system.heis_int) + "\n")
        ###### ZFS
        f.write("Zero field couplings\n")
        f.write("D: " + str(zfs_arr[0]) + "\n")
        f.write("Theta: " + str(zfs_arr[2]) + "\n")
        f.write("Phi: " + str(zfs_arr[3]) + "\n")
        f.write("Parameter to change: " + change_names[to_change] + "\n")
        f.write("Changing values\n")
        for i in range(len(change_lin)):
            f.write(str(change_lin[i]) + "\n")
    return True

def interactive_video(spin_system, lin_paras, to_change, mask, zfs_arr, heis_arr, filename, sys_id, distance):
    """Create the interactive video for the application

    Args:
        spin_system (SpinSystem): The Spin System object
        lin_paras (tuple): The parameters (start, stop, step) fot the changing value
        to_change (int): Identifier for the parameter which has to be changed
        mask (nd.array): The mask for the possibility to change just some parameters instead of all.
        zfs_arr (nd.array): The array with the zero field couplings (D, E, theta, phi)
        heis_arr (nd.array): The Heisenberg interaction. 
        filename (str): The filename for the video
        sys_id (int): The identifier for the system
        distance (float): The Characteristic length in the system for the dipolar interaction
    """    
    gifname = filename + ".gif"
    descname = filename + ".txt"
    temperature = 2.0
    gfactor = 2.0
    states_to_keep = 16
    if spin_system.hilbertdim < states_to_keep:
        states_to_keep = spin_system.hilbertdim
    hamiltonian = PosSpinHamiltonian(spin_system)

    steps = lin_paras[1] #The frames for the video
    bmax = 8.0 #Tesla
    bstep = 0.2 #Tesla
    barr = np.arange(0, bmax, bstep)

    #Create the arrays for the changing values
    idx_change = np.argmax(mask)
    if to_change == 0:
        change_lin = np.zeros((steps, len(heis_arr)))
        for i in range(len(heis_arr)):
            change_lin[:, i] = np.linspace(heis_arr[i], heis_arr[i] + lin_paras[0]*mask[i], steps)
    elif to_change == 1:
        change_lin = np.zeros((steps, spin_system.spinnum))
        for i in range(spin_system.spinnum):
            change_lin[:, i] = np.linspace(zfs_arr[0, i], zfs_arr[0, i] + lin_paras[0]*mask[i], steps)
    elif to_change == 2:
        change_lin = np.zeros((steps, spin_system.spinnum))
        for i in range(spin_system.spinnum):
            change_lin[:, i] = np.linspace(zfs_arr[2, i], zfs_arr[2, i] + lin_paras[0]*mask[i], steps)
    elif to_change == 3:
        change_lin = np.zeros((steps, spin_system.spinnum))
        for i in range(spin_system.spinnum):
            change_lin[:, i] = np.linspace(zfs_arr[3, i], zfs_arr[3, i] + lin_paras[0]*mask[i], steps)
    else:
        print("Invalid parameter to change")
        return False
    
    #Supporting arrays
    temp_arr = np.linspace(1, 15, steps)
    mag_temps = np.arange(2, 5.01, 1)#Typical temperatures for the magnetization
    suscept_temps = np.linspace(1, 30, steps*4)

    #Just to be sure that the standard values for chi -DeltaS are included
    if 0 not in barr:
        barr = np.append(barr, 0)
    if 0.1 not in barr:
        barr = np.append(barr, 0.1)  
    if 1 not in barr:
        barr = np.append(barr, 1)

    barr = np.sort(barr)

    mag_temps = np.append(mag_temps, 0.001)
    mag_temps = np.sort(mag_temps)

    mpl.rcParams['animation.embed_limit'] = 2**26
    hamiltonian.change_heis(heis_arr)
    hamiltonian.change_zfs(zfs_arr)

    #Some storage arrays
    cutoff = 0.05 #The 5% probability cutoff for 2 Kelvin
    occ_label = "$p=$" + str(round(cutoff*100)) + "$\%\ $" + "for $T=$" + str(temperature) + "K"
    eigvals_0b = np.zeros(hamiltonian.dim)
    title_arr = ["Heisenberg interaction $J_{ij}$", "ZFS strength D", "ZFS direction \u03D1", "ZFS direction \u03C6"]
    title_unit = ["K", "K", "°", "°"]

    #create the figure and axes for the gif
    fig, axs = plt.subplots(
    ncols=2,
    nrows=2,
    figsize=(10, 8),
    )
    anim = ani.FuncAnimation(plt.gcf(), animate, frames=steps, fargs=(fig, axs, hamiltonian, barr, 
                                                                      gfactor, states_to_keep, cutoff, temperature, to_change, title_arr, 
                                                                      title_unit, temp_arr, mag_temps, suscept_temps, occ_label, change_lin, zfs_arr, idx_change),
                            init_func=init, interval=100, blit=False)
    anim.writer = ani.PillowWriter()
    anim.save(gifname, writer='pillow')
    save_parameters(spin_system, zfs_arr, heis_arr, to_change, change_lin, descname, sys_id, distance)
    return True