#!/usr/bin/env python

import copy
import numpy as np

muB_kb = 0.6717141002 # muB/kb in Kelvin


#Spin Hamiltonian containing the Heisenberg interaction and the Zeeman interaction 
class SpinHamiltonian():
    """The class for the Hamiltonian of a spin system. It builds a hamiltonmatrix to store interaction terms.
    """    
    def __init__(self, spin_sys):
        """Constructor

        Args:
            spin_sys (SpinSystem): Instance of the spinSystem class containing spin quantum numbers, hilbert space dimension and interaction matrices
        """        
        self.spin_sys = spin_sys
        self.dim = spin_sys.hilbertdim
        self.hamilmat = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.spindim = self.mk_spindim()
        self.magfield = np.zeros(3)
        self.vecmap = self.mk_vecmap()
        if self.spin_sys.heis_int != None and self.spin_sys.heis_int != {}:
            for sh in self.spin_sys.heis_int:
                self.heis_entry_vectorized(sh, self.spin_sys.heis_int[sh], self.vecmap)
        if not np.allclose(self.spin_sys.zfs, np.zeros((4, self.spin_sys.spinnum))):
            for sh in range(self.spin_sys.spinnum):
                veca = self.num_to_vec(np.arange(self.dim))
                self.zfs_vectorized(sh, self.spin_sys.zfs_mats, veca)
    
    def is_hermitian(self):
        """Tests whether the Hamiltonian is Hermitian
        """
        if np.allclose(self.hamilmat, np.conj(self.hamilmat.T)):
            print("The Hamiltonian is Hermitian")
            return True
        else:
            print("The Hamiltonian is not Hermitian")
            return False

    def change_heis(self, new_heis):
        """Changes the Heisenberg interaction terms in the Hamiltonian matrix

        Args:
            new_heis (nd.array): The new Heisenberg interaction terms as an array of the length of the number of interactions
        """        
        heis_diff = {sh: new_heis[i]- self.spin_sys.heis_int[sh] for i, sh in enumerate(self.spin_sys.heis_int)}
        if not np.allclose(list(heis_diff.values()), np.zeros(len(heis_diff))):
            self.spin_sys.update_heis(new_heis)
            for i, sh in enumerate(self.spin_sys.heis_int):
                self.heis_entry_vectorized(sh, heis_diff[sh], self.vecmap)

    def heis_entry_vectorized(self, sh, heis, veca):
        h = np.arange(self.dim)
        self.hamilmat[h, h] += heis * self.spin_sys.s_z_vect(sh[0], veca) * self.spin_sys.s_z_vect(sh[1], veca)
        mask = (veca[:,sh[0]] != 0) & (veca[:,sh[1]] != self.spin_sys.spins[sh[1]])
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_plus_vect(sh[0], veca[mask]) * self.spin_sys.s_minus_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] -= 1
        vecb[mask, sh[1]] += 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += 0.5 * tmpfac * heis
        mask = (veca[:, sh[0]] != self.spin_sys.spins[sh[0]]) & (veca[:, sh[1]] != 0)
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_minus_vect(sh[0], veca[mask]) * self.spin_sys.s_plus_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] += 1
        vecb[mask, sh[1]] -= 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += 0.5 * tmpfac * heis


    #implementation of the zero field splitting terms which are quadratic in the spin operators
    def change_zfs(self, new_zfs):
        """Changes the zero field splitting terms in the Hamiltonian matrix 

        Args:
            new_zfs (nd.array): The new zero field splitting terms in the form of (spinnum, 4) where the 4 columns are the (d, e, theta, phi) values
        """ 
        para_zfs = np.copy(new_zfs)
        for sh in range(self.spin_sys.spinnum):
            veca = self.num_to_vec(np.arange(self.dim))
            self.zfs_vectorized(sh, -self.spin_sys.zfs_mats, veca) 
        self.spin_sys.update_zfs(para_zfs)
        for sh in range(self.spin_sys.spinnum):
            veca = self.num_to_vec(np.arange(self.dim))
            self.zfs_vectorized(sh, self.spin_sys.zfs_mats, veca)

    def zfs_vectorized(self, sh, zfs_mats, veca):
        h = np.arange(self.dim)
        self.hamilmat[h, h] += zfs_mats[sh, 2, 2] * self.spin_sys.s_z_vect(sh, veca) ** 2
        #++ term
        mask = (veca[:,sh] > 1)
        vecb = veca.copy()
        vecb[mask, sh] -= 1
        tmp = self.spin_sys.s_plus_vect(sh, veca[mask]) * self.spin_sys.s_plus_vect(sh, vecb[mask])
        vecb[mask, sh] -= 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += zfs_mats[sh, 0, 0] * tmp
        #-- term
        mask = (veca[:,sh] < self.spin_sys.spins[sh] - 1)
        vecb = veca.copy()
        vecb[mask, sh] += 1
        tmp = self.spin_sys.s_minus_vect(sh, veca[mask]) * self.spin_sys.s_minus_vect(sh, vecb[mask])
        vecb[mask, sh] += 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmp * zfs_mats[sh, 1, 1]
        #z+ and +z terms if s_plus and s_z are both executable otherwise the commutator has to be used 
        mask = (veca[:,sh] > 0)
        vecb = veca.copy()
        vecb[mask, sh] -= 1
        tmp = self.spin_sys.s_plus_vect(sh, veca[mask]) * self.spin_sys.s_z_vect(sh, vecb[mask])
        tmp += self.spin_sys.s_z_vect(sh, veca[mask]) * self.spin_sys.s_plus_vect(sh, veca[mask])
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmp * zfs_mats[sh, 0, 2]
        #Commutator if s_plus is not executable
        mask = (veca[:,sh] <= 0)
        vecb = veca.copy()
        vecb[mask, sh] += 1
        tmp = 2 * self.spin_sys.s_minus_vect(sh, veca[mask]) * self.spin_sys.s_plus_vect(sh, vecb[mask])
        tmp -= 2 * self.spin_sys.s_z_vect(sh, veca[mask])
        self.hamilmat[h[mask], h[mask]] += tmp * zfs_mats[sh, 0, 1]
        #z- and -z terms if s_minus and s_z are both executable otherwise the commutator has to be used
        mask = (veca[:,sh] < self.spin_sys.spins[sh])
        vecb = veca.copy()
        vecb[mask, sh] += 1
        tmp = self.spin_sys.s_minus_vect(sh, veca[mask]) * self.spin_sys.s_z_vect(sh, vecb[mask])
        tmp += self.spin_sys.s_z_vect(sh, veca[mask]) * self.spin_sys.s_minus_vect(sh, veca[mask])
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmp * zfs_mats[sh, 1, 2]
        #Commutator if s_minus is not executable
        mask = (veca[:,sh] >= self.spin_sys.spins[sh])
        vecb = veca.copy()
        vecb[mask, sh] -= 1
        tmp = 2 * self.spin_sys.s_plus_vect(sh, veca[mask]) * self.spin_sys.s_minus_vect(sh, vecb[mask])
        tmp += 2 * self.spin_sys.s_z_vect(sh, veca[mask])
        self.hamilmat[h[mask], h[mask]] += tmp * zfs_mats[sh, 0, 1]
        #+- term if s_plus and s_minus are both executable
        mask = (veca[:,sh] > 0) & (veca[:,sh] < self.spin_sys.spins[sh])
        vecb = veca.copy()
        vecb[mask, sh] -= 1
        tmp = self.spin_sys.s_plus_vect(sh, veca[mask]) * self.spin_sys.s_minus_vect(sh, vecb[mask])
        vecb[mask, sh] += 2
        tmp += self.spin_sys.s_minus_vect(sh, veca[mask]) * self.spin_sys.s_plus_vect(sh, vecb[mask])
        self.hamilmat[h[mask], h[mask]] += tmp * zfs_mats[sh, 0, 1]
     

    def add_zeeman(self, magfield):
        """Adds the Zeeman term (magnetic field interaction) to the Hamiltonian matrix

        Args:
            magfield (nd.array): 3D vector of the magnetic field
        """      
        self.magfield = magfield
        for sh in range(self.spin_sys.spinnum):
            veca = self.num_to_vec(np.arange(self.dim))
            self.zeeman_entry_vectorized(sh, self.magfield, veca)

    def change_zeeman(self, new_magfield):
        """Changes the Zeeman term (magnetic field interaction) of the Hamiltonian matrix

        Args:
            magfield (nd.array): 3D vector of the magnetic field
        """ 
        magfield_diff = new_magfield - self.magfield
        if not np.allclose(magfield_diff, np.zeros(3)):
            self.magfield = new_magfield
            for sh in range(self.spin_sys.spinnum):
                veca = self.num_to_vec(np.arange(self.dim))
                self.zeeman_entry_vectorized(sh, magfield_diff, veca)

    def zeeman_entry_vectorized(self, sh, magfield, veca):
        h = np.arange(self.dim)
        self.hamilmat[h, h] += magfield[2] * self.spin_sys.s_z_vect(sh, veca) * muB_kb * self.spin_sys.gfactor
        mask = (veca[:,sh] != 0)
        vecb = veca.copy()
        vecb[mask, sh] -= 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += 0.5 * self.spin_sys.s_plus_vect(sh, veca[mask]) * muB_kb * self.spin_sys.gfactor * (magfield[0] - magfield[1] * 1j)
        mask = (veca[:,sh] != self.spin_sys.spins[sh])
        vecb = veca.copy()
        vecb[mask, sh] += 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += 0.5 * self.spin_sys.s_minus_vect(sh, veca[mask]) * muB_kb * self.spin_sys.gfactor * (magfield[0] + magfield[1] * 1j)
    
    def vec_to_num(self, vec):
        """Converts an array of vectors of the form [a1, a2, a3, ...] to a vector of numbers of the form a1 * dim1 + a2 * dim2 + a3 * dim3 + ...

        Args:
            vec (nd.array): vector of the form [[a1, a2, a3, ...], [...], ...]

        Returns:
            nd.array: entries in the hilbert space
        """
        num = np.zeros(vec.shape[0], dtype=int)
        num = np.sum(vec * self.spindim, axis=1)
        return num
    
    def num_to_vec(self, num):
        """Converts a vector of numbers of the form a1 * dim1 + a2 * dim2 + a3 * dim3 + ... to a 2d array of the form [[a1, a2, a3, ...], [...], ...]

        Args:
            num (1d.array): entries in the hilbert space

        Returns:
            2d.array: 2d-array of vectors with sz values for the spins
        """  
        return self.vecmap[num] 
     
    def mk_vecmap(self):
        """Creates an array that maps each number in the hilbert space to a vector of sz values for the spins

        Returns:
            np.array: vector of the sz values for the spins for each number in the hilbert space
        """        
        vecmap = np.zeros((self.dim, self.spin_sys.spinnum), dtype=int)
        for i in range(1, self.dim):
            vecmap[i, 0] = i % (self.spin_sys.spins[0]+1)
            for j in range(1, self.spin_sys.spinnum):
                vecmap[i, j] = vecmap[i-1, j] + (not np.any(vecmap[i, 0:j]))
                vecmap[i, j] = vecmap[i, j] % (self.spin_sys.spins[j]+1)
        return vecmap
    
    def mk_spindim(self):
        """Creates an array containing the dimensions of the subspaces of the hilbert space
        """        
        spindim = np.ones(len(self.spin_sys.spins), dtype=int)
        for i in range(1, len(spindim)):
            spindim[i] = (self.spin_sys.spins[i-1]+1) * spindim[i-1]
        return spindim



#Class to include the dipolar interaction in the Hamiltonian of a spin system. 
# Therefore the positions of the spins are needed and will be transformed to receive the contributions of the dipolar interaction

class PosSpinHamiltonian(SpinHamiltonian):
    """Class for the Hamiltonian of a spin system with dipolar interaction. It builds a hamiltonmatrix to store interaction terms.
    """
    def __init__(self, spin_sys):
        """Constructor

        Args:
            spin_sys (SpinSystem): Instance of the spin_sys class containing spin quantum numbers, hilbert space dimension and interaction matrices
        """        
        super().__init__(spin_sys)
        if self.spin_sys.dipolar:
            self.add_dipolar()

    def add_dipolar(self):
        """Adds the dipolar interaction to the Hamiltonian matrix
        """
        for sh in self.spin_sys.dip_ints:
            self.quad_int_vectorized(sh, self.spin_sys.dip_ints[sh], self.vecmap)
    
    def delete_dipolar(self):
        """Deletes the dipolar interaction from the Hamiltonian matrix
        """
        for sh in self.spin_sys.dip_ints:
            self.quad_int_vectorized(sh, -self.spin_sys.dip_ints[sh], self.vecmap) 

    def quad_int_vectorized(self, sh, int_mat, veca):
        """Calculates the contribution of a quadratic interaction term between two spins to the Hamiltonian matrix

        Args:
            spin_sys (SpinSystem): Instance of the spin_sys class containing spin quantum numbers, hilbert space dimension and interaction matrices
            int_mat (np.array): TA general 3x3 interaction matrix
            veca (np.array): The current state in the porduct basis of the Hilbert space
            h (int): The current entry in the Hamiltonian matrix
        """ 
        h = np.arange(self.dim)
        #zz term
        self.hamilmat[h, h] += int_mat[2, 2] * self.spin_sys.s_z_vect(sh[0], veca) * self.spin_sys.s_z_vect(sh[1], veca)
        #++ term
        mask = (veca[:, sh[0]] != 0) & (veca[:, sh[1]] != 0)
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_plus_vect(sh[0], veca[mask]) * self.spin_sys.s_plus_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] -= 1
        vecb[mask, sh[1]] -= 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[0, 0]
        #+- term
        mask = (veca[:, sh[0]] != 0) & (veca[:, sh[1]] != self.spin_sys.spins[sh[1]])
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_plus_vect(sh[0], veca[mask]) * self.spin_sys.s_minus_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] -= 1
        vecb[mask, sh[1]] += 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[0, 1]
        #+z term
        mask = (veca[:, sh[0]] != 0)
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_plus_vect(sh[0], veca[mask]) * self.spin_sys.s_z_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] -= 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[0, 2]
        #-+ term
        mask = (veca[:, sh[0]] != self.spin_sys.spins[sh[0]]) & (veca[:, sh[1]] != 0)
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_minus_vect(sh[0], veca[mask]) * self.spin_sys.s_plus_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] += 1
        vecb[mask, sh[1]] -= 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[1, 0]
        #-- term
        mask = (veca[:, sh[0]] != self.spin_sys.spins[sh[0]]) & (veca[:, sh[1]] != self.spin_sys.spins[sh[1]])
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_minus_vect(sh[0], veca[mask]) * self.spin_sys.s_minus_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] += 1
        vecb[mask, sh[1]] += 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[1, 1]
        #-z term
        mask = (veca[:, sh[0]] != self.spin_sys.spins[sh[0]])
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_minus_vect(sh[0], veca[mask]) * self.spin_sys.s_z_vect(sh[1], veca[mask])
        vecb[mask, sh[0]] += 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[1, 2]
        #z+ term
        mask = (veca[:, sh[1]] != 0)
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_z_vect(sh[0], veca[mask]) * self.spin_sys.s_plus_vect(sh[1], veca[mask])
        vecb[mask, sh[1]] -= 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[2, 0]
        #z- term
        mask = (veca[:, sh[1]] != self.spin_sys.spins[sh[1]])
        vecb = veca.copy()
        tmpfac = self.spin_sys.s_z_vect(sh[0], veca[mask]) * self.spin_sys.s_minus_vect(sh[1], veca[mask])
        vecb[mask, sh[1]] += 1
        self.hamilmat[h[mask], self.vec_to_num(vecb[mask])] += tmpfac * int_mat[2, 1]


    