#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np

muB_kb = 0.6717141002 # muB/kb in Kelvin

#SpinSystem class in which the positions and interaction of a spin system are defined
class SpinSystem:
    """Class for a spin system with a given number of spins and a given interaction matrix
       It contains the spin quantum numbers, the number of spins, the heisenberg interaction matrix and whether dipolar interactions are included or not
       If dipolar is included the positions of the spins are also part of this class otherwise they are set to NONE
    """        
    def __init__(self, spins, gfactor, heis_int = None, zfs = None, radians = False):
        """

        Args:
            spins (nd.array): the spin quantum numbers of the spins in the system as integers (therefore DOUBLED)
            gfactor (float): g-factor of the spins
            heis_int (dict): strengths of the Heisenberg interaction between the spins. 
            zfs (nd.array): zero field splitting parameters for the spins shape: (4, num_spins) where the columns are the D, E, theta and phi
            radians (bool): whether the angles in the zfs matrix are in radians or degrees
        """    
        #check whether spins is only integer values and not float (doubled spin quantum numbers)
        if not np.all(np.equal(np.mod(spins, 1), 0)):
            raise ValueError("Spin quantum numbers must be integer values")
        self.spins = spins
        self.spinnum = len(spins)
        if heis_int is None:
            self.heis_int = {}
        else:
            self.heis_int = heis_int
        self.hilbertdim = self.calc_hilbertdim()
        self.gfactor = gfactor
        if zfs is not None:
            self.zfs = np.copy(zfs)
            if self.zfs.shape != (4, self.spinnum):
                raise ValueError("ZFS matrix has wrong shape")
        else:
            self.zfs = np.zeros((4, self.spinnum))
        if radians == False:
            self.zfs[2:4] = np.deg2rad(self.zfs[2:4])
        self.zfs_xyzmats = np.zeros_like((self.spinnum, 3, 3))
        self.zfs_mats = self.create_zfs_mats()
        


    def calc_hilbertdim(self):
        """Calculates the hilbert space dimension of the system

        Returns:
            int: hilbert space dimension
        """        
        hilbertdim = 1
        for i in range(self.spinnum):
            hilbertdim *= (self.spins[i]+1)
        return hilbertdim
    
    def s_plus(self, i, ai):
        """Defines the action of the s+ operator on a spin

        Args:
            i (int): spin index of the DOUBLED spin quantum number
            ai (int): actual state of the spin

        Returns:
            float: action of the s+ operator on the spin
        """        
        spin = 0.5 * self.spins[i]
        mi = spin - ai
        ret = np.sqrt(spin * (spin + 1.0) - mi * (mi + 1.0))
        return ret

    def s_minus(self, i, ai):
        """Defines the action of the s- operator on a spin

        Args:
            i (int): spin index of the DOUBLED spin quantum number
            ai (int): actual state of the spin

        Returns:
            float: action of the s- operator on the spin
        """ 
        spin = 0.5 * self.spins[i]
        mi = spin - ai
        ret = np.sqrt(spin * (spin + 1.0) - mi * (mi - 1.0))
        return ret
    
    def s_z(self, i, ai):
        """Defines the action of the sz operator on a spin

        Args:
            i (int): spin index of the DOUBLED spin quantum number
            ai (int): actual state of the spin

        Returns:
            float: action of the sz operator on the spin
        """
        spin = 0.5 * self.spins[i]
        mi = spin - ai
        return mi
    
    def s_z_vect(self, sh, veca):
        # Assuming veca is a 2D array where each row is a vector
        # and sh is the index of the spin
        return np.array([self.spins[sh]/2 - veca_i[sh] for veca_i in veca])

    def s_plus_vect(self, sh, veca):
        # Assuming veca is a 2D array where each row is a vector
        # and sh is an array of the same length as the number of rows in veca
        return np.array([np.sqrt(veca_i[sh] * (self.spins[sh]-veca_i[sh]+1)) for veca_i in veca])

    def s_minus_vect(self, sh, veca):
        # Assuming veca is a 2D array where each row is a vector
        # and sh is an array of the same length as the number of rows in veca
        return np.array([np.sqrt((self.spins[sh]-veca_i[sh]) * (veca_i[sh]+1)) for veca_i in veca])
    
    def update_heis(self, heis_vec):
        """Changes the Heisenberg interaction matrix

        Args:
            heis_vec (np.array): new Heisenberg interaction matrix
        """        
        self.heis_int = {sh: heis_vec[i] for i, sh in enumerate(self.heis_int)}
   
    def xyz_in_updownz(self, mat):
        """Transforms a matrix from cartesian coordinates to the up-down-z basis

        Args:
            mat (np.array): matrix in cartesian coordinates
        """
        mat = np.reshape(mat, (9))
        pmzmat = np.zeros((3, 3), dtype=np.complex128)  
        pmzmat[0, 0] = 0.25 * (mat[0] - mat[4]) - 0.25 * (mat[1] + mat[3]) * 1j  # ++
        pmzmat[0, 1] = 0.25 * (mat[0] + mat[4]) + 0.25 * (mat[1] - mat[3]) * 1j  # +-
        pmzmat[0, 2] = 0.5 * mat[2] - 0.5 * mat[5] * 1j  # +z
        pmzmat[1, 0] = 0.25 * (mat[0] + mat[4]) - 0.25 * (mat[1] - mat[3]) * 1j  # -+
        pmzmat[1, 1] = 0.25 * (mat[0] - mat[4]) + 0.25 * (mat[1] + mat[3]) * 1j  # --
        pmzmat[1, 2] = 0.5 * mat[2] + 0.5 * mat[5] * 1j  # -z
        pmzmat[2, 0] = 0.5 * mat[6] - 0.5 * mat[7] * 1j  # z+
        pmzmat[2, 1] = 0.5 * mat[6] + 0.5 * mat[7] * 1j  # z-
        pmzmat[2, 2] = mat[8]
        return pmzmat  
    
    def cartesian_to_spherical(self, vec):
        """Transforms a vector from cartesian to spherical coordinates

        Args:
            vec (np.array): vector in cartesian coordinates

        Returns:
            np.array: vector in spherical coordinates (angles in radians)
        """        
        r = np.linalg.norm(vec)
        theta = np.arccos(vec[2]/r)
        phi = np.arctan2(vec[1], vec[0])
        return np.array([r, theta, phi])
     
    def spherical_to_cartesian(self, vec):
        """Transforms a vector from spherical to cartesian coordinates

        Args:
            vec (np.array): vector in spherical coordinates (angles in radians)

        Returns:
            np.array: vector in cartesian coordinates
        """        
        x = vec[0] * np.sin(vec[1]) * np.cos(vec[2])
        y = vec[0] * np.sin(vec[1]) * np.sin(vec[2])
        z = vec[0] * np.cos(vec[1])
        return np.array([x, y, z])
    
    def update_zfs(self, zfs, radians = False):
        """Changes the zero field splitting parameters

        Args:
            zfs (np.array): new zero field splitting parameters
        """      
        if radians == False:
            zfs[2:4] = np.deg2rad(zfs[2:4])  
        self.zfs = np.copy(zfs)
        self.zfs_mats = self.create_zfs_mats()
    
    def create_zfs_mats(self):
        """Creates the zero field splitting matrices from the zero field splitting parameters

        Returns:
            np.array: The zfs matrices for all spins
        """        
        zfs_mats = np.zeros((self.spinnum, 3, 3))
        zfs_pmz_mats = np.zeros((self.spinnum, 3, 3), dtype=np.complex128)
        for i in range(self.spinnum):
            zfs_mats[i] = self.create_zfs_mat(self.zfs[0, i], self.zfs[1,i], self.zfs[2, i], self.zfs[3, i])
            zfs_pmz_mats[i] = self.xyz_in_updownz(zfs_mats[i])  
        self.zfs_xyzmats = zfs_mats
        return zfs_pmz_mats

    def create_zfs_mat(self, d_scalar, e_scalar, theta, phi):
        """Creates the zero field splitting matrices from the zero field splitting parameters

        Args:
            d_scalar (nd.array): The D parameter of the zero field splitting
            e_scalar (nd.array): The E parameter of the zero field splitting
            theta (nd.array): The theta angle of the zero field splitting
            phi (nd.array): The phi angle of the zero field splitting

        Returns:
            nd.array: The zfs matrices for all spins
        """        
        zfs_mat = np.zeros((3, 3))
        
        zfs_mat[0, 0] = d_scalar * np.sin(theta)**2 * np.cos(phi)**2 + e_scalar * np.cos(theta)**2 * np.cos(2 * phi)  # xx
        zfs_mat[0, 1] = d_scalar * np.sin(theta)**2 * np.cos(phi) * np.sin(phi) + e_scalar * (np.cos(theta)**2 + 1) * np.cos(phi) * np.sin(phi)  # xy and yx
        zfs_mat[0, 2] = (d_scalar - e_scalar) * np.sin(theta) * np.cos(theta) * np.cos(phi)  # xz
        zfs_mat[1, 1] = d_scalar * np.sin(theta)**2 * np.sin(phi)**2 - e_scalar * np.cos(theta)**2 * np.cos(2 * phi)  # yy
        zfs_mat[1, 2] = (d_scalar - e_scalar) * np.sin(theta) * np.cos(theta) * np.sin(phi)  # yz
        zfs_mat[2, 2] = d_scalar * np.cos(theta)**2 + e_scalar * np.sin(theta)**2  # zz
        # Setting values close to zero as 0
        zfs_mat[np.abs(zfs_mat) <= 1E-13] = 0
        zfs_mat = zfs_mat + zfs_mat.T - np.diag(zfs_mat.diagonal())
        return zfs_mat
    
class AnisospinSystem(SpinSystem):
    def __init__(self, spins, gfactor, heis_int=None, dipolar=True, zfs=None, radians=False):
        """Class for a spin system with some anisotropic interactions

        Args:
            spins (np.array): the spin quantum numbers of the spins in the system as integers (therefore DOUBLED)
            gfactor (np.array): g-factors of the spins
            heis_int (dict): strengths of the Heisenberg interaction between the spins. Defaults to None.
            dipolar (boolean): whether the dipolar interaction is included or not. Defaults to True.
            zfs (nd.array): zero field splitting parameters for the spins shape: (4, num_spins) where the columns are the D, E, theta and phi
            radians (bool): whether the angles in the zfs matrix are in radians or degrees
        """        
        super().__init__(spins, gfactor, heis_int, zfs, radians)
        self.dipolar = dipolar
        self.cartesian_dip_ints = {}

    def get_dip_ints(self, connects):
        """Calculates the dipolar interaction matrices from the connecting vectors. Every spin always interacts with every other spin.

        Returns:
            np.array: dipolar interaction matrix in cartesian coordinates
        """        
        dip_ints = {}
        tidx = 0
        for i in range(self.spinnum - 1):
            for j in range(i + 1, self.spinnum):
                xyz_mat = self.get_single_dipmat(connects[tidx])
                self.cartesian_dip_ints[(i, j)] = xyz_mat
                dip_ints[(i, j)] = self.xyz_in_updownz(xyz_mat)
                tidx += 1
        return dip_ints
    
    def get_single_dipmat(self, connect):
        """Calculates the dipolar interaction matrix from a connecting vector.

        Args:
            connect (np.array): connecting vector in spherical coordinates (angles in radians)
            si (int): the spin quantum number of the first spin
            sj (int): the spin quantum number of the second spin
        Returns:
            np.array: dipolar interaction matrix in cartesian coordinates
        """       
        unit_factor = 0.622944725 #prefactor (mu_0*mu_B^2)/(4*pi*Angström^3) to retrieve the interaction in Kelvin
        xyz_mat = np.zeros((3, 3))
        #Diagonal elements of the dipolar interaction matrix
        xyz_mat[0, 0] = (1 - 3 * np.power(np.sin(connect[1]), 2) * np.power(np.cos(connect[2]), 2))
        xyz_mat[1, 1] = (1 - 3 * np.power(np.sin(connect[1]), 2) * np.power(np.sin(connect[2]), 2))
        xyz_mat[2, 2] = (1 - 3 * np.power(np.cos(connect[1]), 2))
        #off-diagonal elements of the dipolar interaction matrix
        xyz_mat[0, 1] = xyz_mat[1, 0] = - 3 * np.power(np.sin(connect[1]), 2) * np.cos(connect[2]) * np.sin(connect[2])
        xyz_mat[0, 2] = xyz_mat[2, 0] = - 3 * np.sin(connect[1]) * np.cos(connect[1]) * np.cos(connect[2])
        xyz_mat[1, 2] = xyz_mat[2, 1] = - 3 * np.sin(connect[1]) * np.cos(connect[1]) * np.sin(connect[2])
        for i in range(3):
            for j in range(3):
                if np.abs(xyz_mat[i, j]) < 1e-13:
                    xyz_mat[i, j] = 0.0
                    
        xyz_mat *= 1.0 / np.power(connect[0], 3)
        xyz_mat *= unit_factor * np.power(self.gfactor, 2)
        return xyz_mat
    
    def approx_dip_strength(self):
        """Approximates the influence of the dipolar interection by choosing the largest value in the dipolar interaction matrices
        
        Returns:
            float: Absolute value of the biggest entry in the dipolar interaction matrices
        """
        if not self.dipolar:
            return None
        else:
            max_val = 0
            for int_mat in self.cartesian_dip_ints.values():
                if np.abs(int_mat).max() > max_val:
                    max_val = np.abs(int_mat).max()
            return max_val


class SpinTetrahedron(AnisospinSystem):
    def __init__(self, spins, gfactor, distance, heis_int=None, dipolar=True, same_ints=False, zfs=None, radians=False):
        """Class for a spin system with 4 spins and dipolar interaction arrangened in a tetrahedron with spin 4 on the tip

        Args:
            spins (np.array): the spin quantum numbers of the spins in the system as integers (therefore DOUBLED) 
            gfactor (np.array): g-factors of the spins
            distance (float): The distances between the spins in the tetrahedron.
            heis_int (dict): strengths of the Heisenberg interaction between the spins.
            dipolar (boolean): whether dipolar interaction is included or not
            same_ints (boolean, optional): If the ligands are all the same for all spins and therefore the interactions are all the same. Defaults to False.
            zfs (nd.array): zero field splitting parameters for the spins shape: (4, num_spins) where the columns are the D, E, theta and phi
            radians (bool): whether the angles in the zfs matrix are in radians or degrees
        """        
        super().__init__(spins, gfactor, heis_int, dipolar, zfs, radians)
        self.distance = distance #distance between the spins in the tetrahedron
        self.same_ints = same_ints #If the ligands are all the same for all spins and therefore the interactions are all the same
        if heis_int is None:
            self.heis_int = {(0,1): 0, (0,2): 0, (0,3): 0, (1,2): 0, (1,3): 0, (2,3): 0}
        if dipolar:
            self.dip_connects = self.get_tetra_cons()
            self.dip_ints = self.get_dip_ints(self.dip_connects)
    
    def change_distance(self, distance):
        """Changes the distance between the spins in the tetrahedron

        Args:
            distance (float): new distance between the spins in the tetrahedron
        """        
        self.distance = distance
        self.dip_connects = self.get_tetra_cons()
        self.dip_ints = self.get_dip_ints(self.dip_connects)
    
    def update_heis(self, heis_vec):
        """Changes the Heisenberg interaction matrix

        Args:
            heis_vec (np.array): new Heisenberg interaction matrix
        """        
        if self.same_ints:
            self.heis_int = {sh: heis_vec[0] for sh in self.heis_int}
        else:
            self.heis_int = {sh: heis_vec[i] for i, sh in enumerate(self.heis_int)}

    def get_tetra_cons(self):
        """Calculates the connecting vectors in spherical coordinates of the tetrahedron with the 4. spin at the tip and edge-length self.distance

        Returns:
            np.array: connecting vectors for the dipolar interaction in spherical coordinates (angles in radians)
        """        
        #1st spin: (-distance/2, 0, 0), 2nd spin: (distance/2, 0, 0), 3rd spin: (0, sqrt(3)/2 * distance, 0), 4th spin: (0, sqrt(3)/6 * distance, sqrt(2/3) * distance)
        dip_connects = np.zeros((6, 3))
        dip_connects[0] = np.array([self.distance, np.pi/2, 0])
        dip_connects[1] = np.array([self.distance, np.pi/2, np.pi/3])
        dip_connects[2] = np.array([self.distance, np.arccos(np.sqrt(2/3)), np.pi/6])
        dip_connects[3] = np.array([self.distance, np.pi/2, 2*np.pi/3])
        dip_connects[4] = np.array([self.distance, np.arccos(np.sqrt(2/3)), 5*np.pi/6])
        dip_connects[5] = np.array([self.distance, np.arccos(np.sqrt(2/3)), 3*np.pi/2])
        return dip_connects

class SpinRing(AnisospinSystem):
    def __init__(self, spins, gfactor, radius, heis_int=None, dipolar=True, same_ints=False, zfs=None, radians=False):
        """Class for a spin system with N spins arranged in a ring where the first spin is at the position (radius, 0, 0)

        Args:
            spins (np.array): the spin quantum numbers of the spins in the system as integers (therefore DOUBLED)
            gfactor (np.array): g-factors of the spins
            radius (float): The radius of the spin ring in angstrom
            heis_int (dict): strengths of the Heisenberg interaction between the spins. Defaults to None.
            dipolar (boolean): whether dipolar interaction is included or not. Defaults to True.
            same_ints (boolean, optional): If the ligands are all the same for all spins and therefore the interactions are all the same. Defaults to False.
            zfs (nd.array): zero field splitting parameters for the spins shape: (4, num_spins) where the columns are the D, E, theta and phi
            radians (bool): whether the angles in the zfs matrix are in radians or degrees
        """        
        super().__init__(spins, gfactor, heis_int, dipolar, zfs, radians)
        self.radius = radius
        self.same_ints = same_ints
        if heis_int is None:
            self.heis_int = {(i, i+1): 0 for i in range(self.spinnum - 1)}
            self.heis_int[(0, self.spinnum - 1)] = 0
        if dipolar:
            self.number_connect = int(1/2 * self.spinnum * (self.spinnum - 1)) #number of connecting vectors (every spin interactions with every other spin)
            self.neighbour_dist = self.get_neighbour_dist()
            self.dip_connects = self.get_ring_cons()
            self.dip_ints = self.get_dip_ints(self.dip_connects)
        
    def change_radius(self, radius):
        """Changes the radius of the ring

        Args:
            radius (float): new radius of the ring
        """        
        self.radius = radius
        self.neighbour_dist = self.get_neighbour_dist()
        self.dip_connects = self.get_ring_cons()
        self.dip_ints = self.get_dip_ints(self.dip_connects)

    def update_heis(self, heis_vec):
        """Changes the Heisenberg interaction matrix

        Args:
            heis_vec (np.array): new Heisenberg interaction matrix
        """        
        if self.same_ints:
            self.heis_int = {sh: heis_vec[0] for sh in self.heis_int}
        else:
            self.heis_int = {sh: heis_vec[i] for i, sh in enumerate(self.heis_int)}

    def get_ring_cons(self):
        """Calculates the connecting vectors in spherical coordinates of the ring with the spins on the N-th roots of unity on a circle in the xy-plane with the radius self.radius

        Returns:
            np.array: connecting vectors for the dipolar interaction in spherical coordinates (angles in radians)
        """        
        #The spins sit on the N-th roots of unity on the unit circle in the xy-plane with the radius self.radius first spin at (self.radius, 0, 0)
        dip_connects = np.zeros((self.number_connect, 3))
        idx = 0
        for i in range(self.spinnum - 1):
            for j in range(i + 1, self.spinnum):
                dip_connects[idx] = [self.neighbour_dist[j-i-1], np.pi/2, np.pi/self.spinnum *(j+i-2)]
                idx += 1
        return dip_connects
    
    def get_neighbour_dist(self):
        """Calculates the distances between the first spin and its i-th neighbours

        Returns:
            np.array: distances between the first spin and its i-th neighbours
        """        
        neighbour_dist = np.zeros(self.spinnum - 1)
        for i in range(self.spinnum//2): 
            neighbour_dist[i] = neighbour_dist[-(i+1)] = 2 * self.radius * np.sin(np.pi * (i+1)/self.spinnum)
        return neighbour_dist   
    

class SpinChain(AnisospinSystem):
    def __init__(self, spins, gfactor, distance, heis_int=None, dipolar=True, same_ints=False, zfs=None, radians=False):
        """Class for a spin system with N spins arranged in a straight chain aligned on the x axis

        Args:
            spins (np.array): the spin quantum numbers of the spins in the system as integers (therefore DOUBLED)
            gfactor (np.array): g-factors of the spins
            distance (float): The distance of the spin chain in angstrom
            heis_int (dict): strengths of the Heisenberg interaction between the spins. Defaults to None.
            dipolar (boolean): whether dipolar interaction is included or not. Defaults to True.
            same_ints (boolean, optional): If the ligands are all the same for all spins and therefore the interactions are all the same. Defaults to False.
            zfs (nd.array): zero field splitting parameters for the spins shape: (4, num_spins) where the columns are the D, E, theta and phi
            radians (bool): whether the angles in the zfs matrix are in radians or degrees
        """        
        super().__init__(spins, gfactor, heis_int, dipolar, zfs, radians)
        self.distance = distance
        self.same_ints = same_ints
        if heis_int is None:
            self.heis_int = {(i, i+1): 0 for i in range(self.spinnum - 1)}
        if dipolar:
            self.number_connect = int(1/2 * self.spinnum * (self.spinnum - 1)) #number of connecting vectors (every spin interactions with every other spin)
            self.dip_connects = self.get_chain_cons()
            self.dip_ints = self.get_dip_ints(self.dip_connects)
        
    def change_distance(self, distance):
        """Changes the distance of the chain

        Args:
            distance (float): new distance of the chain
        """        
        self.distance = distance
        self.dip_connects = self.get_chain_cons()
        self.dip_ints = self.get_dip_ints(self.dip_connects)

    def update_heis(self, heis_vec):
        """Changes the Heisenberg interaction matrix

        Args:
            heis_vec (np.array): new Heisenberg interaction matrix
        """        
        if self.same_ints:
            self.heis_int = {sh: heis_vec[0] for sh in self.heis_int}
        else:
            self.heis_int = {sh: heis_vec[i] for i, sh in enumerate(self.heis_int)}
    
    def get_chain_cons(self):
        """Calculates the connecting vectors of the chain with spacings distance

        Returns:
            np.array: connecting vectors for the dipolar interaction (angles in radians)
        """        
        dip_connects = np.zeros((self.number_connect, 3))
        idx = 0
        for i in range(self.spinnum - 1):
            for j in range(i + 1, self.spinnum):
                dip_connects[idx] = [self.distance * (j-i), 0, 0]
                idx += 1
        return dip_connects
    
    
class SpinButterfly(AnisospinSystem):
    def __init__(self, spins, gfactor, diagonals, heis_int=None, dipolar=True, zfs=None, radians=False):
        """Class for a spin system in a butterfly shape with 4 spins and non-negligible dipolar interaction
           The diagonal interaction is between spin 0 and 2 while no interaction is between spin 1 and 3

        Args:
            spins (np.array): the spin quantum numbers of the spins in the system as integers (therefore DOUBLED)
            gfactor (np.array): g-factors of the spins
            diagonals (tuple): The two diagonal distances between the spins in the butterfly in angstrom
            heis_int (dict): strengths of the Heisenberg interaction between the spins. Defaults to None.
            dipolar (boolean): whether dipolar interaction is included or not. Defaults to True.
            zfs (nd.array): zero field splitting parameters for the spins shape: (4, num_spins) where the columns are the D, E, theta and phi
            radians (bool): whether the angles in the zfs matrix are in radians or degrees
        """        
        super().__init__(spins, gfactor, heis_int, dipolar, zfs, radians)
        self.diagonals = diagonals
        if heis_int is None:
            self.heis_int = {(0,1): 0, (0,2): 0, (0,3): 0, (1,2): 0, (2,3): 0}
        if dipolar:
            self.number_connect = int(1/2 * self.spinnum * (self.spinnum - 1)) #number of connecting vectors (every spin interactions with every other spin)
            self.dip_connects = self.get_butterfly_cons()
            self.dip_ints = self.get_dip_ints(self.dip_connects)

    def change_diagonals(self, diagonals):
        """Changes the diagonal distances between the spins in the butterfly

        Args:
            diagonals (tuple): new diagonal distances between the spins in the butterfly
        """        
        self.diagonals = diagonals
        self.dip_connects = self.get_butterfly_cons()
        self.dip_ints = self.get_dip_ints(self.dip_connects)
    
    def get_butterfly_cons(self):
        """Calculates the connecting vectors in spherical coordinates of the butterfly with diagonals self.diagonals (the first diagonal is between spin 1 and 3)

        Returns:
            np.array: connecting vectors for the dipolar interaction in spherical coordinates (angles in radians)
        """        
        dip_connects = np.zeros((self.number_connect, 3))
        spin_pos = np.zeros((self.spinnum, 3))
        spin_pos[0] = np.array([self.diagonals[0]/2, 0, 0])
        spin_pos[1] = np.array([0, self.diagonals[1]/2, 0])
        spin_pos[2] = np.array([-self.diagonals[0]/2, 0, 0])
        spin_pos[3] = np.array([0, -self.diagonals[1]/2, 0])
        tidx = 0
        for i in range(self.spinnum-1):
            for j in range(i+1, self.spinnum):
                dip_connects[tidx] = self.cartesian_to_spherical(spin_pos[j] - spin_pos[i])
                tidx += 1
        return dip_connects

