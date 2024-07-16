#!/usr/bin/env python
"""Module providing a test class for testing the spin classes and hamiltonian classse"""
import copy
import numpy as np
import spin_class as spc
import hamiltonian_class as hc
import thermodyn_obs as tdob

muB_kb = 0.6717141002 # muB/kb in Kelvin


#Class for testing the functionality of the spin classes and functions
class TestClass:
    """Class for automatic testing of all implemented classes
    """
    tspins = [1, 1, 1, 1] #4 spin 1/2 (doubled) as test system
    tgfac = 2.0 #standard g-factors of the spins
    theis = {(0,1):1, (0,2):-1, (0,3):2, (1,2):-2, (1,3):4, (2,3):-4} #Heisenberg interaction between all spins
    spin_sys = spc.SpinSystem(tspins, tgfac, theis) #SpinSystem instance
    def __init__(self) -> None:
        """Constructor
        """
        self.test()

    def test(self):
        """Tests the functionality of the spin classes and functions
        """
        test_ham_justJ = hc.SpinHamiltonian(self.spin_sys) #SpinHamiltonian instance
        assert(test_ham_justJ.dim == 16), "Dimension test failed"
        #test the heisenberg term
        assert test_ham_justJ.is_hermitian(), "Hamiltonian not hermitian"
        teigvals_justJ = np.linalg.eigvalsh(test_ham_justJ.hamilmat)
        assert np.allclose(teigvals_justJ, eigvals_justJ), "Eigenvalues of just Heisenberg interactions are wrong"
        print("Heisenberg spin-1/2 test passed")
        test_ham_7Tz = copy.deepcopy(test_ham_justJ) #Copy the Hamiltonian

        #test the zeeman term
        test_ham_7Tz.add_zeeman([0, 0, 7.0]) #Add Zeeman interaction to the Hamiltonian
        assert test_ham_7Tz.is_hermitian(), "Hamiltonian with Heisenberg and Zeeman not hermitian"
        teigvals_7tesla = np.linalg.eigvalsh(test_ham_7Tz.hamilmat)
        assert np.allclose(teigvals_7tesla, eigvals_7tesla), "Zeeman Term not giving correct eigenvalues"

        #test the rotational symmetry of the Zeeman term
        test_ham_7Tr = copy.deepcopy(test_ham_justJ)
        test_randbdir = np.array([0.74478079, 0.22116196, 0.62959428])#Some random direction for the magnetic field
        test_randbdir *= [7, 7, 7]
        test_ham_7Tr.add_zeeman(test_randbdir)
        assert np.allclose(teigvals_7tesla, np.linalg.eigvalsh(test_ham_7Tr.hamilmat))
        print("Zeeman spin-1/2 test passed")
        print("Standard spin-1/2 test passed\n")

    def test_spin1(self):
        """Tests the functionality of the spin classes and functions for spin-1 systems
        """
        tspins1 = [2, 2, 2, 2]
        spin_sys1 = spc.SpinSystem(tspins1, self.tgfac, self.theis)
        test_ham1 = hc.SpinHamiltonian(spin_sys1)
        assert test_ham1.is_hermitian()
        tevals_spin1_justJ = np.linalg.eigvalsh(test_ham1.hamilmat)
        assert np.allclose(tevals_spin1_justJ, evals_spin1_justJ)
        print("Spin 1 test passed (Heisenberg)\n")
        return True
    
    def test_dip(self):
        """Tests the functionality of the spin classes and functions for dipolar interaction
        """
        tnoheis = {}
        spin_sys_dip = spc.SpinTetrahedron(self.tspins, self.tgfac, 1.0, tnoheis, True)
        test_ham_dip = hc.PosSpinHamiltonian(spin_sys_dip)
        for sh in test_ham_dip.spin_sys.dip_ints:
            assert np.allclose(test_ham_dip.spin_sys.cartesian_dip_ints[sh], tdips_ints[sh])
        assert test_ham_dip.is_hermitian()
        assert np.allclose(eigvals_justDip, np.linalg.eigvalsh(test_ham_dip.hamilmat))
        spin_sys_dipring = spc.SpinRing(self.tspins, self.tgfac, 0.5, tnoheis, True)
        test_ham_dipring = hc.PosSpinHamiltonian(spin_sys_dipring)
        assert np.allclose(eigvals_justDip_4ring, np.linalg.eigvalsh(test_ham_dipring.hamilmat))
        spin_sys_dipbutter = spc.SpinButterfly(self.tspins, self.tgfac, (2, 3), tnoheis, True)
        test_ham_dipbutter = hc.PosSpinHamiltonian(spin_sys_dipbutter)
        assert np.allclose(eigvals_justDip_butterfly, np.linalg.eigvalsh(test_ham_dipbutter.hamilmat))

        spin_tetra_nodip = spc.SpinTetrahedron(self.tspins, self.tgfac, 1.0, self.theis, False)
        test_ham_nodip = hc.PosSpinHamiltonian(spin_tetra_nodip)
        test_ham_nodip.change_heis(np.array(list(self.theis.values())))
        assert np.allclose(eigvals_justJ, np.linalg.eigvalsh(test_ham_nodip.hamilmat))
        print("Dipolar spin-1/2 test passed\n")
    
    def test_zfs(self):
        """Tests the functionality of the spin classes and functions for zero field splitting
        """
        tnoheis = {}
        spins = [1, 1, 1]
        gfactor = 2.0
        spin_sys_zfs = spc.SpinSystem(spins, gfactor)
        d = [0.02]*3
        e = [0.0]*3
        theta = [48, 132, 90]
        phi = [0, 60, 120]
        zfsmat = [d, e, theta, phi]
        zfsmat = np.array(zfsmat)
        spin_sys_zfs.update_zfs(zfsmat)
        pmz_test_zfs = [spin_sys_zfs.xyz_in_updownz(test_zfs[i]) for i in range(spin_sys_zfs.spinnum)]
        assert np.allclose(pmz_test_zfs, spin_sys_zfs.zfs_mats), "Zero field splitting not set correctly"
        spins = [3, 3]
        heis = {(0, 1): 2}
        aniso = np.zeros((4, 2))
        aniso[0, :] = -5.0
        aniso[2, :] = [50, 130]
        aniso[3, :] = [60, 120]
        gfactor = 2.0

        spin_zfs_sys2 = spc.SpinSystem(spins, gfactor, heis, aniso, radians=False)
        test_ham_zfs = hc.SpinHamiltonian(spin_zfs_sys2)
        eig = np.linalg.eigvalsh(test_ham_zfs.hamilmat)
        assert np.allclose(eig, eigvals_zfs), "ZFS Hamiltonian incorrect"
        print("Zero field splitting test passed\n")

class TestFunctions:
    """Class for automatic testing of all implemented (thermodynamic) functions
    """
    tspins = [1] #4 spin 1/2 (doubled) as test system
    tgfac = 2.0 #standard g-factors of the spins
    T = 1.3647 #Some arbitrary temperature in Kelvin
    tb_arr = np.linspace(0, 7, 10) #Magnetic field array
    test_spin1_2 = spc.SpinSystem(tspins, tgfac)
    test_ham_s1_2 = hc.SpinHamiltonian(test_spin1_2)
    def __init__(self) -> None:
        """Constructor
        """
        self.test()
    
    def test(self):
        tentropy = np.zeros(len(self.tb_arr)) #Entropy array
        tgibbs_entropy = np.zeros(len(self.tb_arr)) #Entropy array
        t_analytic_entropy = np.zeros(len(self.tb_arr)) #Analytic entropy array
        for i, tb in enumerate(self.tb_arr):
            bvec = np.array([0, 0, tb])
            tentropy[i] = tdob.entropy(self.test_ham_s1_2, self.T, bvec)
            tgibbs_entropy[i] = tdob.entropy_gibbs(self.test_ham_s1_2, self.T, bvec)
            t_analytic_entropy[i] = tdob.analytic_test_entropy_s1_2(self.T, bvec)
        
        assert np.allclose(tentropy, t_analytic_entropy), "Entropy not calculated correctly"
        assert np.allclose(tgibbs_entropy, t_analytic_entropy), "Gibbs entropy not calculated correctly"
        print("Entropy test passed")


#Test Hamiltonians for spin systems with Heisenberg interaction and a Zeeman term
eigvals_justJ = [-3.3644149e+00, -3.3644149e+00, -3.3644149e+00, -2.5980761e+00,
 -1.5209994e+00, -1.5209994e+00, -1.5209994e+00, -1.3825261e-16,
  0.0000000e+00,  0.0000000e+00,  2.4632907e-16,  7.6080150e-16,
  2.5980761e+00,  4.8854141e+00,  4.8854141e+00,  4.8854141e+00]
eigvals_7tesla = [-18.8079948056, -12.76841226670681, -10.924996874504563, -9.403997402799972,
                  -4.518583067188592, -3.3644148639068323, -2.5980762113533036,
                  -1.5209994717045365, 1.2434497875801753e-14, 2.5980762113533213, 
                  4.885414335611387, 6.039582538893155, 7.8829979310954625, 
                  9.403997402799991, 14.289411738411381, 18.8079948056]
evals_spin1_justJ = [-9.5655804e+00, -9.5655804e+00, -9.5655804e+00, -9.5655804e+00,
 -9.5655804e+00, -8.5207977e+00, -8.5207977e+00, -8.5207977e+00,
 -8.5207977e+00, -8.5207977e+00, -8.5207977e+00, -8.0052538e+00,
 -8.0052538e+00, -8.0052538e+00, -8.0052538e+00, -8.0052538e+00,
 -7.9372540e+00, -6.7288299e+00, -6.7288299e+00, -6.7288299e+00,
 -6.7288299e+00, -6.7288299e+00, -6.7288299e+00, -6.7288299e+00,
 -4.9980402e+00, -4.9980402e+00, -4.9980402e+00, -4.9980402e+00,
 -4.9980402e+00, -3.0419989e+00, -3.0419989e+00, -3.0419989e+00,
 -3.0419989e+00, -3.0419989e+00, -3.0419989e+00, -3.0419989e+00,
 -1.3245553e+00, -1.3245553e+00, -1.3245553e+00, -4.2806360e-15,
 -2.9998051e-15, -2.2947029e-15, -2.1949974e-15, -1.4254408e-15,
 -2.1935030e-16, -1.6160692e-31,  1.1259139e-18,  2.6363151e-16,
  4.6000225e-15,  6.0721077e-02,  6.0721077e-02,  6.0721077e-02,
  6.0721077e-02,  6.0721077e-02,  3.5207973e+00,  3.5207973e+00,
  3.5207973e+00,  3.5207973e+00,  3.5207973e+00,  3.5207973e+00,
  5.8632030e+00,  5.8632030e+00,  5.8632030e+00,  5.8632030e+00,
  5.8632030e+00,  7.9372540e+00,  9.7708282e+00,  9.7708282e+00,
  9.7708282e+00,  9.7708282e+00,  9.7708282e+00,  9.7708282e+00,
  9.7708282e+00,  1.1324555e+01,  1.1324555e+01,  1.1324555e+01,
  1.6644949e+01,  1.6644949e+01,  1.6644949e+01,  1.6644949e+01,
  1.6644949e+01]

#Tests for spin systems with dipolar interaction
tdips_ints = {(0,1): [[-4.9835578, 0, 0],[0, 2.4917789, 0],[0, 0, 2.4917789]],
              (0,2): [[0.622944725, -3.236915742, 0],[-3.236915742, -3.114723625, 0],[0, 0, 2.4917789]],
              (0,3): [[0.622944725, -1.078971914, -3.0517934284],[-1.078971914, 1.868834175, -1.7619537574076],[-3.0517934284, -1.7619537574076, -2.4917789]],
              (1,2): [[0.622944725, 3.236915742, 0],[3.236915742, -3.114723625, 0],[0, 0, 2.4917789]],
              (1,3): [[0.622944725, 1.078971914, 3.0517934284],[1.078971914, 1.868834175, -1.7619537574076],[3.0517934284, -1.7619537574076, -2.4917789]],
              (2,3): [[2.4917789, 0, 0],[0, 0, 3.5239075148151113],[0, 3.5239075148151113, -2.4917789]]}
hamiltonian_jusDip = [[-2.220446049250313e-16 + 0.0*1j, 0.0 + 2.220446049250313e-16*1j, 0.0 + 0.8809768787037778*1j, 0.622944725 + 0.0*1j, 0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + 0.5394859570035108*1j, 0.9344170874999996 + 1.6184578710105344*1j, 0.0 + 0.0*1j, -0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + -0.5394859570035108*1j, 0.9344170874999999 + -1.6184578710105342*1j, 0.0 + 0.0*1j, -1.868834175 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j],
[0.0 + -2.220446049250313e-16*1j, 3.7376683500000007 + 0.0*1j, 0.622944725 + 0.0*1j, 0.0 + -0.8809768787037778*1j, 0.6229447250000003 + 0.0*1j, -0.7629483571041937 + 0.4404884393518888*1j, 0.0 + 0.0*1j, 0.9344170874999996 + 1.6184578710105344*1j, 0.6229447250000003 + 0.0*1j, 0.7629483571041937 + 0.4404884393518888*1j, 0.0 + 0.0*1j, 0.9344170874999999 + -1.6184578710105342*1j, 0.0 + 0.0*1j, -1.868834175 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j],
[0.0 + -0.8809768787037778*1j, 0.622944725 + 0.0*1j, -1.2458894500000006 + 0.0*1j, 0.0 + -1.7619537574075554*1j, -0.6229447250000003 + 0.0*1j, 0.0 + 0.0*1j, 0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + 0.5394859570035108*1j, -0.6229447250000001 + 0.0*1j, 0.0 + 0.0*1j, -0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + -0.5394859570035108*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -1.868834175 + 0.0*1j, 0.0 + 0.0*1j],
[0.622944725 + 0.0*1j, 0.0 + 0.8809768787037778*1j, 0.0 + 1.7619537574075554*1j, -2.220446049250313e-16 + 0.0*1j, 0.0 + 0.0*1j, -0.6229447250000003 + 0.0*1j, 0.6229447250000003 + 0.0*1j, -0.7629483571041937 + 0.4404884393518888*1j, 0.0 + 0.0*1j, -0.6229447250000001 + 0.0*1j, 0.6229447250000003 + 0.0*1j, 0.7629483571041937 + 0.4404884393518888*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -1.868834175 + 0.0*1j],
[0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, -0.6229447250000003 + 0.0*1j, 0.0 + 0.0*1j, -1.2458894500000002 + 0.0*1j, -1.5258967142083875 + 0.8809768787037778*1j, 0.0 + 0.8809768787037778*1j, 0.622944725 + 0.0*1j, -0.622944725 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + -0.5394859570035108*1j, 0.9344170874999999 + -1.6184578710105342*1j, 0.0 + 0.0*1j],
[-0.3114723624999999 + -0.5394859570035108*1j, -0.7629483571041937 + -0.4404884393518888*1j, 0.0 + 0.0*1j, -0.6229447250000003 + 0.0*1j, -1.5258967142083875 + -0.8809768787037778*1j, 2.220446049250313e-16 + 0.0*1j, 0.622944725 + 0.0*1j, 0.0 + -0.8809768787037778*1j, 0.0 + 0.0*1j, -0.622944725 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.6229447250000003 + 0.0*1j, 0.7629483571041937 + 0.4404884393518888*1j, 0.0 + 0.0*1j, 0.9344170874999999 + -1.6184578710105342*1j],
[0.9344170874999996 + -1.6184578710105344*1j, 0.0 + 0.0*1j, 0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, 0.0 + -0.8809768787037778*1j, 0.622944725 + 0.0*1j, 2.220446049250313e-16 + 0.0*1j, -1.5258967142083875 + -0.8809768787037778*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -0.622944725 + 0.0*1j, 0.0 + 0.0*1j, -0.6229447250000001 + 0.0*1j, 0.0 + 0.0*1j, -0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + -0.5394859570035108*1j],
[0.0 + 0.0*1j, 0.9344170874999996 + -1.6184578710105344*1j, -0.3114723624999999 + -0.5394859570035108*1j, -0.7629483571041937 + -0.4404884393518888*1j, 0.622944725 + 0.0*1j, 0.0 + 0.8809768787037778*1j, -1.5258967142083875 + 0.8809768787037778*1j, -1.2458894500000002 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -0.622944725 + 0.0*1j, 0.0 + 0.0*1j, -0.6229447250000001 + 0.0*1j, 0.6229447250000003 + 0.0*1j, 0.7629483571041937 + 0.4404884393518888*1j],
[-0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, -0.6229447250000001 + 0.0*1j, 0.0 + 0.0*1j, -0.622944725 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -1.2458894500000002 + 0.0*1j, 1.5258967142083875 + 0.8809768787037778*1j, 0.0 + 0.8809768787037778*1j, 0.622944725 + 0.0*1j, 0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + 0.5394859570035108*1j, 0.9344170874999996 + 1.6184578710105344*1j, 0.0 + 0.0*1j],
[-0.3114723624999999 + 0.5394859570035108*1j, 0.7629483571041937 + -0.4404884393518888*1j, 0.0 + 0.0*1j, -0.6229447250000001 + 0.0*1j, 0.0 + 0.0*1j, -0.622944725 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 1.5258967142083875 + -0.8809768787037778*1j, 2.220446049250313e-16 + 0.0*1j, 0.622944725 + 0.0*1j, 0.0 + -0.8809768787037778*1j, 0.6229447250000003 + 0.0*1j, -0.7629483571041937 + 0.4404884393518888*1j, 0.0 + 0.0*1j, 0.9344170874999996 + 1.6184578710105344*1j],
[0.9344170874999999 + 1.6184578710105342*1j, 0.0 + 0.0*1j, -0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -0.622944725 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + -0.8809768787037778*1j, 0.622944725 + 0.0*1j, 2.220446049250313e-16 + 0.0*1j, 1.5258967142083875 + -0.8809768787037778*1j, -0.6229447250000003 + 0.0*1j, 0.0 + 0.0*1j, 0.7629483571041937 + -0.4404884393518888*1j, -0.3114723624999999 + 0.5394859570035108*1j],
[0.0 + 0.0*1j, 0.9344170874999999 + 1.6184578710105342*1j, -0.3114723624999999 + 0.5394859570035108*1j, 0.7629483571041937 + -0.4404884393518888*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -0.622944725 + 0.0*1j, 0.622944725 + 0.0*1j, 0.0 + 0.8809768787037778*1j, 1.5258967142083875 + 0.8809768787037778*1j, -1.2458894500000002 + 0.0*1j, 0.0 + 0.0*1j, -0.6229447250000003 + 0.0*1j, 0.6229447250000003 + 0.0*1j, -0.7629483571041937 + 0.4404884393518888*1j],
[-1.868834175 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, -0.6229447250000001 + 0.0*1j, 0.0 + 0.0*1j, 0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, -0.6229447250000003 + 0.0*1j, 0.0 + 0.0*1j, -2.220446049250313e-16 + 0.0*1j, 0.0 + 1.7619537574075554*1j, 0.0 + 0.8809768787037778*1j, 0.622944725 + 0.0*1j],
[0.0 + 0.0*1j, -1.868834175 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -0.3114723624999999 + 0.5394859570035108*1j, 0.7629483571041937 + -0.4404884393518888*1j, 0.0 + 0.0*1j, -0.6229447250000001 + 0.0*1j, -0.3114723624999999 + -0.5394859570035108*1j, -0.7629483571041937 + -0.4404884393518888*1j, 0.0 + 0.0*1j, -0.6229447250000003 + 0.0*1j, 0.0 + -1.7619537574075554*1j, -1.2458894500000006 + 0.0*1j, 0.622944725 + 0.0*1j, 0.0 + -0.8809768787037778*1j],
[0.0 + 0.0*1j, 0.0 + 0.0*1j, -1.868834175 + 0.0*1j, 0.0 + 0.0*1j, 0.9344170874999999 + 1.6184578710105342*1j, 0.0 + 0.0*1j, -0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, 0.9344170874999996 + -1.6184578710105344*1j, 0.0 + 0.0*1j, 0.7629483571041937 + 0.4404884393518888*1j, 0.6229447250000003 + 0.0*1j, 0.0 + -0.8809768787037778*1j, 0.622944725 + 0.0*1j, 3.7376683500000007 + 0.0*1j, 0.0 + -2.220446049250313e-16*1j],
[0.0 + 0.0*1j, 0.0 + 0.0*1j, 0.0 + 0.0*1j, -1.868834175 + 0.0*1j, 0.0 + 0.0*1j, 0.9344170874999999 + 1.6184578710105342*1j, -0.3114723624999999 + 0.5394859570035108*1j, 0.7629483571041937 + -0.4404884393518888*1j, 0.0 + 0.0*1j, 0.9344170874999996 + -1.6184578710105344*1j, -0.3114723624999999 + -0.5394859570035108*1j, -0.7629483571041937 + -0.4404884393518888*1j, 0.622944725 + 0.0*1j, 0.0 + 0.8809768787037778*1j, 0.0 + 2.220446049250313e-16*1j, -2.220446049250313e-16 + 0.0*1j],
]
eigvals_justDip = [-4.147287244783609, -4.147287244783608, -3.114723625000002, -3.114723625, -3.114723625, -2.3497486692629046, -2.3497486692629015, -2.3497486692629006, -1.245889450, 0.9749693920467081, 0.9749693920467143, 2.9726933942628864, 2.9726933942628957, 2.972693394262905, 7.532930927736875, 7.532930927736915]
eigvals_justDip_4ring = [-12.203981997230978, -11.817611994445338, -9.398053636267893, -8.079943644445336, -5.311377098335974, -4.611737192228444, -4.611737192228436, -1.245889449999991, -1.245889449999991, 1.1043491566376478, 3.3658477422284414, 3.36584774222845, 8.293704479630213, 9.325833094445331, 13.063501444445345, 20.007137995566957]
eigvals_justDip_butterfly = [-0.7028217778844884, -0.6923462388566206, -0.5796399355309613, -0.48241456212196165, -0.39709307515331166, -0.2719946237132005, -0.20922203037938525, -0.15573618125000044, -0.15437834140921725, -0.056106278416911515, -0.04614405370370378, 0.4904660039029173, 0.5458626359645545, 0.7340182769401781, 0.8008535450607187, 1.176696636551394]
test_zfs = [[[ 0.01104528, 0., 0.00994522], [ 0., 0., 0.], [ 0.00994522, 0., 0.00895472]], [[ 0.00276132, 0.00478275, -0.00497261], [ 0.00478275, 0.00828396, -0.00861281], [-0.00497261, -0.00861281, 0.00895472]], [[ 0.005, -0.00866025, 0.], [-0.00866025, 0.015, 0.], [ 0., 0., 0.]]]
eigvals_zfs = [-24.601861935995917, -24.44383986194549, -23.49283432415717, -23.474542486263477, -16.293771449322378, -15.386212942836861, -15.125545150081205, -14.000000000000004, -10.112615128106862, -9.217923806171095, -8.718118071208156, -8.000000000000004, -6.1760469496936246, -1.3411620506177293, 0.09697107316509346, 0.287503083234828]
