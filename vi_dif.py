#import h5py
from dolfin import *
import numpy as np
from petsc4py import PETSc
from model.support.physical_constants import *
from scipy.sparse import csr_matrix

"""
Associated with the steady state PDE F(u, p) = 0 is a NCP that enforces that u >= 0.
Here u is the PDE unknown and p is a scalar parameter. This NCP is solved by solving
the equivalent system Phi(u, p) = 0. This object handles differentiation of Phi
by u and p.
"""

class VIDif(object):

    def __init__(self, u, F, F_du, comm):
        # MPI comm
        self.comm = comm
        # Unknown
        self.u = u
        # PDE residual
        self.F = F
        # PDE Jacobian
        self.F_du = F_du


        ### Storage matrices and vectors

        # Local vector length
        self.N_local = len(self.u.vector().array())
        # Unknown H as a petsc vector
        self.u_p = as_backend_type(self.u.vector()).vec()
        # PDE residual as petsc vector
        self.F_p = self.u_p.duplicate()
        # Diagonal matrices for phi_du
        self.D_a_diag_p = self.u_p.duplicate()
        self.D_b_diag_p = self.u_p.duplicate()
        # Local arrays for setting D_a_diag_p, D_b_diag_p
        self.D_a_diag_np = np.zeros(self.N_local)
        self.D_b_diag_np = np.zeros(self.N_local)
        # Generic vector to store F
        self.F_d = assemble(self.F)
        # Generic matrix to store F / du
        self.F_du_d = assemble(self.F_du)
        # Petsc matrix to store Jacobian of phi
        self.phi_du_p = as_backend_type(self.F_du_d).mat()
        # Generic vector for storing phi residual
        self.phi_d = assemble(self.F)
        # dg_du as petsc vector
        self.dg_du_p = self.u_p.duplicate()
        self.dg_du_p.setArray(np.zeros(self.N_local))
        # phi_dp as petsc vector
        self.phi_dp_p = self.u_p.duplicate()
        # Generic vector storing du / dp
        self.u_dp_p = self.u_p.duplicate()


        ### Linear solver

        PETScOptions().set("ksp_monitor", "")
        self.ksp = PETSc.KSP()
        self.ksp.create(self.comm)
        self.ksp.setFromOptions()
        self.ksp.setTolerances(1e-16, 1e-16)


    ### Compute phi / dp
    def __assemble_phi_dp__(self, F_dp_np):
        # u as numpy array
        u_np = self.u.vector().array()
        # F as numpy array
        F_np = self.F_d.array()
        self.phi_dp_p.setArray((1.0 - (F_np / np.sqrt(u_np**2 + F_np**2))) * F_dp_np)


    ### Assemble PDE residual stored in F
    def __assemble_F__(self):
        assemble(self.F, tensor = self.F_d)


    ### Assemble PDE Jacobian dF / du stored in F_du_d
    def __assemble_F_du__(self):
        assemble(self.F_du, tensor = self.F_du_d)


    ### Assemble constrained problem residual stored in phi_d
    def __assemble_phi__(self):
        u_np = self.u.vector().array()
        F_np = self.F_d.array()
        phi_np = u_np + F_np - np.sqrt(u_np**2 + F_np**2)
        self.phi_d.set_local(phi_np)
        self.phi_d.apply("insert")


    ### Assemble constrained problem Jacobian phi / du stored in phi_du_p
    def __assemble_phi_du__(self):

        ### Compute the diagonal entries in D_a and D_b
        #######################################################################

        # Set default diagonal values
        self.D_a_diag_np[:] = 1.0
        self.D_b_diag_np[:] = 0.0
        # Residual as numpy array
        F_np = self.F_d.array()
        # u value as numpy array()
        u_np = self.u.vector().array()
        # Diagonal denominator
        diag_denom = np.sqrt(u_np**2 + F_np**2)
        # Get indexes where the diagonal denominator is non-zero
        indexes = diag_denom > 0.0
        # Compute diagonal values
        self.D_a_diag_np[indexes] = 1.0 - u_np[indexes] / diag_denom[indexes]
        self.D_b_diag_np[indexes] = 1.0 - F_np[indexes] / diag_denom[indexes]
        # Set local values of numpy diagonal vectors
        self.D_a_diag_p.setArray(self.D_a_diag_np)
        self.D_b_diag_p.setArray(self.D_b_diag_np)


        ### Now compute phi / du = D_a + D_b * F_du
        #######################################################################

        # When the generic matrix F_du_d is updated, phi_du_p is updated to
        # the same values
        self.phi_du_p.diagonalScale(self.D_b_diag_p)
        self.phi_du_p.setDiagonal(self.D_a_diag_p, 2)


    ### Assemble vector du / dp stored in u_dp_p
    def __assemble_u_dp__(self):
        # (phi / du) (u / dp) = -(f / dp)
        self.ksp.setOperators(self.phi_du_p)
        self.ksp.solve(-self.phi_dp_p, self.u_dp_p)


    ### Assemble everything that depends on u
    def update_u(self):
        self.__assemble_F__()
        self.__assemble_phi__()
        self.__assemble_F_du__()
        self.__assemble_phi_du__()
