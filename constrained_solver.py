#import h5py
from dolfin import *
import numpy as np
from petsc4py import PETSc
from model.support.physical_constants import *

class ConstrainedSolver(object):

    def __init__(self, model):
        # Model
        self.model = model
        # MPI comm
        self.comm = model.comm
        # Unknown
        self.u = model.H
        # PDE residual
        self.F = model.F
        # PDE Jacobian
        self.F_du = model.J


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
        # Generic matrix to store phi / du
        self.F_du_d = assemble(self.F_du)
        # Petsc matrix to store Jacobian of phi
        self.phi_du_p = as_backend_type(self.F_du_d).mat()
        # Generic vector for storing phi residual
        self.phi_d = assemble(self.F)


        ### Linear solver
        #PETScOptions().set("ksp_view", "")
        self.ksp = PETSc.KSP()
        self.ksp.create(self.comm)
        self.ksp.setFromOptions()


    ### Assemble PDE residual stored in F_d.
    def __assemble_F__(self):
        assemble(self.F, tensor = self.F_d)


    ### Assemble PDE Jacobian dF / du stored in F_du_d. The petsc matrix
    ### phi_du_p is automatically updated upon assembly
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
        indexes = diag_denom > 1e-16
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


    def __assemble_lam__(self):
        self.ksp.setOperators(self.phi_du_p)
        self.ksp.solveTranspose(self.dg_du_p, self.lam_p)
        print self.lam_p.getArray()


    ### Assemble vector du / dp stored in u_dp_p
    def __assemble_u_dp__(self):
        # (phi / du) (u / dp) = -(f / dp)
        self.ksp.setOperators(self.phi_du_p)
        self.ksp.solve(-self.phi_dp_p, self.u_dp_p)

        from matplotlib import pyplot as plt
        plt.plot(self.u_dp_p.getArray())
        plt.show()


    ### Compute the adjoint vector lambda
    def compute_adjoint(self):
        self.update_u()

        print



    ### Compute value of scalar merit function
    def psi(self, u):
        return self.phi_d.norm('l2')


    ### Assemble everything that depends on u
    def assemble_u(self):
        self.__assemble_F__()
        self.__assemble_phi__()
        self.__assemble_F_du__()
        self.__assemble_phi_du__()
        self.__assemble_dg_du__()


    ### Assemble everything that depends on p
    def assemble_p(self):
        self.__assemble_phi_dp__()
