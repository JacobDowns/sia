#import h5py
from dolfin import *
import numpy as np
import petsc4py
from petsc4py import PETSc
import sys

petsc4py.init(sys.argv)

### Basically a copy of SNESVI solver
class ConstrainedSolver(object):

    def __init__(self, u, F, F_du, tol = 1e-6, iter_limit = 200):
        # Unknown
        self.u = u
        # PDE residual
        self.F = F
        # PDE Jacobian
        self.F_du = F_du
        # Newton tolerance
        self.tol = tol
        # Maximumum iterations
        self.iter_limit = iter_limit



        # Unknown as petsc vector
        self.u_p = as_backend_type(u.vector()).vec()
        self.u_p_solve = self.u_p.copy()

        # Local vector length
        self.N_local = len(u.vector().array())
        # Generic vector to store F
        self.F_d = assemble(F)
        # Petsc vector to store F
        self.F_p = as_backend_type(self.F_d).vec()
        # Generic matrix to store phi / du
        self.F_du_d = assemble(F_du)
        # Petsc matrix to store Jacobian of phi
        self.F_du_p = as_backend_type(self.F_du_d).mat()

        ### Create linear solver object


        PETScOptions().set("snes_view", "")
        PETScOptions().set("snes_monitor", "")
        PETScOptions().set("pc_type", "lu")
        snes = PETSc.SNES().create()
        snes.setType('vinewtonrsls')
        snes.setFunction(self.__assemble_F__, self.F_p)
        snes.setJacobian(self.__assemble_F_du__, self.F_du_p)
        snes.setFromOptions()

        #snes.view()

        #self.u_p_solve.setArray(1000.*np.ones(self.N_local))
        print self.u_p_solve.getArray()
        thing = snes.solve(None, self.u_p_solve)


        print snes.getIterationNumber()
        print snes.getLinearSolveIterations()
        print snes.getConvergedReason()



    ### Assemble PDE residual stored in F
    def __assemble_F__(self, snes, u, F_d):
        print u.getArray()
        self.u.vector().set_local(u.getArray())
        self.u.vector().apply("insert")
        assemble(self.F, tensor = self.F_d)


    ### Assemble PDE Jacobian dF / du stored in F_du_d
    def __assemble_F_du__(self, snes, u, A, P):
        print u
        self.u.vector().set_local(u.getArray())
        self.u.vector().apply("insert")
        assemble(self.F_du, tensor = self.F_du_d)






    ### Do a constrained solve
    def solve(self):
        i = 0
        self.update_u()

        # Do Newton iteration
        while self.psi() > self.tol and i <= 1000:
            # Compute Jacobian of non-smooth function at u
            phi_du_u = self.phi_du(u)

            # Value of non-smooth function at u0
            phi_u = self.phi(u)
            # Compute search direction
            d = np.linalg.solve(phi_du_u, -phi_u)
            # Update u0
            u += d
            i += 1

        self.set_u(u)
        return u
