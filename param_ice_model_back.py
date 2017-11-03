#import h5py
from dolfin import *
from model.model_inputs import *
from model.ice_model import *
from model.support.physical_constants import *
from vi_dif import *
from petsc4py import PETSc
from adjointer import *
import numpy as np

"""
A parameterized model that solves F(p) = 0.
"""

class ParamIceModel(object):

    def __init__(self, model_inputs):

        ### Define form of surface mass balance

        p_func = Function(model_inputs.V)
        p = Constant(1.0)
        p_var = variable(p)
        x = SpatialCoordinate(model_inputs.mesh)


        # Margin location
        margin_x = 200e3
        # Linear smb
        smb =  p_var*Constant(1./pcs['spy'])*(1.0 - (2./margin_x)*x[0])
        model_inputs.smb = smb
        self.p = p
        self.p_var = p_var
        self.smb = smb


        ### Create model

        model = IceModel(model_inputs, "out", "out.hdf5")
        self.model = model


        ### Create object for doing differentiation of constrained problem

        self.vi_dif = VIDif(model.H, model.F_steady, model.J_steady, model.comm)
        # Generic vector to store dF_dp
        self.F_dp_d = assemble(self.model.F_steady)


        ### Define objective function

        data_inputs = ModelInputs('out/steady/steady_p2.h5')
        self.H_data = project(data_inputs.H0, self.model.V)
        self.g = (self.model.H - self.H_data)**2 * dx
        self.g_du_d = assemble(self.model.F_steady)


    def __assemble_F_dp__(self):
        assemble(diff(self.model.F_steady, self.p_var), tensor = self.F_dp_d)


    def __assemble_g_du__(self):
        assemble(derivative(self.g, self.model.H), tensor = self.g_du_d)


    def get_u_dp(self, p):
        self.run_steady(p)
        self.vi_dif.update_u()
        self.__assemble_F_dp__()
        self.vi_dif.__assemble_phi_dp__(self.F_dp_d.array())
        self.vi_dif.__assemble_u_dp__()

        from matplotlib import pyplot as plt
        plt.plot(self.vi_dif.u_dp_p.getArray())
        plt.show()


    def get_g_dp(self, p):
        self.run_steady(p)
        self.vi_dif.update_u()

        ### Compute u_dp
        self.__assemble_F_dp__()
        self.vi_dif.__assemble_phi_dp__(self.F_dp_d.array())
        self.vi_dif.__assemble_u_dp__()

        ### Compute g_du
        self.__assemble_g_du__()

        return np.dot(self.g_du_d.array(), self.vi_dif.u_dp_p.getArray())


    def run_steady(self, p):
        T = 135000.0 * pcs['spy']
        self.model.t = 0.
        dt = 10. * pcs['spy']
        self.p.assign(p)

        while self.model.t < T:
            self.model.step(dt)


    def get_g(self):
        return assemble(self.g)



model_inputs = ModelInputs('out/steady/steady_base2.h5')
model = ParamIceModel(model_inputs)


p = 1.0
g = 1e16
i = 0
while g > 6000.0 and i < 15:
    g_dp = model.get_g_dp(p)
    g = model.get_g()
    p = p - (g / g_dp)
    print p
    print g
    print
    i += 1

print
print
print p
print g
quit()
dp = 1e-6

model.run_steady(1.0)
g1 = model.get_g()

model.run_steady(1.0 + dp)
g2 = model.get_g()

g_dp = (g2 - g1) / dp
print g_dp



#print model.get_g_dp(1.)
