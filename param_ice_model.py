#import h5py
from dolfin import *
from model.model_inputs import *
from model.ice_model import *
from model.support.physical_constants import *
from vi_dif import *
from petsc4py import PETSc
from adjointer import *

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




    def __assemble_F_dp__(self):
        assemble(diff(self.model.F_steady, self.p_var), tensor = self.F_dp_d)


    def get_H_dp(self, p):
        self.run_steady(p)
        self.vi_dif.update_u()
        self.__assemble_F_dp__()
        self.vi_dif.__assemble_phi_dp__(self.F_dp_d.array())
        self.vi_dif.__assemble_u_dp__()

        from matplotlib import pyplot as plt
        plt.plot(self.vi_dif.u_dp_p.getArray())
        plt.show()


    def run_steady(self, p):
        T = 135000.0 * pcs['spy']
        self.model.t = 0.
        dt = 10. * pcs['spy']
        self.p.assign(p)

        while self.model.t < T:
            self.model.step(dt)



model_inputs = ModelInputs('out/steady/steady_base2.h5')
model = ParamIceModel(model_inputs)
model.get_H_dp(1.5)
