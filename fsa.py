#import h5py
from dolfin import *
from model.model_inputs import *
from model.ice_model import *
from model.support.physical_constants import *
from vi_dif import *
from petsc4py import PETSc
from adjointer import *

"""
Object for doing forward sensitivity analysis.
"""

class FSA(object):

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
        self.p_var = p_var
        self.smb = smb


        ### Create model

        self.model = IceModel(model_inputs, "out", "out.hdf5")


        ### Create object for doing differentiation of constrained problem

        self.vi_dif = VIDif(self.model)

        quit()

        # Generic vector to store dF_dp
        self.F_dp_d = assemble(self.model.F_steady)
        self.__assemble_F_dp__()
        #plot(project(smb, model_inputs.V), interactive = True)



    def __assemble_F_dp__(self):
        assemble(diff(self.model.F_steady, self.p_var), tensor = self.F_dp_d)





model_inputs = ModelInputs('out/steady/steady_base2.h5')
fsa = FSA(model_inputs)
