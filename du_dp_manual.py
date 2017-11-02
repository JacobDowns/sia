
#from model.ideal_inputs import *
from model.model_inputs import *
from model.ice_model import *
from model.support.physical_constants import *
from adjointer import *


### Steady state for p = 1
model_inputs = ModelInputs('out/steady/steady_base2.h5')


### Surface mass blance function
spy = 31556926.0

class SMB(Expression):

    def __init__(self, degree, p):
        self.p = p

    def eval(self, value, x):
        value[0] = (self.p / spy) * (1.0 - (1. / 100000.)*x[0])





model = IceModel(model_inputs, "out", "out.hdf5")
smb_form = SMB(degree = 1, p = 1.5)

#Setup adjointer object
adjointer = Adjointer(model)

#adjointer.model.smb.assign(project(smb_form, adjointer.model.V))
#adjointer.run_steady()
#adjointer.model.write_steady('steady_base2')

#quit()
du_dp = adjointer.u_dp_p.getArray() / 1.5

u1 = Function(adjointer.model.V)
u2 = Function(adjointer.model.V)
dp = 1e-7

"""
smb_form.p = 1.
adjointer.model.smb.assign(project(smb_form, adjointer.model.V))
#adjointer.run_steady()
adjointer.__assemble_F__()
adjointer.__assemble_phi__()
u1.vector()[:] = adjointer.phi_d.array()

smb_form.p = 1. + dp
adjointer.model.smb.assign(project(smb_form, adjointer.model.V))
adjointer.__assemble_F__()
adjointer.__assemble_phi__()
u2.vector()[:] = adjointer.phi_d.array()"""

smb_form.p = 1.5
adjointer.model.smb.assign(project(smb_form, adjointer.model.V))
adjointer.run_steady()
u1.vector()[:] = adjointer.model.H.vector().array()

smb_form.p = 1.5 + dp
adjointer.model.smb.assign(project(smb_form, adjointer.model.V))
adjointer.run_steady()
u2.vector()[:] = adjointer.model.H.vector().array()


print u1.vector().array()
print u2.vector().array()


du = (u2.vector().array() - u1.vector().array()) / dp









from matplotlib import pyplot as plt
plt.plot(du, 'r', linewidth = 1.1)
plt.plot(du_dp, 'k--', linewidth = 1.1)
plt.show()
#plt.ylim([-10., 10.])
