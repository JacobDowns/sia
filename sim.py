
#from model.ideal_inputs import *
from model.model_inputs import *
from model.ice_model import *
from model.support.physical_constants import *
from adjointer import *


### Steady state for p = 1
model_inputs = ModelInputs('out/steady/steady_base.h5')


### Surface mass blance function
spy = 31556926.0

class SMB(Expression):

    def __init__(self, degree, p):
        self.p = p

    def eval(self, value, x):
        value[0] = (self.p / spy) * (1.0 - (1. / 100000.)*x[0])

model_inputs.smb = SMB(degree = 1, p = 1.0)


model = IceModel(model_inputs, "out", "out.hdf5")


#Setup adjointer object
adjointer = Adjointer(model)
adjointer.run_steady()
