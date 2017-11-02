import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

class UserContext(object):
    def __init__(self, da):
        self.da = da
        self.localX = da.createLocalVec()

    def formFunction(self, snes, X, F):

        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        f = self.da.getVecArray(F)

        (x0, x1), (y0, y1)= self.da.getRanges()
        hx = 1/float(x1-1)
        hy = 1/float(y1-1)

        for j in range(y0, y1):
            for i in range(x0, x1):
                if i == 0 or i == (x1-1) or j == 0 or j == (y1-1 ):
                    f[j,i] = x[j,i] - (1.0 - (2.0 * hx * i - 1.0) * (2.0 *
hx * i -1.0))
                else:
                    gradup    = (x[j+1, i] -x[j, i])/hy
                    graddown  = (x[j, i]   -x[j-1, i])/hy
                    gradright = (x[j, i+1] -x[j, i])/hx
                    gradleft  = (x[j, i]   -x[j, i-1])/hx

                    gradx = 0.5 * (x[j, i+1] - x[j, i-1])/hx
                    grady = 0.5 * (x[j+1, i] - x[j-1, i])/hy

                    coeffup = 1.0/np.sqrt(1.0 + gradup * gradup + gradx *
gradx)
                    coeffdown = 1.0/np.sqrt(1.0 + graddown * graddown +
gradx * gradx)

                    coeffleft  = 1.0/np.sqrt(1.0 + gradleft * gradleft +
grady * grady)
                    coeffright = 1.0/np.sqrt(1.0 + gradright * gradright +
grady * grady)

                    f[j, i] = (coeffup * gradup - coeffdown * graddown)*hx + (coeffright * gradright - coeffleft * gradleft) * hy


snes = PETSc.SNES().create()
da = PETSc.DMDA().create(dim=(-5, -5),
                        stencil_type = PETSc.DMDA.StencilType.STAR,
                        stencil_width =1,
                        setup=False)
da.setFromOptions()
da.setUp()

snes.setDM(da)

ctx = UserContext(da)
da.setAppCtx(ctx)

F = da.createGlobalVec()
snes.setFunction(ctx.formFunction, F)

snes.setFromOptions()

x = da.createGlobalVector()

snes.solve(None, x)
its = snes.getIterationNumber()
lits = snes.getLinearSolveIterations()

print "Number of SNES iterations = %d" % its
print "Number of Linear iterations = %d" % lits
