from dolfin import *

L = 300e3
mesh = IntervalMesh(12, 0.0, L)
V = FunctionSpace(mesh, 'CG', 1)
spy = 31556926.0

class Adot(Expression):
  def eval(self, value, x):
    value[0] = (1.0 - (1. / 100000.)*x[0]) / spy

adot = project(Adot(degree = 1), V)
B = Function(V)

class H(Expression):
  def eval(self, value, x):
    L_m = 150e3
    x = x[0] / L_m
    H0 = 2000.0

    value[0] = 0.0
    if x < 1.0:
        value[0] = H0*sqrt(1.0 - x)

H = project(H(degree = 1), V)

S = Function(V)
S.assign(H)

plot(H, interactive = True)
