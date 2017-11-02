from dolfin import *
import numpy as np

L = 300e3
mesh = IntervalMesh(15, 0.0, L)

V = FunctionSpace(mesh, 'CG', 1)

B = Function(V)
H = Function(V)
H.interpolate(Constant(1000.0))
H0 = Function(V)

A = Constant(1e-16)
g = Constant(9.81)
rho_i = Constant(911.)
n = Constant(3.0)
spy = 31556926.0
#a = interpolate(Constant(0.3 / spy), V)

class Adot(Expression):
  def eval(self, value, x):
    value[0] = (1.0 - (1. / 100000.)*x[0]) / spy

dbc0 = DirichletBC(V,0.,lambda x,o: x[0] > 200000.0 and o)

a = project(Adot(degree = 1), V)
C = Constant((2.0*A*(rho_i*g)**n)/(n + 2.))
S = B + H
q = -C * H**(n+2.) * (dot(grad(S), grad(S)) + Constant(1e-16))**((n-1.)/2.) * grad(S)

dt = Constant(spy * 2.0)
v = TestFunction(V)
#F = (((H - H0)/dt)*v - dot(q, grad(v)) - a*v)*dx
F = (dot(q, grad(v)) - a*v)*dx

dH = TrialFunction(V)
JH = derivative(F, H, dH)


# Local vector length
N_local = len(H.vector().array())
# Unknown H as a petsc vector
H_p = as_backend_type(H.vector()).vec()
# PDE residual as petsc vector
F_p = H_p.duplicate()
# Diagonal matrices for phi_du
D_a_diag_p = H_p.duplicate()
D_b_diag_p = H_p.duplicate()
# Local arrays for setting D_a_diag_p, D_b_diag_p
D_a_diag_np = np.zeros(N_local)
D_b_diag_np = np.zeros(N_local)
# Generic matrix to store phi / du
F_du_d = assemble(JH)
phi_du_p = as_backend_type(F_du_d).mat()







def assemble_phi_du_p():
    ### Compute the diagonals of D_a and D_b

    # Set default diagonal values
    D_a_diag_np[:] = 0.0
    D_b_diag_np[:] = 1.0
    # Residual as numpy array
    F_np = assemble(F).array()
    # H value as numpy array()
    H_np = H.vector().array()
    # Diagonal denominator
    diag_denom = np.sqrt(H_np**2 + F_np**2)
    # Get indexes where the diagonal denominator is non-zero
    indexes = diag_denom > 0.0
    # Compute diagonal values
    D_a_diag_np[indexes] = 1.0 - H_np[indexes] / diag_denom[indexes]
    D_b_diag_np[indexes] = 1.0 - F_np[indexes] / diag_denom[indexes]
    # Set local values of numpy diagonal vectors
    D_a_diag_p.setArray(D_a_diag_np)
    D_b_diag_p.setArray(D_b_diag_np)


    ### Now compute phi / du as D_a + D_b * F_du

    assemble(JH, tensor = F_du_d)
    phi_du_p.diagonalScale(D_b_diag_p)
    phi_du_p.setDiagonal(D_b_diag_p, 2)




assemble_phi_du_p()

"""
quit()


v = Function(V)
v_vec = as_backend_type(v.vector()).vec()



#print v_vec
N = v_vec.getArray().size
v_vec.setArray(np.array(range(N)))

#print v_vec.getArray()

#J_mat.convert("dense")

J_mat.setDiagonal(v_vec)
J_mat.diagonalScale(v_vec)

J_mat.assemble()





print J_mat.getValuesCSR()


#quit()

#J_mat.diagonalScale()



print J_mat




"""


"""
print type(J_mat)
quit()

H_min = interpolate(Constant(0.0), V)
H_max = interpolate(Constant(1e10), V)
problem = NonlinearVariationalProblem(F, H, [dbc0], JH)
problem.set_bounds(H_min, H_max)
solver = NonlinearVariationalSolver(problem)

snes_params = {"nonlinear_solver": "snes",
                "snes_solver": {"linear_solver": "lu",
                "maximum_iterations": 100,
                "line_search": "basic",
                "report": True,
                "error_on_nonconvergence": True,
                "relative_tolerance" : 1e-100,
                "absolute_tolerance" : 1e-8}}

solver.parameters.update(snes_params)

solver.solve()
solver.solve()
solver.solve()
solver.solve()
plot(H, interactive = True)
quit()

def run_forward():
    t = 0.0
    T = 25000.0*spy

    while t <= T:
      print "t: " + (str(t / spy))
      (i, converged) = solver.solve()
      H0.assign(H)
      t += float(dt)
      #plot(H0, interactive = True)


import numpy as np
xs = interpolate(Expression('x[0]', degree = 1), V)

run_forward()
plot(H0, interactive = True)
np.savetxt('xs.txt', xs.vector().array())
np.savetxt('hs.txt', H.vector().array())
np.savetxt('qs.txt', project(sqrt(dot(q,q)), V).vector().array())"""
