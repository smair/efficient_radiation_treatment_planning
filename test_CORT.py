import numpy as np
import cvxpy as cp
import scipy
import CORT

################################################################################

# start with the Prostate case

# specify the data path
data_path = './data/CORT/Prostate/'

# the gantry and couch angles to consider
gantry_angles = [0, 72, 144, 216, 288]
couch_angles = [0, 0, 0, 0, 0]


# define the structures
OBJ = {
    'PTV_68':{},
    'Rectum':{},
    'BODY':{},
    'Bladder':{}
}


D = CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))


################################################################################

"""
min(mean Rectum + 0.6 mean Bladder + 0.6 mean BODY)
PTV_68 >= 1
x <= 50
"""

D_rectum_mean = D[OBJ['Rectum']['IDX']].mean(0)
D_bladder_mean = D[OBJ['Bladder']['IDX']].mean(0)
D_body_mean = D[OBJ['BODY']['IDX']].mean(0)

c = D_rectum_mean + 0.6*D_bladder_mean + 0.6*D_body_mean

"""
min c^T x
Aub x <= bub
Aeq x  = beq
lb <= x <= ub
"""

A_ub = -D[OBJ['PTV_68']['IDX']]
b_ub = -1.0*np.ones(A_ub.shape[0])
bounds = ((0.0, 50.0),)*A_ub.shape[1]

sol = scipy.optimize.linprog(c,
                             A_ub=A_ub, b_ub=b_ub,
                             A_eq=None, b_eq=None,
                             bounds=bounds,
                             options={'disp':True},
                             method='highs-ipm')

assert np.allclose(np.round(D_rectum_mean @ sol.x, 4).item(), 0.2842)
assert np.allclose(np.round(D_bladder_mean @ sol.x, 4).item(), 0.4035)
assert np.allclose(np.round(D_body_mean @ sol.x, 4).item(), 0.0905)

print(D_rectum_mean @ sol.x)
print(D_bladder_mean @ sol.x)
print(D_body_mean @ sol.x)



################################################################################


# Define and solve the CVXPY problem.
x = cp.Variable(D.shape[1])
prob = cp.Problem(cp.Minimize(c@x),
                  [A_ub @ x <= b_ub,
                  0.0 <= x,
                  x <= 50.0])
prob.solve(verbose=True)

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)


assert np.allclose(np.round(D_rectum_mean @ x.value, 4).item(), 0.2842)
assert np.allclose(np.round(D_bladder_mean @ x.value, 4).item(), 0.4035)
assert np.allclose(np.round(D_body_mean @ x.value, 4).item(), 0.0905)

print(D_rectum_mean @ x.value)
print(D_bladder_mean @ x.value)
print(D_body_mean @ x.value)


################################################################################
################################################################################

# now the Liver case

# specify the data path
data_path = './data/CORT/Liver/'

# the gantry and couch angles to consider
gantry_angles = [58, 106, 212, 328, 216, 226, 296]
couch_angles = [0, 0, 0, 0, 32, -13, 17]


# define the structures
OBJ = {
    'PTV':{},
    'Liver':{},
    'Heart':{},
    'entrance':{}
}


D = CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))


################################################################################

"""
min(mean Liver + mean Heart + 0.6 mean entrance)
PTV >= 1
x <= 25
"""

D_liver_mean = D[OBJ['Liver']['IDX']].mean(0)
D_heart_mean = D[OBJ['Heart']['IDX']].mean(0)
D_entrance_mean = D[OBJ['entrance']['IDX']].mean(0)

c = D_liver_mean + D_heart_mean + 0.6*D_entrance_mean

"""
min c^T x
Aub x <= bub
Aeq x  = beq
lb <= x <= ub
"""

A_ub = -D[OBJ['PTV']['IDX']]
b_ub = -1.0*np.ones(A_ub.shape[0])
bounds = ((0.0, 25.0),)*A_ub.shape[1]

sol = scipy.optimize.linprog(c,
                             A_ub=A_ub, b_ub=b_ub,
                             A_eq=None, b_eq=None,
                             bounds=bounds,
                             options={'disp':True},
                             method='highs-ipm')

assert np.allclose(np.round(D_liver_mean @ sol.x, 4).item(), 0.1771)
assert np.allclose(np.round(D_heart_mean @ sol.x, 4).item(), 0.1258)
assert np.allclose(np.round(D_entrance_mean @ sol.x, 4).item(), 0.0186)

print(D_liver_mean @ sol.x)
print(D_heart_mean @ sol.x)
print(D_entrance_mean @ sol.x)


################################################################################
################################################################################

# now the Head & Neck case

# specify the data path
data_path = './data/CORT/HeadAndNeck/'

# the gantry and couch angles to consider
gantry_angles = [0, 72, 144, 216, 288, 180, 220, 260, 300, 340]
couch_angles = [0, 0, 0, 0, 0, 20, 20, 20, 20, 20]


# define the structures
OBJ = {
    'PTV56':{},
    'PTV63':{},
    'PTV70':{},
    'PAROTID_LT':{},
    'PAROTID_RT':{},
    'SPINAL_CORD':{},
    'BRAIN_STEM':{}
}


D = CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))

# OBJ['PTV'] = {'IDX':np.union1d(np.union1d(OBJ['PTV56']['IDX'], OBJ['PTV63']['IDX']), OBJ['PTV70']['IDX'])}
OBJ['PTV'] = {'IDX':np.hstack((OBJ['PTV56']['IDX'], OBJ['PTV63']['IDX'], OBJ['PTV70']['IDX']))}


################################################################################

"""
min(mean PAROTID_LT + mean PAROTID_RT)
PTV >= 1
SPINAL_CORD <= 0.5
BRAIN_STEM <= 0.5
x <= 25
"""

D_parotid_lt_mean = D[OBJ['PAROTID_LT']['IDX']].mean(0)
D_parotid_rt_mean = D[OBJ['PAROTID_RT']['IDX']].mean(0)

c = D_parotid_lt_mean + D_parotid_rt_mean

"""
min c^T x
Aub x <= bub
Aeq x  = beq
lb <= x <= ub
"""

A_ub = scipy.sparse.vstack((-D[OBJ['PTV']['IDX']],
                            D[OBJ['SPINAL_CORD']['IDX']],
                            D[OBJ['BRAIN_STEM']['IDX']]))
b_ub = np.hstack((-1.0*np.ones(len(OBJ['PTV']['IDX'])),
                  0.5*np.ones(len(OBJ['SPINAL_CORD']['IDX'])),
                  0.5*np.ones(len(OBJ['BRAIN_STEM']['IDX']))))
bounds = ((0.0, 25.0),)*A_ub.shape[1]

sol = scipy.optimize.linprog(c,
                             A_ub=A_ub, b_ub=b_ub,
                             A_eq=None, b_eq=None,
                             bounds=bounds,
                             options={'disp':True},
                             method='highs-ipm')

# note: this one is slightly off
assert np.allclose(np.round(D_parotid_lt_mean @ sol.x, 4).item(), 0.4959)
assert np.allclose(np.round(D_parotid_rt_mean @ sol.x, 4).item(), 0.3437)

print(D_parotid_lt_mean @ sol.x)
print(D_parotid_rt_mean @ sol.x)

