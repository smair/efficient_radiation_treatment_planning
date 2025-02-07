import cvxpy as cp
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from time import time


def plot_DVH(D_full, x, OBJ, PTV_dose, title='', save_location='',
             dashed_x=None, dashed_legend='', dotted_x=None, dotted_legend=''):
    dose_ = D_full@x / PTV_dose * 100

    dose_max = 110
    dom = np.linspace(0, dose_max, 111)
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(title)>0:
        ax.set_title(title)
    ax.set_xlabel('Dose [%]')
    ax.set_ylabel('Volume [%]')
    ax.set_xticks(np.arange(0, dose_max+1, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticks(np.arange(0, dose_max+1, 5), minor=True)
    ax.set_yticks(np.arange(0, 101, 5), minor=True)
    ax.grid(which='major')
    ax.grid(which='minor', alpha=0.25)
    ax.axvline(100, ls='-', color='k', alpha=0.99)
    #
    for key in OBJ:
        hist = [np.sum(dose_[OBJ[key]['IDX']]>thres) for thres in dom]
        hist = np.array(hist) / OBJ[key]['IDX'].shape[0] * 100.0
        ax.plot(dom, hist, ls='-', color=OBJ[key]['COLOR'], label=key)

    if type(dashed_x) == np.ndarray:
        dose_ = D_full@dashed_x / PTV_dose * 100
        for key in OBJ:
            hist = [np.sum(dose_[OBJ[key]['IDX']]>thres) for thres in dom]
            hist = np.array(hist) / OBJ[key]['IDX'].shape[0] * 100.0
            ax.plot(dom, hist, ls='--', color=OBJ[key]['COLOR'])
        if len(dashed_legend)>0:
            ax.plot([], [], ls='--', color='black', label=dashed_legend)

    if type(dotted_x) == np.ndarray:
        dose_ = D_full@dotted_x / PTV_dose * 100
        for key in OBJ:
            hist = [np.sum(dose_[OBJ[key]['IDX']]>thres) for thres in dom]
            hist = np.array(hist) / OBJ[key]['IDX'].shape[0] * 100.0
            ax.plot(dom, hist, ls=':', color=OBJ[key]['COLOR'])
        if len(dotted_legend)>0:
            ax.plot([], [], ls=':', color='black', label=dotted_legend)

    ax.legend()

    if len(save_location)>0:
        fig.savefig(f'{save_location}.pdf', dpi=300, transparent=True, bbox_inches='tight')
        fig.savefig(f'{save_location}.png', dpi=300, transparent=True, bbox_inches='tight')


def compute_radiation_plan(D_BDY, D_OAR, D_PTV, BDY_threshold, OAR_threshold,
                           target_dose_PTV, weights_BDY_over, weights_OAR_over,
                           weights_PTV_over, weights_PTV_under,
                           loss='absolute', verbose=True):

    assert loss in ['absolute', 'squared']
    assert D_BDY.shape[0] == weights_BDY_over.shape[0]
    assert D_OAR.shape[0] == weights_OAR_over.shape[0]
    assert D_PTV.shape[0] == weights_PTV_over.shape[0]
    assert D_PTV.shape[0] == weights_PTV_under.shape[0]

    d = D_OAR.shape[1]

    x = cp.Variable(d)

    def identity(z):
        return z

    modifier = identity
    if loss == 'squared':
        modifier = cp.square

    prob = cp.Problem(cp.Minimize(  cp.sum(cp.multiply(weights_BDY_over, modifier(cp.maximum(D_BDY@x-BDY_threshold,   0))))
                                  + cp.sum(cp.multiply(weights_OAR_over, modifier(cp.maximum(D_OAR@x-OAR_threshold,   0))))
                                  + cp.sum(cp.multiply(weights_PTV_over, modifier(cp.maximum(D_PTV@x-target_dose_PTV, 0))))
                                  + cp.sum(cp.multiply(weights_PTV_under,modifier(cp.maximum(target_dose_PTV-D_PTV@x, 0))))),
                      [x >= 0.0])

    t1 = time()
    prob.solve(verbose=verbose,
               mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':'MSK_SOLVE_DUAL'},
               solver='MOSEK')
    t2 = time()

    return x.value.copy(), t2-t1


def objective(D_BDY, D_OAR, D_PTV, BDY_threshold, OAR_threshold, target_dose_PTV,
              weights_BDY_over, weights_OAR_over, weights_PTV_over,
              weights_PTV_under, x, loss='absolute'):

    assert loss in ['absolute', 'squared']
    assert D_BDY.shape[0] == weights_BDY_over.shape[0]
    assert D_OAR.shape[0] == weights_OAR_over.shape[0]
    assert D_PTV.shape[0] == weights_PTV_over.shape[0]
    assert D_PTV.shape[0] == weights_PTV_under.shape[0]

    def identity(z):
        return z

    modifier = identity
    if loss == 'squared':
        modifier = np.square

    obj  = np.sum(np.multiply(weights_BDY_over, modifier(np.clip(D_BDY@x-BDY_threshold,   a_min=0, a_max=None))))
    obj += np.sum(np.multiply(weights_OAR_over, modifier(np.clip(D_OAR@x-OAR_threshold,   a_min=0, a_max=None))))
    obj += np.sum(np.multiply(weights_PTV_over, modifier(np.clip(D_PTV@x-target_dose_PTV, a_min=0, a_max=None))))
    obj += np.sum(np.multiply(weights_PTV_under,modifier(np.clip(target_dose_PTV-D_PTV@x, a_min=0, a_max=None))))

    return obj


def compute_subset(D, weights, m, seed, scores):

    assert D.shape[0] == weights.shape[0]
    assert D.shape[0] == scores.shape[0]

    n = D.shape[0]

    # sampling probabilities
    q = scores / scores.sum()

    np.random.seed(seed)
    sample_idx = np.random.choice(n, size=m, p=q)

    """
    The re-weighting is done as follows:
        new_weight = old_weight / (m*q)
    """
    new_weights  = weights / (m*q)

    # we need to return the indices and not the smaller arrays
    # because we need to split the structures
    return sample_idx, new_weights


def compute_scores(D, target_dose, weights, eta, steps):

    if eta == -1.0:
        print('compute_scores(): No eta given, computing learning rate...')
        wD = sp.sparse.diags(np.sqrt(weights)) * D
        _,s,_ = sp.sparse.linalg.svds(2*wD.T@wD)
        # L = s.max()
        eta = 2/(s.min() + s.max())

    x_hist = []
    loss_hist = []
    score_residual_hist = []
    score_gradnorm_hist = []

    print(f'Running PGD for {steps} steps with a learning rate of eta={eta}')

    t_start = time()
    time_PGD = []

    # initialize x with zeros
    x = np.zeros(D.shape[1])
    for i in range(steps):
        # compute the current doses
        dose = D@x
        # compute the residual
        res = dose - target_dose
        # use square loss and the weighing
        loss = np.sum(np.multiply(weights, res**2))
        loss_hist.append(loss)
        print(f'loss={loss} max_dose={dose.max()}')
        # compute the gradient per data point
        grad_per_point = D.multiply(2*np.multiply(weights, res).reshape(-1, 1))
        grad = np.array(grad_per_point.sum(0)).reshape(-1)
        # gradient descent step
        x = x - eta*grad
        # projection to x>=0
        x[x<0] = 0
        # remember x
        x_hist.append(x)
        # compute the scores
        score_residual = np.abs(res)
        score_residual_hist.append(score_residual)
        score_gradnorm = sp.sparse.linalg.norm(grad_per_point, ord=2, axis=1)
        score_gradnorm_hist.append(score_gradnorm)
        # track the time
        time_PGD.append(time())

    time_PGD = np.array(time_PGD) - t_start

    return x_hist, loss_hist, score_residual_hist, score_gradnorm_hist, time_PGD

