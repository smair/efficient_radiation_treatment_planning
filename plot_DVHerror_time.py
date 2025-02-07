import fire
import os.path

import numpy as np

import matplotlib.pylab as plt

from matplotlib import ticker

import CORT
import config


def main(case=None, loss='squared'):

    print(f'running the {case} case with {loss} loss')

    # set the directory where the result files (.npz) are located
    location = './results'

    repetitions = 50


    # set the subsample sizes per case
    M = []
    if case == 'Prostate':
        M = [1726, 3452, 5178, 6904, 17259, 34519, 51778, 69037, 138075, 172593, 207112, 276149, 345186, 414224, 483261, 517780, 552298, 621336]
    elif case == 'Liver':
        M = [4818, 9637, 14455, 19274, 48184, 96368, 144552, 192736, 385471, 481839, 578207, 770943, 963678, 1156414, 1349150, 1445518, 1541886, 1734621]
    elif case == 'HeadAndNeck':
        M = [1260, 1889, 2519, 6297, 12595, 18892, 25189, 50379, 62973, 75568, 100757, 125946, 151136, 176325, 188920, 201514, 226704]


    # set the under- and overdosing penalties
    if loss == 'absolute':
        if case == 'Prostate':
            w_BDY_over = 0.5
            w_OAR_over = 0.5
            w_PTV_over = 30.0
            w_PTV_under = 30.0
        elif case == 'Liver':
            w_BDY_over = 0.75
            w_OAR_over = 0.25
            w_PTV_over = 10.0
            w_PTV_under = 30.0
        elif case == 'HeadAndNeck':
            w_BDY_over = 0.5
            w_OAR_over = 0.75
            w_PTV_over = 30.0
            w_PTV_under = 30.0
    elif loss == 'squared':
        w_BDY_over = 1.0
        w_OAR_over = 1.0
        w_PTV_over = 4096.0
        w_PTV_under = 4096.0


    # set OAR, BODY, and PTV structures per case
    if case == 'Prostate':
        OAR_keys = ['Rectum', 'Bladder', 'Penile_bulb', 'Lt_femoral_head', 'Rt_femoral_head', 'Lymph_Nodes']
        BDY_keys = ['BODY']
        PTV_keys = ['PTV_68']
    elif case == 'Liver':
        OAR_keys = ['Heart', 'Liver', 'SpinalCord', 'Stomach']
        BDY_keys = ['Skin']
        PTV_keys = ['PTV']
    elif case == 'HeadAndNeck':
        OAR_keys = ['BRAIN_STEM', 'CHIASMA', 'SPINAL_CORD', 'PAROTID_LT', 'PAROTID_RT', 'LARYNX', 'LIPS']
        BDY_keys = ['External']
        PTV_keys = ['PTV70']


    cfg = config.get_config(case)
    data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BDY_threshold, OAR_structures, OAR_threshold, eta, steps = cfg

    # load full dose influence matrix
    D_full = CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))

    # set the indices for body (BDY), OAR, and PTV
    BDY_indices = OBJ[BODY_structure]['IDX']
    PTV_indices = OBJ[PTV_structure]['IDX']
    OAR_indices = np.unique(np.hstack([OBJ[OAR_structure]['IDX'] for OAR_structure in OAR_structures]))
    # fix the indices
    OAR_indices = np.setdiff1d(OAR_indices, PTV_indices)
    BDY_indices = np.setdiff1d(BDY_indices, np.union1d(PTV_indices, OAR_indices))

    assert len(np.intersect1d(BDY_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, BDY_indices)) == 0

    n_BDY = len(BDY_indices)
    n_OAR = len(OAR_indices)
    n_PTV = len(PTV_indices)

    n = n_BDY + n_OAR + n_PTV

    ###############################################################################

    # create a new OBJ object to use for the DVH plots
    OBJ_DVH = {}
    # inlude the PTV
    OBJ_DVH[PTV_structure] = dict(OBJ[PTV_structure])
    for structure in OAR_structures:
        # include the relevant OAR structures
        OBJ_DVH[structure] = dict(OBJ[structure])
        # subtract PTV voxels
        OBJ_DVH[structure]['IDX'] = np.setdiff1d(OBJ[structure]['IDX'], OBJ[PTV_structure]['IDX'])
    OBJ_DVH[BODY_structure] = dict(OBJ[BODY_structure])
    OBJ_DVH[BODY_structure]['IDX'] = BDY_indices

    ###############################################################################

    filename = f'{location}/result_{case}_full_0_0_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
    if loss == 'squared':
        filename = f'{location}/result_{case}_{loss}_full_0_0_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'

    npz = np.load(filename, allow_pickle=True)
    obj_full = npz['obj_MOSEK']
    time_full = npz['time_MOSEK']
    sol_MOSEK = npz['sol_MOSEK']

    ###############################################################################


    dose_max = 110
    dom = np.linspace(0, dose_max, 111)
    def get_DVH(x, keys):
        dose_ = D_full@x / PTV_dose * 100
        res_hist = []
        for key in keys: # OBJ_DVH
            hist = [np.sum(dose_[OBJ_DVH[key]['IDX']]>thres) for thres in dom]
            hist = np.array(hist) / OBJ_DVH[key]['IDX'].shape[0] * 100.0
            res_hist.append(hist)
        return np.array(res_hist)

    DVH_full_OAR = get_DVH(sol_MOSEK, OAR_keys)
    DVH_full_BDY = get_DVH(sol_MOSEK, BDY_keys)
    DVH_full_PTV = get_DVH(sol_MOSEK, PTV_keys)

    print('load uniform...')
    res_obj_uniform = []
    res_time_uniform = []
    res_DVHerror_uniform_OAR = []
    res_DVHerror_uniform_BDY = []
    res_DVHerror_uniform_PTV = []
    cachefile = f'{location}/cache_{case}_{loss}_uniform_{len(M)}.npz'
    if os.path.exists(cachefile):
        print(f'loading cache {cachefile}')
        npz = np.load(cachefile, allow_pickle=True)
        res_obj_uniform = npz['res_obj_uniform']
        res_time_uniform = npz['res_time_uniform']
        res_DVHerror_uniform_OAR = npz['res_DVHerror_uniform_OAR']
        res_DVHerror_uniform_BDY = npz['res_DVHerror_uniform_BDY']
        res_DVHerror_uniform_PTV = npz['res_DVHerror_uniform_PTV']
    else:
        for m in M:
            print(f'\t{m}')
            filename = f'{location}/result_{case}_uniform_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
            if loss == 'squared':
                filename = f'{location}/result_{case}_{loss}_uniform_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
            npz = np.load(filename, allow_pickle=True)
            obj_uniform = npz['res_subset_obj']
            time_uniform = npz['res_subset_time']
            # print(f'\t\texpect {repetitions} reps and found {len(npz["res_subset_x"])}')
            res_obj_uniform.append(obj_uniform)
            res_time_uniform.append(time_uniform)
            res_DVHerror_uniform_OAR.append([np.sum((get_DVH(x, OAR_keys)-DVH_full_OAR)**2)/len(OAR_keys) for x in npz['res_subset_x']]) # reps
            res_DVHerror_uniform_BDY.append([np.sum((get_DVH(x, BDY_keys)-DVH_full_BDY)**2)/len(BDY_keys) for x in npz['res_subset_x']]) # reps
            res_DVHerror_uniform_PTV.append([np.sum((get_DVH(x, PTV_keys)-DVH_full_PTV)**2)/len(PTV_keys) for x in npz['res_subset_x']]) # reps
        res_obj_uniform = np.array(res_obj_uniform)
        res_time_uniform = np.array(res_time_uniform)
        res_DVHerror_uniform_OAR = np.array(res_DVHerror_uniform_OAR)
        res_DVHerror_uniform_BDY = np.array(res_DVHerror_uniform_BDY)
        res_DVHerror_uniform_PTV = np.array(res_DVHerror_uniform_PTV)
        print(f'saving cache {cachefile}')
        np.savez(cachefile,
                 res_obj_uniform = res_obj_uniform,
                 res_time_uniform = res_time_uniform,
                 res_DVHerror_uniform_OAR = res_DVHerror_uniform_OAR,
                 res_DVHerror_uniform_BDY = res_DVHerror_uniform_BDY,
                 res_DVHerror_uniform_PTV = res_DVHerror_uniform_PTV)


    print('load gradnorm...')
    res_obj_gradnorm = []
    res_time_gradnorm = []
    res_DVHerror_gradnorm_OAR = []
    res_DVHerror_gradnorm_BDY = []
    res_DVHerror_gradnorm_PTV = []
    cachefile = f'{location}/cache_{case}_{loss}_gradnorm_{len(M)}.npz'
    if os.path.exists(cachefile):
        print(f'loading cache {cachefile}')
        npz = np.load(cachefile, allow_pickle=True)
        res_obj_gradnorm = npz['res_obj_gradnorm']
        res_time_gradnorm = npz['res_time_gradnorm']
        res_DVHerror_gradnorm_OAR = npz['res_DVHerror_gradnorm_OAR']
        res_DVHerror_gradnorm_BDY = npz['res_DVHerror_gradnorm_BDY']
        res_DVHerror_gradnorm_PTV = npz['res_DVHerror_gradnorm_PTV']
    else:
        for m in M:
            print(f'\t{m}')
            filename = f'{location}/result_{case}_gradnorm_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
            if loss == 'squared':
                filename = f'{location}/result_{case}_{loss}_gradnorm_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
            npz = np.load(filename, allow_pickle=True)
            obj_gradnorm = npz['res_subset_obj']
            time_gradnorm = npz['res_subset_time']
            # print(f'\t\texpect {repetitions} reps and found {len(npz["res_subset_x"])}')
            res_obj_gradnorm.append(obj_gradnorm)
            res_time_gradnorm.append(time_gradnorm)
            res_DVHerror_gradnorm_OAR.append([np.sum((get_DVH(x, OAR_keys)-DVH_full_OAR)**2)/len(OAR_keys) for x in npz['res_subset_x']]) # reps
            res_DVHerror_gradnorm_BDY.append([np.sum((get_DVH(x, BDY_keys)-DVH_full_BDY)**2)/len(BDY_keys) for x in npz['res_subset_x']]) # reps
            res_DVHerror_gradnorm_PTV.append([np.sum((get_DVH(x, PTV_keys)-DVH_full_PTV)**2)/len(PTV_keys) for x in npz['res_subset_x']]) # reps
        res_obj_uniform = np.array(res_obj_uniform)
        res_time_uniform = np.array(res_time_uniform)
        res_DVHerror_gradnorm_OAR = np.array(res_DVHerror_gradnorm_OAR)
        res_DVHerror_gradnorm_BDY = np.array(res_DVHerror_gradnorm_BDY)
        res_DVHerror_gradnorm_PTV = np.array(res_DVHerror_gradnorm_PTV)
        print(f'saving cache {cachefile}')
        np.savez(cachefile,
                 res_obj_gradnorm = res_obj_gradnorm,
                 res_time_gradnorm = res_time_gradnorm,
                 res_DVHerror_gradnorm_OAR = res_DVHerror_gradnorm_OAR,
                 res_DVHerror_gradnorm_BDY = res_DVHerror_gradnorm_BDY,
                 res_DVHerror_gradnorm_PTV = res_DVHerror_gradnorm_PTV)

    ###############################################################################



    fig, ax = plt.subplots(figsize=(6,3))
    ax.grid(which='both')
    ax.set_xlabel('time')
    ax.set_ylabel('DVH error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #
    #ax.plot(time_full, 0, 'k*', label='full')
    ax.axvline(time_full, color='k')
    #
    ax.plot(np.median(res_time_uniform, axis=1),
            np.median(res_DVHerror_uniform_PTV, axis=1), 'r', ls='-.')
    ax.plot(np.median(res_time_uniform, axis=1),
            np.median(res_DVHerror_uniform_OAR, axis=1), 'r', ls='--')
    ax.plot(np.median(res_time_uniform, axis=1),
            np.median(res_DVHerror_uniform_BDY, axis=1), 'r', ls=':')
    #
    ax.plot(np.median(res_time_gradnorm, axis=1),
            np.median(res_DVHerror_gradnorm_PTV, axis=1), 'g', ls='-.')
    ax.plot(np.median(res_time_gradnorm, axis=1),
            np.median(res_DVHerror_gradnorm_OAR, axis=1), 'g', ls='--')
    ax.plot(np.median(res_time_gradnorm, axis=1),
            np.median(res_DVHerror_gradnorm_BDY, axis=1), 'g', ls=':')
    #
    ax.plot([], [], 'k', ls='-.', label='PTV')
    ax.plot([], [], 'k', ls='--', label='OAR')
    ax.plot([], [], 'k', ls=':', label='BDY')
    # ax.plot([], [], 'k', ls='-', label='full')
    ax.plot([], [], 'r', ls='-', label='uniform')
    ax.plot([], [], 'g', ls='-', label='gradnorm')
    #
    ax.legend()


    fig.savefig(f'/tmp/DVHerror_time_{case}_{loss}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/DVHerror_time_{case}_{loss}.png', dpi=300, transparent=True, bbox_inches='tight')


    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################


    fig, ax = plt.subplots(figsize=(6,3))
    ax.grid()
    ax.set_yscale('log')
    ax.set_xlabel('Subsample size $m$')
    ax.set_ylabel('Objective function value')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    #
    # Define function and its inverse
    f = lambda x: np.array(x)/n*100
    g = lambda x: np.array(x)/100*n
    ax2 = ax.secondary_xaxis('top', functions=(f,g))
    ax2.set_xlabel('Subsample size [%]')
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax2.locator_params(axis='x', nbins=10)
    #
    ax.axhline(obj_full, ls='--', color='k', label='full')
    #
    median = np.median(res_obj_uniform, axis=1)
    quantile_upper = np.quantile(res_obj_uniform, q=0.75, axis=1)
    quantile_lower = np.quantile(res_obj_uniform, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='r', alpha=0.25)
    ax.plot(M, median, 'r-', label='uniform')
    #
    median = np.median(res_obj_gradnorm, axis=1)
    quantile_upper = np.quantile(res_obj_gradnorm, q=0.75, axis=1)
    quantile_lower = np.quantile(res_obj_gradnorm, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='g', alpha=0.25)
    ax.plot(M, median, 'g-', label='gradnorm')
    #
    ax.locator_params(axis='x', nbins=6)
    #
    ax.legend()
    fig.savefig(f'/tmp/obj_subsample_{case}_{loss}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/obj_subsample_{case}_{loss}.png', dpi=300, transparent=True, bbox_inches='tight')

    ###############################################################################


    fig, ax = plt.subplots(figsize=(6,3))
    ax.grid()
    ax.set_xlabel('Subsample size $m$')
    ax.set_ylabel('Relative objective function value [%]')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    #
    # Define function and its inverse
    f = lambda x: np.array(x)/n*100
    g = lambda x: np.array(x)/100*n
    ax2 = ax.secondary_xaxis('top', functions=(f,g))
    ax2.set_xlabel('Subsample size [%]')
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    axins = ax.inset_axes([0.25, 0.30, 0.45, 0.5]) # [x0, y0, width, height]
    axins.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    axins.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax3 = axins.secondary_xaxis('top', functions=(f,g))
    ax3.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    #
    ax.axhline(100, ls='--', color='k', label='full')
    axins.axhline(100, ls='--', color='k')
    #
    median = np.median(res_obj_uniform/obj_full*100, axis=1)
    quantile_upper = np.quantile(res_obj_uniform/obj_full*100, q=0.75, axis=1)
    quantile_lower = np.quantile(res_obj_uniform/obj_full*100, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='r', alpha=0.25)
    ax.plot(M, median, 'r-', label='uniform')
    #
    median = np.median(res_obj_gradnorm/obj_full*100, axis=1)
    quantile_upper = np.quantile(res_obj_gradnorm/obj_full*100, q=0.75, axis=1)
    quantile_lower = np.quantile(res_obj_gradnorm/obj_full*100, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='g', alpha=0.25)
    ax.plot(M, median, 'g-', label='gradnorm')
    axins.plot(M, median, 'g-', )
    #
    if case == 'Prostate':
        ax.set_ylim(95, 200)
        ax.locator_params(axis='x', nbins=10)
    elif case == 'Liver':
        ax.set_ylim(95, 200)
        ax.locator_params(axis='x', nbins=6)
    elif case == 'HeadAndNeck':
        ax.set_ylim(95, 400)
    #
    ax.legend()

    if case == 'Prostate':
        x1, x2, y1, y2 = 0, 60_000, 99.5, 105
        axins.set_yticks([100,101,102,103,104])
        axins.locator_params(axis='x', nbins=5)
    elif case == 'Liver':
        x1, x2, y1, y2 = 0, 90_000, 99.5, 105
        axins.set_yticks([100,101,102,103,104])
        axins.locator_params(axis='x', nbins=4)
    elif case == 'HeadAndNeck':
        x1, x2, y1, y2 = 20_000, 60_000, 99.5, 110
        axins.set_yticks([100,102,104,106,108])
        axins.locator_params(axis='x', nbins=5)
    axins.grid()
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticklabels([])
    #axins.set_yticklabels([])
    #axins.locator_params(axis='x', nbins=5)
    ax.indicate_inset_zoom(axins, edgecolor='black')


    fig.savefig(f'/tmp/relobj_subsample_{case}_{loss}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/relobj_subsample_{case}_{loss}.png', dpi=300, transparent=True, bbox_inches='tight')

    ###############################################################################

    fig, ax = plt.subplots(figsize=(6,3))
    ax.grid()
    ax.set_yscale('log')
    ax.set_xlabel('Subsample size $m$')
    ax.set_ylabel('DVH error')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    #
    # Define function and its inverse
    f = lambda x: np.array(x)/n*100
    g = lambda x: np.array(x)/100*n
    ax2 = ax.secondary_xaxis('top', functions=(f,g))
    ax2.set_xlabel('Subsample size [%]')
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    #
    median = np.median(res_DVHerror_uniform_PTV, axis=1)
    quantile_upper = np.quantile(res_DVHerror_uniform_PTV, q=0.75, axis=1)
    quantile_lower = np.quantile(res_DVHerror_uniform_PTV, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='r', alpha=0.25)
    ax.plot(M, median, 'r-.')
    #
    median = np.median(res_DVHerror_uniform_OAR, axis=1)
    quantile_upper = np.quantile(res_DVHerror_uniform_OAR, q=0.75, axis=1)
    quantile_lower = np.quantile(res_DVHerror_uniform_OAR, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='r', alpha=0.25)
    ax.plot(M, median, 'r--')
    #
    median = np.median(res_DVHerror_uniform_BDY, axis=1)
    quantile_upper = np.quantile(res_DVHerror_uniform_BDY, q=0.75, axis=1)
    quantile_lower = np.quantile(res_DVHerror_uniform_BDY, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='r', alpha=0.25)
    ax.plot(M, median, 'r:')
    #
    median = np.median(res_DVHerror_gradnorm_PTV, axis=1)
    quantile_upper = np.quantile(res_DVHerror_gradnorm_PTV, q=0.75, axis=1)
    quantile_lower = np.quantile(res_DVHerror_gradnorm_PTV, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='g', alpha=0.25)
    ax.plot(M, median, 'g-.')
    #
    median = np.median(res_DVHerror_gradnorm_OAR, axis=1)
    quantile_upper = np.quantile(res_DVHerror_gradnorm_OAR, q=0.75, axis=1)
    quantile_lower = np.quantile(res_DVHerror_gradnorm_OAR, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='g', alpha=0.25)
    ax.plot(M, median, 'g--')
    #
    median = np.median(res_DVHerror_gradnorm_BDY, axis=1)
    quantile_upper = np.quantile(res_DVHerror_gradnorm_BDY, q=0.75, axis=1)
    quantile_lower = np.quantile(res_DVHerror_gradnorm_BDY, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='g', alpha=0.25)
    ax.plot(M, median, 'g:')
    #
    ax.plot([], [], 'k', ls='-.', label='PTV')
    ax.plot([], [], 'k', ls='--', label='OAR')
    ax.plot([], [], 'k', ls=':', label='BDY')
    # ax.plot([], [], 'k', ls='-', label='full')
    ax.plot([], [], 'r', ls='-', label='uniform')
    ax.plot([], [], 'g', ls='-', label='gradnorm')
    #
    ax.locator_params(axis='x', nbins=6)
    #
    ax.legend()
    fig.savefig(f'/tmp/DVHerror_subsample_{case}_{loss}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/DVHerror_subsample_{case}_{loss}.png', dpi=300, transparent=True, bbox_inches='tight')


    ###############################################################################


    fig, ax = plt.subplots(figsize=(6,3))
    ax.grid()
    ax.set_xlabel('Subsample size $m$')
    ax.set_ylabel('Relative time')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    #
    # Define function and its inverse
    f = lambda x: np.array(x)/n*100
    g = lambda x: np.array(x)/100*n
    ax2 = ax.secondary_xaxis('top', functions=(f,g))
    ax2.set_xlabel('Subsample size [%]')
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax2.locator_params(axis='x', nbins=10)

    if case == 'Prostate' or case == 'Liver':
        axins = ax.inset_axes([0.12, 0.51, 0.35, 0.3]) # [x0, y0, width, height]
        axins.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        axins.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
        ax3 = axins.secondary_xaxis('top', functions=(f,g))
        ax3.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
    #
    ax.axhline(time_full/time_full, ls='--', color='k', label='full')
    #
    median = np.median(res_time_uniform/time_full, axis=1)
    quantile_upper = np.quantile(res_time_uniform/time_full, q=0.75, axis=1)
    quantile_lower = np.quantile(res_time_uniform/time_full, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='r', alpha=0.25)
    ax.plot(M, median, 'r-', label='uniform')
    if case == 'Prostate' or case == 'Liver':
        axins.plot(M, median, 'r-')
    #
    median = np.median(res_time_gradnorm/time_full, axis=1)
    quantile_upper = np.quantile(res_time_gradnorm/time_full, q=0.75, axis=1)
    quantile_lower = np.quantile(res_time_gradnorm/time_full, q=0.25, axis=1)
    ax.fill_between(M, quantile_upper, quantile_lower, color='g', alpha=0.25)
    ax.plot(M, median, 'g-', label='gradnorm')
    if case == 'Prostate' or case == 'Liver':
        axins.plot(M, median, 'g-')
    #
    ax.locator_params(axis='x', nbins=7)
    ax.locator_params(axis='y', nbins=11)
    #
    ax.legend()

    if case == 'Prostate':
        x1, x2, y1, y2 = 1000, 60_000, 0.0, 0.1
        axins.set_yticks([0, 0.05, 0.07, 0.10])
        ax3.set_xticks([1.0, 5.0, 7.5])
    elif case == 'Liver':
        x1, x2, y1, y2 = 4000, 25_000, 0.0, 0.04
        # axins.set_yticks([0, 0.05, 0.07, 0.10])
        ax3.set_xticks([0.5, 1.0])
        ax.locator_params(axis='x', nbins=6)
        axins.set_xticks([5_000, 20_000])
        axins.set_yticks([0.01, 0.02, 0.03, 0.04])

    if case == 'Prostate' or case == 'Liver':
        axins.grid()
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        #axins.set_xticklabels([])
        #axins.set_yticklabels([])
        # axins.locator_params(axis='x', nbins=5)
        ax.indicate_inset_zoom(axins, edgecolor='black')

    fig.savefig(f'/tmp/reltime_subsample_{case}_{loss}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/reltime_subsample_{case}_{loss}.png', dpi=300, transparent=True, bbox_inches='tight')


    ###############################################################################


    fig, ax = plt.subplots(figsize=(6,3))
    ax.grid(which='both')
    ax.set_xlabel('Relative time')
    ax.set_ylabel('Relative objective function value')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #
    ax.plot(np.median(res_time_uniform/time_full, axis=1),
            np.median(res_obj_uniform/obj_full, axis=1), 'ro-', label='uniform')
    #
    ax.plot(np.median(res_time_gradnorm/time_full, axis=1),
            np.median(res_obj_gradnorm/obj_full, axis=1), 'go-', label='gradnorm')
    #
    ax.plot(time_full/time_full, obj_full/obj_full, 'k*', label='full')
    #
    ax.legend()
    fig.savefig(f'/tmp/relobj_reltime_{case}_{loss}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/relobj_reltime_{case}_{loss}.png', dpi=300, transparent=True, bbox_inches='tight')




if __name__ == '__main__':
  fire.Fire(main)

