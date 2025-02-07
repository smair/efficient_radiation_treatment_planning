import fire
import os.path

import numpy as np
import scipy as sp
import matplotlib.pylab as plt

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
        M = [690, 1726, 3452, 5178, 6904, 17259, 34519, 51778, 69037, 138075, 172593, 207112, 276149, 345186, 414224, 483261, 517780, 552298, 621336]
    elif case == 'Liver':
        M = [1927, 4818, 9637, 14455, 19274, 48184, 96368, 144552, 192736, 385471, 481839, 578207, 770943, 963678, 1156414, 1349150, 1445518, 1541886, 1734621]
    elif case == 'HeadAndNeck':
        M = [252, 630, 1260, 1889, 2519, 6297, 12595, 18892, 25189, 50379, 62973, 75568, 100757, 125946, 151136, 176325, 188920, 201514, 226704]


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


    # specify the target dose
    # initialize the target dose to zero
    target_dose = np.zeros(D_full.shape[0])
    # set the PTV dose
    target_dose[OBJ[PTV_structure]['IDX']] = PTV_dose
    # set the OAR target dose to a threshold to prevent penalizing small violations
    target_dose[OAR_indices] = OAR_threshold
    # set the BDY target dose to a threshold to prevent penalizing small violations
    target_dose[BDY_indices] = BDY_threshold


    # set D and overwrite target_dose to only consider BODY, OAR, and PTV voxels,
    # i.e., skip all other voxels outside the actual BODY
    D = sp.sparse.vstack((D_full[BDY_indices],
                          D_full[OAR_indices],
                          D_full[PTV_indices]))
    target_dose = np.hstack((target_dose[BDY_indices],
                             target_dose[OAR_indices],
                             target_dose[PTV_indices]))


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
    sol_MOSEK = npz['sol_MOSEK']

    ###############################################################################

    dose_max = 110
    dom = np.linspace(0, dose_max, 110)


    for score_method in ['gradnorm', 'uniform']:
        for m in M:

            filename = f'{location}/result_{case}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
            if loss == 'squared':
                filename = f'{location}/result_{case}_{loss}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
            print(f'current file: {filename}')
            if not os.path.exists(filename):
                print(f'file {filename} does not exist')
                continue
            if os.path.exists(f'/tmp/DVHdist_{case}_{loss}_{score_method}_m{m}.pdf'):
                print('image already exists; delete to re-create')
                continue
            npz = np.load(filename, allow_pickle=True)
            tmp = {}
            for key in OBJ_DVH:
                tmp[key] = []
            if 'res_subset_x' not in npz:
                print(f'res_subset_x not in {filename}')
                continue
            for x in npz['res_subset_x']:
                # let dose_ be percentages of the PTV_dose
                dose_ = D_full@x / PTV_dose * 100
                for key in OBJ_DVH:
                    hist = [np.sum(dose_[OBJ_DVH[key]['IDX']]>thres) for thres in dom]
                    hist = np.array(hist) / OBJ_DVH[key]['IDX'].shape[0] * 100.0
                    tmp[key].append(hist)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(rf'$m$={m:,}   ($\approx${m/D.shape[0]*100:.2f}%)')

            ax.set_xlabel('Dose [%]')
            ax.set_ylabel('Volume [%]')
            ax.set_xticks(np.arange(0, dose_max+1, 10))
            ax.set_yticks(np.arange(0, 101, 10))
            ax.set_xticks(np.arange(0, dose_max+1, 5), minor=True)
            ax.set_yticks(np.arange(0, 101, 5), minor=True)
            ax.grid(which='major')
            ax.grid(which='minor', alpha=0.25)
            # ax.axvline(PTV_dose, ls='-', color='k', alpha=0.99)
            ax.axvline(100, ls='-', color='k', alpha=0.99)
            #
            for key in OBJ_DVH:
                median = np.median(tmp[key], axis=0)
                quantile_upper = np.quantile(tmp[key], q=0.75, axis=0)
                quantile_lower = np.quantile(tmp[key], q=0.25, axis=0)
                if True:
                    ax.fill_between(dom, quantile_upper, quantile_lower, color=OBJ_DVH[key]['COLOR'], alpha=0.25)
                    ax.plot(dom, median, ls='--', color=OBJ_DVH[key]['COLOR'])
                else:
                    for hist in tmp[key]:
                        ax.plot(dom, hist, ls=':', color=OBJ_DVH[key]['COLOR'])
            #
            # let dose_ be percentages of the PTV_dose
            dose_ = D_full@sol_MOSEK / PTV_dose * 100
            for key in OBJ_DVH:
                hist = [np.sum(dose_[OBJ_DVH[key]['IDX']]>thres) for thres in dom]
                hist = np.array(hist) / OBJ_DVH[key]['IDX'].shape[0] * 100.0
                ax.plot(dom, hist, ls='-', color=OBJ_DVH[key]['COLOR'], label=key)
            ax.legend()

            fig.savefig(f'/tmp/DVHdist_{case}_{loss}_{score_method}_m{m}.pdf', dpi=300, transparent=True, bbox_inches='tight')
            fig.savefig(f'/tmp/DVHdist_{case}_{loss}_{score_method}_m{m}.png', dpi=300, transparent=True, bbox_inches='tight')





if __name__ == '__main__':
  fire.Fire(main)

