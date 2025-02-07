import fire

import numpy as np
import scipy as sp
import matplotlib.pylab as plt

import matplotlib

import CORT
import config
import utils


def main(case=None, loss='squared'):

    print(f'running the {case} case with {loss} loss')

    # set the directory where the result files (.npz) are located
    location = './results'

    repetitions = 50


    # set the subsample size
    if case == 'Prostate':
        m = 51778
    elif case == 'Liver':
        m = 9637
    elif case == 'HeadAndNeck':
        m = 25189


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

    n_BDY = len(BDY_indices)
    n_OAR = len(OAR_indices)
    n_PTV = len(PTV_indices)



    # specify the target dose
    # initialize the target dose to zero
    target_dose = np.zeros(D_full.shape[0])
    # set the PTV dose
    target_dose[OBJ[PTV_structure]['IDX']] = PTV_dose
    # set the OAR target dose to a threshold to prevent penalizing small violations
    target_dose[OAR_indices] = OAR_threshold
    # set the BDY target dose to a threshold to prevent penalizing small violations
    target_dose[BDY_indices] = BDY_threshold


    # overwrite target_dose to only consider BODY, OAR, and PTV voxels,
    # i.e., skip all other voxels outside the actual BODY
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

    ################################################################################


    if case == 'Prostate':
        dim = np.array([184, 184, 90])
        dim = np.roll(dim, 1)
        slice_ = 50
        x1 = 10
        x2 = 160
        y1 = 15
        y2 = 130
        ncol = 4
    elif case == 'Liver':
        dim = np.array([217, 217, 168])
        dim = np.roll(dim, 1)
        slice_ = 38
        x1 = 25
        x2 = 200
        y1 = 25
        y2 = 170
        ncol = 3
    elif case == 'HeadAndNeck':
        dim = np.array([160, 160, 67])
        dim = np.roll(dim, 1)
        slice_ = 30
        x1 = 40
        x2 = 120
        y1 = 10
        y2 = 115
        ncol = 2

    ###############################################################################

    nVoxels = np.prod(dim)

    def get_mask(obj):
        mask = np.zeros(nVoxels)
        mask[obj] = 1.0
        return mask.reshape(dim)

    #keys = [BODY_structure]+list(OBJ_DVH.keys())
    keys = OBJ_DVH.keys()

    for key in keys:
        OBJ[key]['MASK'] = get_mask(OBJ[key]['IDX'])

    ###############################################################################

    filename = f'{location}/result_{case}_gradnorm_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
    if loss == 'squared':
        filename = f'{location}/result_{case}_{loss}_gradnorm_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
    npz = np.load(filename, allow_pickle=True)

    list(npz.keys())

    x_hist = npz['x_hist']

    ###############################################################################

    # specify the target dose
    # initialize the target dose to zero
    target_dose_full = np.zeros(D_full.shape[0])
    # set the PTV dose
    target_dose_full[OBJ[PTV_structure]['IDX']] = PTV_dose
    # set the OAR target dose to a threshold to prevent penalizing small violations
    target_dose_full[OAR_indices] = OAR_threshold
    # set the BDY target dose to a threshold to prevent penalizing small violations
    target_dose_full[BDY_indices] = BDY_threshold

    weights = np.zeros(D_full.shape[0])
    weights[BDY_indices] = 1.0/n_BDY
    weights[OAR_indices] = 1.0/n_OAR
    weights[OBJ[PTV_structure]['IDX']] = 1.0/n_PTV

    # compute the residual
    res = D_full@x_hist[-1] - target_dose_full
    # compute the gradient per data point
    grad_per_point = D_full.multiply(2*np.multiply(weights, res).reshape(-1, 1))
    score_gradnorm = sp.sparse.linalg.norm(grad_per_point, ord=2, axis=1)
    dist_gradnorm = score_gradnorm / score_gradnorm.sum()

    score = score_gradnorm
    dist = dist_gradnorm

    ###############################################################################
    # plot distribution and samples

    sample_idx, new_weights = utils.compute_subset(D_full, weights, m, 0, score)
    chosen_sample = np.zeros(D_full.shape[0])
    chosen_sample[:] = np.nan
    chosen_sample[sample_idx] = 1.0

    cmap1 = 'rainbow'
    temperature = 0.4

    cmap2 = matplotlib.colors.ListedColormap([(0.0, 0.0, 0.0, 0.4)])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, x2-x1)
    ax.set_ylim(0, y2-y1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for key in keys:
        con = ax.contour(OBJ[key]['MASK'][slice_,x1:x2,y1:y2].T, levels=[0.5], colors=OBJ[key]['COLOR'])
        # hack to check whether the contour is empty
        if con.levels[0] > 0.0:
            # add label for legend
            ax.plot([], [], c=OBJ[key]['COLOR'], label=key)
    im = ax.imshow((dist**temperature).reshape(dim)[slice_,x1:x2,y1:y2].T, cmap=cmap1) #, vmin=0.0, vmax=0.035) # Greys
    im2 = ax.imshow(chosen_sample.reshape(dim)[slice_,x1:x2,y1:y2].T, interpolation='nearest', cmap=cmap2)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    ax.legend(ncol=ncol)

    fig.savefig(f'/tmp/samples{m}_{case}_slice{slice_}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/samples{m}_{case}_slice{slice_}.png', dpi=300, transparent=True, bbox_inches='tight')



if __name__ == '__main__':
  fire.Fire(main)

