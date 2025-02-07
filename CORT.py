import scipy as sp

from os.path import exists


def load_structure(file):
    if exists(file):
        MAT = sp.io.loadmat(file)
        # subtract 1 due to indexing mismatch
        # MATLAB starts with 1 and Python with 0
        return MAT['v'].reshape(-1) - 1
    else:
        print(f'ERROR: file {file} not found')
        return None


def load_data(data_path, OBJ, angles):
    # angles is a list of lists [[gantry_angle, couch_angle],...]

    if not type(data_path) == str or len(data_path)<1:
        print(f'ERROR: the variable data_path={data_path} should be properly set')
        return False

    # load the structure indices
    for key in OBJ.keys():
        OBJ[key]['IDX'] = load_structure(f'{data_path}/{key}_VOILIST.mat')

    # load the dose influence matrix D per gantry angle and concatenate them
    D = []
    for gantry_angle, couch_angle in angles:
        file = f'{data_path}/Gantry{gantry_angle}_Couch{couch_angle}_D.mat'
        if exists(file):
            beam_D = sp.io.loadmat(file)
            D.append(beam_D['D'])
        else:
            print(f'ERROR: file {file} not found')
    D_full = sp.sparse.hstack(D)

    return D_full


