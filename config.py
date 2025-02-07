import numpy as np

dataset = 'CORT'

# path to the dataset
base_path = './data/CORT'


def get_config(case):
    if case == 'Prostate':
        # specify the data path
        data_path = f'{base_path}/Prostate'

        # gantry levels to consider
        # np.arange(0, 359, 360/5)
        gantry_angles = [0, 72, 144, 216, 288]
        couch_angles = [0, 0, 0, 0, 0]

        # every structure is assigned a color
        OBJ = {
            'PTV_68':{'COLOR':'tab:blue'},
            'PTV_56':{'COLOR':'tab:cyan'},
            'Rectum':{'COLOR':'tab:green'},
            'BODY':{'COLOR':'black'},
            'Bladder':{'COLOR':'tab:orange'},
            'Penile_bulb':{'COLOR':'tab:red'},
            'Lt_femoral_head':{'COLOR':'tab:pink'},
            'Rt_femoral_head':{'COLOR':'tab:purple'},
            'Lymph_Nodes':{'COLOR':'tab:brown'},
            'prostate_bed':{'COLOR':'tab:olive'}
        }

        # planned target volume
        PTV_structure = 'PTV_68'
        PTV_dose = 68.0

        # body
        BODY_structure = 'BODY'
        BODY_threshold = 5.0

        # organs at risk
        OAR_structures = ['Rectum','Bladder','Penile_bulb',
                          'Lt_femoral_head','Rt_femoral_head',
                          'Lymph_Nodes']
        OAR_threshold = 5.0

        dim = np.array([184, 184, 90])
        dim = np.roll(dim, 1)

        # parameters for the projected gradient descent
        # a learning rate eta = -1 means that the step-size will be computed
        eta = -1.0
        steps = 20

        return data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BODY_threshold, OAR_structures, OAR_threshold, eta, steps


    elif case == 'Liver':
        # specify the data path
        data_path = f'{base_path}/Liver/'

        # gantry levels to consider
        gantry_angles = [32, 90, 148, 212, 270, 328]
        couch_angles = [0, 0, 0, 0, 0, 0]

        # every structure is assigned a color
        OBJ = {
            'CTV':{'COLOR':'tab:olive'},
            'Celiac':{'COLOR':'tab:olive'},
            'DoseFalloff':{'COLOR':'tab:olive'},
            'GTV':{'COLOR':'tab:olive'},
            'LargeBowel':{'COLOR':'tab:olive'},
            'SmallBowel':{'COLOR':'tab:olive'},
            'SMASMV':{'COLOR':'tab:olive'},
            'duodenum':{'COLOR':'tab:olive'},
            'entrance':{'COLOR':'tab:olive'},
            'PTV':{'COLOR':'tab:blue'},
            'Heart':{'COLOR':'tab:green'},
            'Skin':{'COLOR':'black'},
            'Liver':{'COLOR':'tab:orange'},
            'SpinalCord':{'COLOR':'tab:red'},
            'KidneyL':{'COLOR':'tab:pink'},
            'KidneyR':{'COLOR':'tab:purple'},
            'Stomach':{'COLOR':'tab:brown'}
        }

        # planned target volume
        PTV_structure = 'PTV'
        PTV_dose = 56.0

        # body
        BODY_structure = 'Skin'
        BODY_threshold = 5.0

        # organs at risk
        OAR_structures = ['Heart','Liver','SpinalCord',
                          #'KidneyL','KidneyR',
                          'Stomach']
        OAR_threshold = 5.0

        dim = np.array([217, 217, 168])
        dim = np.roll(dim, 1)

        # parameters for the projected gradient descent
        # a learning rate eta = -1 means that the step-size will be computed
        eta = -1.0
        steps = 20

        return data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BODY_threshold, OAR_structures, OAR_threshold, eta, steps


    elif case == 'HeadAndNeck':
        # specify the data path
        data_path = f'{base_path}/HeadAndNeck/'

        # gantry levels to consider
        # np.arange(0, 359, int(360/7)+1)
        gantry_angles = [0, 52, 104, 156, 208, 260, 312]
        couch_angles = [0, 0, 0, 0, 0, 0, 0]

        # every structure is assigned a color
        OBJ = {
            'CEREBELLUM':{'COLOR':'tab:olive'},
            'CTV56':{'COLOR':'tab:olive'},
            'CTV63':{'COLOR':'tab:olive'},
            'GTV':{'COLOR':'tab:olive'},
            'LARYNX':{'COLOR':'tab:brown'},
            'LENS_LT':{'COLOR':'tab:olive'},
            'LENS_RT':{'COLOR':'tab:olive'},
            'LIPS':{'COLOR':'tab:cyan'},
            'OPTIC_NRV_LT':{'COLOR':'tab:olive'},
            'OPTIC_NRV_RT':{'COLOR':'tab:olive'},
            'TEMP_LOBE_LT':{'COLOR':'tab:olive'},
            'TEMP_LOBE_RT':{'COLOR':'tab:olive'},
            #'TM_JOINT_LT':{'COLOR':'tab:olive'},
            #'TM_JOINT_RT':{'COLOR':'tab:olive'},
            'PTV56':{'COLOR':'tab:olive'},
            'PTV63':{'COLOR':'tab:olive'},
            'PTV70':{'COLOR':'tab:blue'},
            'BRAIN_STEM_PRV':{'COLOR':'tab:olive'},
            'BRAIN_STEM':{'COLOR':'tab:green'},
            'External':{'COLOR':'black'},
            'CHIASMA':{'COLOR':'tab:orange'},
            'SPINAL_CORD':{'COLOR':'tab:red'},
            'SPINL_CRD_PRV':{'COLOR':'tab:olive'},
            'PAROTID_LT':{'COLOR':'tab:pink'},
            'PAROTID_RT':{'COLOR':'tab:purple'}
        }

        # planned target volume
        PTV_structure = 'PTV70'
        PTV_dose = 70.0

        # body
        BODY_structure = 'External'
        BODY_threshold = 5.0

        # organs at risk
        OAR_structures = ['BRAIN_STEM','CHIASMA','SPINAL_CORD',
                          'PAROTID_LT','PAROTID_RT',
                          'LARYNX', 'LIPS']
        OAR_threshold = 5.0

        dim = np.array([160, 160, 67])
        dim = np.roll(dim, 1)

        # parameters for the projected gradient descent
        # a learning rate eta = -1 means that the step-size will be computed
        eta = -1.0
        steps = 20

        return data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BODY_threshold, OAR_structures, OAR_threshold, eta, steps

    else:
        raise NotImplementedError

