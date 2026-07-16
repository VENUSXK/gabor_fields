import os
from os.path import join, dirname
import numpy as np
import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import ScalarTransform4f as T

OBJECTS = {
    'resources': { 'type': 'resources', 'path': dirname(__file__) },
    'integrator': {
        'type': 'gfields_tomography',
    },
    'primitives_pyr1': {
        'type': 'ellipsoidsmesh',
        'extent': 3.0,
        'extent_adaptive_clamping': False,
        'filename': r'data/root.primitives_pyr1.ply',
    },
    'primitives_pyr0': {
        'type': 'ellipsoidsmesh',
        'extent': 3.0,
        'extent_adaptive_clamping': False,
        'filename': r'data/root.primitives_pyr0.ply',
    },
}

SENSORS = {
    'resources': { 'type': 'resources', 'path': dirname(__file__) },
    'cam_0000': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[4, 0, 0],
             target=[3.45107, -0.0343081, 0],
             up=[-0.0343081, 0.548929, 0],
         ),
        'shutter_close': 0.0,
    },
    'cam_0001': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.73205, 3, 2],
             target=[-1.50473, 2.57347, 1.73752],
             up=[0.279232, 0.347231, -0.322429],
         ),
        'shutter_close': 0.0,
    },
    'cam_0002': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.93649, -3.3541, 1],
             target=[-1.65568, -2.90397, 0.854989],
             up=[-0.39995, 0.316044, 0.206533],
         ),
        'shutter_close': 0.0,
    },
    'cam_0003': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.02676, 1.70066, 3],
             target=[1.75571, 1.43978, 2.59879],
             up=[-0.146041, 0.484194, -0.216168],
         ),
        'shutter_close': 0.0,
    },
    'cam_0004': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.72929, 1.35735, 0.5],
             target=[-3.22799, 1.14129, 0.432789],
             up=[0.214147, 0.505783, -0.0287115],
         ),
        'shutter_close': 0.0,
    },
    'cam_0005': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[0.542216, -3.07506, 2.5],
             target=[0.463967, -2.66737, 2.13922],
             up=[0.086414, 0.36917, 0.398429],
         ),
        'shutter_close': 0.0,
    },
    'cam_0006': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[0.643905, 3.65176, 1.5],
             target=[0.560171, 3.14438, 1.30494],
             up=[-0.200144, 0.212273, -0.466243],
         ),
        'shutter_close': 0.0,
    },
    'cam_0007': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.81971, -0.662319, 3.5],
             target=[-1.56737, -0.605143, 3.01466],
             up=[-0.0263749, 0.54702, 0.0507292],
         ),
        'shutter_close': 0.0,
    },
    'cam_0008': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.05819, -2.56612, 0.25],
             target=[2.62067, -2.23477, 0.214234],
             up=[0.330255, 0.438979, 0.0269976],
         ),
        'shutter_close': 0.0,
    },
    'cam_0009': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.21804, 0.76269, 2.25],
             target=[2.78157, 0.625337, 1.94483],
             up=[-0.112567, 0.532573, -0.0787049],
         ),
        'shutter_close': 0.0,
    },
    'cam_0010': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.60749, 2.76378, 1.25],
             target=[-2.26413, 2.36691, 1.08539],
             up=[0.357871, 0.380783, -0.171559],
         ),
        'shutter_close': 0.0,
    },
    'cam_0011': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.668781, -2.23388, 3.25],
             target=[-0.573635, -1.95164, 2.78763],
             up=[-0.0568876, 0.472058, 0.27645],
         ),
        'shutter_close': 0.0,
    },
    'cam_0012': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.34627, 3.15159, 0.75],
             target=[2.03901, 2.70612, 0.651781],
             up=[-0.424317, 0.322581, -0.135635],
         ),
        'shutter_close': 0.0,
    },
    'cam_0013': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.8851, 0.337219, 2.75],
             target=[-2.49123, 0.257054, 2.37458],
             up=[0.0580281, 0.544126, -0.0553109],
         ),
        'shutter_close': 0.0,
    },
    'cam_0014': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.42465, -3.30271, 1.75],
             target=[1.21824, -2.86043, 1.49646],
             up=[0.279227, 0.326936, 0.342995],
         ),
        'shutter_close': 0.0,
    },
    'cam_0015': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.0809342, 1.38959, 3.75],
             target=[-0.0700601, 1.1693, 3.24616],
             up=[0.00475329, 0.503957, -0.220239],
         ),
        'shutter_close': 0.0,
    },
    'cam_0016': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.34032, -2.19696, 0.125],
             target=[-2.86539, -1.92014, 0.107227],
             up=[-0.276626, 0.475259, 0.0103518],
         ),
        'shutter_close': 0.0,
    },
    'cam_0017': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.0284, -1.52092, 2.125],
             target=[2.6026, -1.34223, 1.82622],
             up=[0.146274, 0.520163, 0.102639],
         ),
        'shutter_close': 0.0,
    },
    'cam_0018': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.43024, 1.72273, 1.125],
             target=[2.97164, 1.45899, 0.974594],
             up=[-0.250609, 0.482638, -0.0821911],
         ),
        'shutter_close': 0.0,
    },
    'cam_0019': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.08611, 1.37205, 3.125],
             target=[-1.80575, 1.15406, 2.70502],
             up=[0.121031, 0.504955, -0.181306],
         ),
        'shutter_close': 0.0,
    },
    'cam_0020': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.229723, -3.94419, 0.625],
             target=[-0.196063, -3.40291, 0.533424],
             up=[-0.186736, 0.0975658, 0.508046],
         ),
        'shutter_close': 0.0,
    },
    'cam_0021': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.19544, 2.77134, 2.625],
             target=[1.03803, 2.37352, 2.27937],
             up=[-0.164877, 0.379788, -0.362045],
         ),
        'shutter_close': 0.0,
    },
    'cam_0022': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.63033, -0.424325, 1.625],
             target=[-3.12881, -0.400243, 1.40051],
             up=[-0.021981, 0.549473, 0.00983908],
         ),
        'shutter_close': 0.0,
    },
    'cam_0023': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.00978, -1.35637, 3.625],
             target=[0.868181, -1.20123, 3.11669],
             up=[0.0416305, 0.527666, 0.149449],
         ),
        'shutter_close': 0.0,
    },
    'cam_0024': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.14216, 3.81508, 0.375],
             target=[-0.993972, 3.28766, 0.326346],
             up=[0.501103, 0.155972, -0.164525],
         ),
        'shutter_close': 0.0,
    },
    'cam_0025': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.20874, -2.34112, 2.375],
             target=[-1.89394, -2.04309, 2.03651],
             up=[-0.202965, 0.46225, 0.218243],
         ),
        'shutter_close': 0.0,
    },
    'cam_0026': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.65499, -0.86625, 1.375],
             target=[3.14651, -0.780517, 1.18371],
             up=[0.0802428, 0.543277, 0.0301871],
         ),
        'shutter_close': 0.0,
    },
    'cam_0027': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.14049, 0.166372, 3.375],
             target=[1.8475, 0.10938, 2.91304],
             up=[-0.0305243, 0.547039, -0.0481289],
         ),
        'shutter_close': 0.0,
    },
    'cam_0028': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.20763, 3.21881, 0.875],
             target=[-1.9188, 2.76497, 0.76052],
             up=[0.421907, 0.310694, -0.167224],
         ),
        'shutter_close': 0.0,
    },
    'cam_0029': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.19972, -2.50899, 2.875],
             target=[-1.02824, -2.18612, 2.46408],
             up=[-0.12434, 0.445257, 0.297969],
         ),
        'shutter_close': 0.0,
    },
    'cam_0030': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.52254, 2.47409, 1.875],
             target=[2.18898, 2.11387, 1.62706],
             up=[-0.289103, 0.415622, -0.21489],
         ),
        'shutter_close': 0.0,
    },
    'cam_0031': {
        'type': 'orthographic',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 768,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.955815, 0.266069, 3.875],
             target=[-0.825186, 0.195539, 3.34541],
             up=[0.0168908, 0.545459, -0.0684775],
         ),
        'shutter_close': 0.0,
    },
}

EMITTERS = {
    'resources': { 'type': 'resources', 'path': dirname(__file__) },
    'environment': {
        'type': 'constant',
        'radiance': {
            'type': 'uniformspectrum',
            'value': 1.0,
        },
    },
}
