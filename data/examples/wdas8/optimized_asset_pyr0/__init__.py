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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[4, 0, 0],
             target=[3.05185, -0.0592594, 0],
             up=[-0.0592594, 0.94815, 0],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.73205, 3, 2],
             target=[-1.33941, 2.26326, 1.54662],
             up=[0.482309, 0.599763, -0.556923],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.93649, -3.3541, 1],
             target=[-1.45145, -2.57661, 0.749526],
             up=[-0.690823, 0.545894, 0.356739],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.02676, 1.70066, 3],
             target=[1.55858, 1.25005, 2.30699],
             up=[-0.252252, 0.836334, -0.373382],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.72929, 1.35735, 0.5],
             target=[-2.86341, 0.984151, 0.383908],
             up=[0.36989, 0.873626, -0.0495925],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[0.542216, -3.07506, 2.5],
             target=[0.407059, -2.37086, 1.87683],
             up=[0.14926, 0.637658, 0.688196],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[0.643905, 3.65176, 1.5],
             target=[0.499274, 2.77537, 1.16308],
             up=[-0.345703, 0.366653, -0.805329],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.81971, -0.662319, 3.5],
             target=[-1.38385, -0.563561, 2.66168],
             up=[-0.0455567, 0.944853, 0.0876232],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.05819, -2.56612, 0.25],
             target=[2.30247, -1.99378, 0.188222],
             up=[0.57044, 0.758237, 0.0466322],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.21804, 0.76269, 2.25],
             target=[2.46414, 0.525445, 1.72289],
             up=[-0.194434, 0.919899, -0.135945],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.60749, 2.76378, 1.25],
             target=[-2.01441, 2.07828, 0.965681],
             up=[0.618141, 0.657716, -0.296329],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.668781, -2.23388, 3.25],
             target=[-0.504438, -1.74637, 2.45136],
             up=[-0.0982603, 0.815373, 0.477505],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.34627, 3.15159, 0.75],
             target=[1.81554, 2.38214, 0.580349],
             up=[-0.73291, 0.557185, -0.234279],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.8851, 0.337219, 2.75],
             target=[-2.20478, 0.198751, 2.10154],
             up=[0.10023, 0.939855, -0.095537],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.42465, -3.30271, 1.75],
             target=[1.06813, -2.53877, 1.31206],
             up=[0.482301, 0.564708, 0.592445],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.0809342, 1.38959, 3.75],
             target=[-0.0621517, 1.00909, 2.87973],
             up=[0.00821023, 0.87047, -0.380412],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.34032, -2.19696, 0.125],
             target=[-2.51999, -1.71882, 0.094302],
             up=[-0.477808, 0.820902, 0.0178803],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.0284, -1.52092, 2.125],
             target=[2.29293, -1.21227, 1.60893],
             up=[0.252655, 0.898463, 0.177286],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.43024, 1.72273, 1.125],
             target=[2.63811, 1.26718, 0.865208],
             up=[-0.432871, 0.833648, -0.141966],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.08611, 1.37205, 3.125],
             target=[-1.60185, 0.995523, 2.39959],
             up=[0.209054, 0.872195, -0.313164],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.229723, -3.94419, 0.625],
             target=[-0.171584, -3.00925, 0.466824],
             up=[-0.322543, 0.168523, 0.877534],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.19544, 2.77134, 2.625],
             target=[0.923559, 2.08419, 2.02799],
             up=[-0.284788, 0.655998, -0.62535],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.63033, -0.424325, 1.625],
             target=[-2.76407, -0.382728, 1.23724],
             up=[-0.0379672, 0.949089, 0.0169948],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.00978, -1.35637, 3.625],
             target=[0.765203, -1.0884, 2.747],
             up=[0.0719072, 0.911424, 0.25814],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.14216, 3.81508, 0.375],
             target=[-0.886198, 2.90408, 0.290961],
             up=[0.865542, 0.269405, -0.284179],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.20874, -2.34112, 2.375],
             target=[-1.665, -1.82634, 1.79033],
             up=[-0.350575, 0.798432, 0.376965],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.65499, -0.86625, 1.375],
             target=[2.7767, -0.718165, 1.04459],
             up=[0.138601, 0.938387, 0.0521414],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.14049, 0.166372, 3.375],
             target=[1.63442, 0.0679308, 2.57706],
             up=[-0.0527237, 0.944886, -0.0831317],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.20763, 3.21881, 0.875],
             target=[-1.70874, 2.43491, 0.677262],
             up=[0.728749, 0.536653, -0.288841],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.19972, -2.50899, 2.875],
             target=[-0.903538, -1.95131, 2.16524],
             up=[-0.21477, 0.769081, 0.514674],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.52254, 2.47409, 1.875],
             target=[1.94638, 1.8519, 1.44674],
             up=[-0.49936, 0.717893, -0.371173],
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
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.955815, 0.266069, 3.875],
             target=[-0.730183, 0.144245, 2.96026],
             up=[0.029175, 0.942157, -0.118279],
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
