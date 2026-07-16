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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[4, 0, 0],
             target=[3.5, 0, 0],
             up=[0, 0.5, 0],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.73205, 3, 2],
             target=[-1.51554, 2.625, 1.75],
             up=[0.245495, 0.330719, -0.283473],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.93649, -3.3541, 1],
             target=[-1.69443, -2.93484, 0.875],
             up=[-0.372525, 0.272431, 0.192371],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.02676, 1.70066, 3],
             target=[1.77342, 1.48807, 2.625],
             up=[-0.119005, 0.452558, -0.17615],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.72929, 1.35735, 0.5],
             target=[-3.26313, 1.18768, 0.4375],
             up=[0.168164, 0.470332, -0.0225464],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[0.542216, -3.07506, 2.5],
             target=[0.474439, -2.69068, 2.1875],
             up=[0.0814732, 0.319766, 0.375649],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[0.643905, 3.65176, 1.5],
             target=[0.563417, 3.19529, 1.3125],
             up=[-0.18006, 0.204046, -0.419457],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.81971, -0.662319, 3.5],
             target=[-1.59224, -0.579529, 3.0625],
             up=[-0.0381905, 0.493098, 0.0734551],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.05819, -2.56612, 0.25],
             target=[2.67591, -2.24536, 0.21875],
             up=[0.319699, 0.383549, 0.0261347],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.21804, 0.76269, 2.25],
             target=[2.81579, 0.667354, 1.96875],
             up=[-0.0781325, 0.490827, -0.0546289],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.60749, 2.76378, 1.25],
             target=[-2.28156, 2.41831, 1.09375],
             up=[0.311526, 0.361454, -0.149342],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.668781, -2.23388, 3.25],
             target=[-0.585183, -1.95465, 2.84375],
             up=[-0.0562814, 0.414762, 0.273505],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.34627, 3.15159, 0.75],
             target=[2.05299, 2.75764, 0.65625],
             up=[-0.375244, 0.307903, -0.119949],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.8851, 0.337219, 2.75],
             target=[-2.52446, 0.295067, 2.40625],
             up=[0.0305121, 0.49822, -0.0290833],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.42465, -3.30271, 1.75],
             target=[1.24657, -2.88987, 1.53125],
             up=[0.260638, 0.282072, 0.320161],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.0809342, 1.38959, 3.75],
             target=[-0.0708174, 1.21589, 3.28125],
             up=[0.00374796, 0.468859, -0.173658],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.34032, -2.19696, 0.125],
             target=[-2.92278, -1.92234, 0.109375],
             up=[-0.274428, 0.417832, 0.0102695],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.0284, -1.52092, 2.125],
             target=[2.64985, -1.3308, 1.85938],
             up=[0.155624, 0.462446, 0.1092],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.43024, 1.72273, 1.125],
             target=[3.00146, 1.50739, 0.984375],
             up=[-0.204618, 0.451252, -0.0671076],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.08611, 1.37205, 3.125],
             target=[-1.82534, 1.20055, 2.73438],
             up=[0.0952225, 0.469665, -0.142644],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.229723, -3.94419, 0.625],
             target=[-0.201007, -3.45116, 0.546875],
             up=[-0.170088, 0.0832351, 0.462755],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.19544, 2.77134, 2.625],
             target=[1.04601, 2.42492, 2.29688],
             up=[-0.143573, 0.360549, -0.315264],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-3.63033, -0.424325, 1.625],
             target=[-3.17654, -0.371285, 1.42188],
             up=[-0.048412, 0.497179, 0.02167],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[1.00978, -1.35637, 3.625],
             target=[0.883554, -1.18682, 3.17188],
             up=[0.0454963, 0.470377, 0.163327],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.14216, 3.81508, 0.375],
             target=[-0.99939, 3.3382, 0.328125],
             up=[0.453089, 0.150268, -0.148761],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.20874, -2.34112, 2.375],
             target=[-1.93264, -2.04848, 2.07812],
             up=[-0.199291, 0.405415, 0.214293],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[3.65499, -0.86625, 1.375],
             target=[3.19812, -0.757968, 1.20312],
             up=[0.101347, 0.488134, 0.0381265],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.14049, 0.166372, 3.375],
             target=[1.87293, 0.145575, 2.95312],
             up=[-0.0111383, 0.499567, -0.0175622],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-2.20763, 3.21881, 0.875],
             target=[-1.93168, 2.81646, 0.765625],
             up=[0.374042, 0.296839, -0.148252],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-1.19972, -2.50899, 2.875],
             target=[-1.04975, -2.19537, 2.51562],
             up=[-0.120779, 0.38941, 0.289435],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[2.52254, 2.47409, 1.875],
             target=[2.20723, 2.16483, 1.64062],
             up=[-0.248206, 0.392883, -0.184491],
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
            'width': 256,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'to_world': T().look_at(
             origin=[-0.955815, 0.266069, 3.875],
             target=[-0.836338, 0.232811, 3.39062],
             up=[0.00796492, 0.498893, -0.0322908],
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
