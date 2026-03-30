import numpy as np
from scipy.spatial.transform import Rotation

def invert(t):
    R = t[0:3,0:3]
    t = t[0:3,3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3,:3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def compute_relative(t_self, t_ref):
    
    t_rel = invert(t_ref) @ t_self
    return t_rel

def xyzquat2mat(pos, quat):
    T = np.eye(4)
    R = Rotation.from_quat(quat).as_matrix()
    t = pos
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def mat2xyzquat(t):
    quat = Rotation.from_matrix(t[0:3,0:3]).as_quat()
    pos = t[0:3,3]
    return pos, quat

