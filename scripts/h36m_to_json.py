import h5py
import numpy as np
import json

from os.path import join as os_join
from posixpath import join

if __name__ == '__main__':
    # camera
    dn = "D:/datasets/human3.6m/annotations/h36m"

    cameras_fn = os_join(dn, "cameras.h5")
    sub_ids = [1, 5, 6, 7, 8, 9, 11]
    cam_ids = [1, 2, 3, 4]
    print("load cameras from ", cameras_fn)

    cameras = dict()
    f_flag = True
    with h5py.File(cameras_fn, 'r') as f:
        for s_id in sub_ids:
            k_sub = f'subject{s_id}'
            cameras[k_sub] = dict()
            for c_id in cam_ids:
                k_cam = f'camera{c_id}'
                cam_R = f[join(k_sub, k_cam, "R")][()]
                cam_T = f[join(k_sub, k_cam, "T")][()]
                cam_c = f[join(k_sub, k_cam, "c")][()]
                cam_f = f[join(k_sub, k_cam, "f")][()]
                cam_k = f[join(k_sub, k_cam, "k")][()]
                cam_p = f[join(k_sub, k_cam, "p")][()]

                if f_flag:
                    print("R", cam_R.shape)
                    print("T", cam_T.shape)
                    print("c", cam_c.shape)
                    print("f", cam_f.shape)
                    print("k", cam_k.shape)
                    print("p", cam_p.shape)
                    f_flag = False

                cameras[k_sub][k_cam] = dict(
                    R=cam_R.tolist(),
                    T=cam_T.tolist(),
                    c=cam_c.tolist(),
                    f=cam_f.tolist(),
                    k=cam_k.tolist(),
                    p=cam_p.tolist(),
                )

        f.close()

    # output_fn = os_join(dn, "cameras.json")
    # with open(output_fn, "w") as f:
    #     json.dump(cameras, f)
    #     f.close()

    # print(json.dumps(cameras))

    kp_fn = "D:/datasets/human3.6m/annotations/h36m/S1/MyPoses/3D_positions/Directions_1.h5"
    keys = []
    with h5py.File(kp_fn, 'r') as f:
        f.visit(keys.append)
        print(f['3D_positions'])
        f.close()
    print(keys)