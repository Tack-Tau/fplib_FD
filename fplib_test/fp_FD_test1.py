import numpy as np
import fplib_FD
import sys

# Move function `readvasp(vp)` from test set to `fplib_FD.py`

def test4(v1, v2):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat1, rxyz1, types = fplib_FD.readvasp(v1)
    lat2, rxyz2, types = fplib_FD.readvasp(v2)
    contract = False
    iterations = 100
    fpd_opt = get_fpd_optimize(v1, v2, iterations, contract, ntyp, nx, lmax, znucl, cutoff)
    print ('Optimized fingerprint distance: ', fpd_opt)


if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    v2 = args[2]
    test4(v1, v2)
