import numpy as np
import fplib_FD
import sys

# Move function `readvasp(vp)` from test set to `fplib_FD.py`

def test3(v1, v2):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat1, rxyz1, types = fplib_FD.readvasp(v1)
    lat2, rxyz2, types = fplib_FD.readvasp(v2)
    contract = True
    fp1 = fplib_FD.get_fp(contract, ntyp, nx, lmax, lat1, rxyz1, types, znucl, cutoff)
    fp2 = fplib_FD.get_fp(contract, ntyp, nx, lmax, lat2, rxyz2, types, znucl, cutoff)

    dist = fplib_FD.get_fpdist(ntyp, types, fp1, fp2)
    print ('fingerprint distance: ', dist)


if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    v2 = args[2]
    test3(v1, v2)
