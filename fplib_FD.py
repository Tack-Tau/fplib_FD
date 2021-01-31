import numpy as np
from scipy.optimize import linear_sum_assignment
import rcovdata
# import numba

# @numba.jit()
def get_gom(lseg, rxyz, rcov, amp):
    # s orbital only lseg == 1
    nat = len(rxyz)    
    if lseg == 1:
        om = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[iat][jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1.0*d2*r) * amp[iat] * amp[jat]
    else:
        # for both s and p orbitals
        om = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[4*iat][4*jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1*d2*r) * amp[iat] * amp[jat]
                
                # <s_i | p_j>
                sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                om[4*iat][4*jat+1] = stv * d[0] * amp[iat] * amp[jat]
                om[4*iat][4*jat+2] = stv * d[1] * amp[iat] * amp[jat]
                om[4*iat][4*jat+3] = stv * d[2] * amp[iat] * amp[jat]

                # <p_i | s_j> 
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                om[4*iat+1][4*jat] = stv * d[0] * amp[iat] * amp[jat]
                om[4*iat+2][4*jat] = stv * d[1] * amp[iat] * amp[jat]
                om[4*iat+3][4*jat] = stv * d[2] * amp[iat] * amp[jat]

                # <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                om[4*iat+1][4*jat+1] = stv * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat]
                om[4*iat+1][4*jat+2] = stv * (d[1] * d[0]        ) * amp[iat] * amp[jat]
                om[4*iat+1][4*jat+3] = stv * (d[2] * d[0]        ) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+1] = stv * (d[0] * d[1]        ) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+2] = stv * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+3] = stv * (d[2] * d[1]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+1] = stv * (d[0] * d[2]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+2] = stv * (d[1] * d[2]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+3] = stv * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat]
    
    # for i in range(len(om)):
    #     for j in range(len(om)):
    #         if abs(om[i][j] - om[j][i]) > 1e-6:
    #             print ("ERROR", i, j, om[i][j], om[j][i])
    return om


# @numba.jit()
def get_fp_nonperiodic(rxyz, znucls):
    rcov = []
    amp = [1.0] * len(rxyz)
    for x in znucls:
        rcov.append(rcovdata.rcovdata[x][2])
    gom = get_gom(1, rxyz, rcov, amp)
    fp = np.linalg.eigvals(gom)
    fp = sorted(fp)
    fp = np.array(fp, float)
    return fp

# @numba.jit()
def get_fpdist_nonperiodic(fp1, fp2):
    d = fp1 - fp2
    return np.sqrt(np.vdot(d, d))

# @numba.jit()
def get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2
    ixyz = get_ixyz(lat, cutoff)
    NC = 3
    wc = cutoff / np.sqrt(2.* NC)
    fc = 1.0 / (2.0 * NC * wc**2)
    nat = len(rxyz)
    cutoff2 = cutoff**2 
    
    n_sphere_list = []
    lfp = []
    sfp = []
    for iat in range(nat):
        rxyz_sphere = []
        rcov_sphere = []
        ind = [0] * (lseg * nx)
        amp = []
        xi, yi, zi = rxyz[iat]
        n_sphere = 0
        for jat in range(nat):
            for ix in range(-ixyz, ixyz+1):
                for iy in range(-ixyz, ixyz+1):
                    for iz in range(-ixyz, ixyz+1):
                        xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                        yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                        zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                        d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > nx:
                                print ("FP WARNING: the cutoff is too large.")
                            amp.append((1.0-d2*fc)**NC)
                            # print (1.0-d2*fc)**NC
                            rxyz_sphere.append([xj, yj, zj])
                            rcov_sphere.append(rcovdata.rcovdata[znucl[types[jat]-1]][2]) 
                            if jat == iat and ix == 0 and iy == 0 and iz == 0:
                                ityp_sphere = 0
                            else:
                                ityp_sphere = types[jat]
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
                                    # ind[il+lseg*(n_sphere-1)] == ityp_sphere * l + 1
        n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere, float)
        
        # full overlap matrix
        nid = lseg * n_sphere
        gom = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
        val, vec = np.linalg.eig(gom)
        val = np.real(val)
        fp0 = np.zeros(nx*lseg)
        for i in range(len(val)):
            fp0[i] = val[i]
        lfp.append(sorted(fp0))
        pvec = np.real(np.transpose(vec)[0])
        
        # contracted overlap matrix
        if contract:
            nids = l * (ntyp + 1)
            omx = np.zeros((nids, nids))
            for i in range(nid):
                for j in range(nid):
                    # print ind[i], ind[j]
                    omx[ind[i]][ind[j]] = omx[ind[i]][ind[j]] + pvec[i] * gom[i][j] * pvec[j]
            # for i in range(nids):
            #     for j in range(nids):
            #         if abs(omx[i][j] - omx[j][i]) > 1e-6:
            #             print ("ERROR", i, j, omx[i][j], omx[j][i])
            # print omx
            sfp0 = np.linalg.eigvals(omx)
            sfp.append(sorted(sfp0))


    # print ("n_sphere_min", min(n_sphere_list))
    # print ("n_shpere_max", max(n_sphere_list)) 

    if contract:
        sfp = np.array(sfp, float)
        return sfp
    else:
        lfp = np.array(lfp, float)
        return lfp

# @numba.jit()
def get_ixyz(lat, cutoff):
    lat2 = np.matmul(lat, np.transpose(lat))
    # print lat2
    vec = np.linalg.eigvals(lat2)
    # print (vec)
    ixyz = int(np.sqrt(1.0/max(vec))*cutoff) + 1
    return ixyz

# @numba.jit()
def get_fpdist(ntyp, types, fp1, fp2):
    nat, lenfp = np.shape(fp1)
    fpd = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        MX = np.zeros((nat, nat))
        for iat in range(nat):
            if types[iat] == itype:
                for jat in range(nat):
                    if types[jat] == itype:
                        tfpd = fp1[iat] - fp2[jat]
                        MX[iat][jat] = np.sqrt(np.vdot(tfpd, tfpd)/lenfp)

        row_ind, col_ind = linear_sum_assignment(MX)
        # print(row_ind, col_ind)
        total = MX[row_ind, col_ind].sum()
        fpd += total

    fpd = fpd / nat
    return fpd

# @numba.jit()
def readvasp(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())

    lat = np.array(buff[2:5], float) 
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    nat = sum(typt)
    pos = np.array(buff[7:7 + nat], float)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    rxyz = np.dot(pos, lat)
    #rxyz = pos
    return lat, rxyz, types

# @numba.jit()
def get_rxyz_delta(rxyz):
    nat = len(rxyz)
    rxyz_delta = np.random.rand(nat, 3)
    for iat in range(nat):
        r_norm = np.linalg.norm(rxyz_delta[iat])
        rxyz_delta[iat] = np.divide(rxyz_delta[iat], r_norm)
    # rxyz_plus = np.add(rxyz, rxyz_delta)
    # rxyz_minus = np.subtract(rxyz, rxyz_delta)
        
    return rxyz_delta

# @numba.jit()
def get_fpd_optimize(v1, v2, iter_max, atol, contract, ntyp, nx, lmax, znucl, cutoff):
    lat1, rxyz1, types = readvasp(v1)
    lat2, rxyz2, types = readvasp(v2)
    nat2 = len(rxyz2)
    rxyz2_right = rxyz2.copy()
    rxyz2_left = rxyz2.copy()
    fp1 = get_fp(contract, ntyp, nx, lmax, lat1, rxyz1, types, znucl, cutoff)
    fp2 = get_fp(contract, ntyp, nx, lmax, lat2, rxyz2, types, znucl, cutoff)
    fpd_init = get_fpdist(ntyp, types, fp1, fp2)
    print ('fpd_init', fpd_init)
    nfp2, lenfp2 = np.shape(fp2)
    fp_FD = np.ones((nat2, 3))
    fp2_right = np.zeros((nat2, 3, nfp2, lenfp2))
    fp2_left = np.zeros((nat2, 3, nfp2, lenfp2))
    fpd_right = np.full((nat2, 3), fpd_init)
    fpd_left = np.full((nat2, 3), fpd_init)
    step_size = 1e-4
    d = 1e-8
    n_iter = 0
    while min( abs( fp_FD.ravel() ) ) >= atol and n_iter <= iter_max:
        n_iter = n_iter + 1
        rxyz2_delta = get_rxyz_delta(rxyz2)
        rxyz2_delta = d*rxyz2_delta
        rxyz2_plus = np.add(rxyz2, rxyz2_delta)
        rxyz2_minus = np.subtract(rxyz2, rxyz2_delta)
        for inat2 in range(nat2):
            for x_i in range (3):
                # Calculate numerical gradient using Finite Difference in high-dimension
                rxyz2_right[inat2][x_i] = rxyz2_plus[inat2][x_i]
                rxyz2_left[inat2][x_i] = rxyz2_minus[inat2][x_i]
                fp2_right[inat2, x_i, :, :] = get_fp(contract, ntyp, nx, \
                       lmax, lat2, rxyz2_right, types, znucl, cutoff)
                fp2_left[inat2, x_i, :, :] = get_fp(contract, ntyp, nx,  \
                       lmax, lat2, rxyz2_left, types, znucl, cutoff)
                fpd_right[inat2][x_i] = get_fpdist(ntyp, types, fp1, fp2_right[inat2, x_i, :, :])
                fpd_left[inat2][x_i] = get_fpdist(ntyp, types, fp1, fp2_left[inat2, x_i, :, :])
                fp_FD[inat2][x_i] = ( fpd_right[inat2][x_i] - fpd_left[inat2][x_i] ) \
                                       / 2.0*abs( rxyz2_delta[inat2][x_i] )
        # R(x,y,z) matrix update using Steepest Descent method
        # At this moment the step size is fixed, but an adaptive step size can be implemented from:
        # https://github.com/yrlu/non-convex
        # https://github.com/tamland/non-linear-optimization
        rxyz2 = np.subtract(rxyz2, step_size * fp_FD)
    
    fp1 = get_fp(contract, ntyp, nx, lmax, lat1, rxyz1, types, znucl, cutoff)
    fp2 = get_fp(contract, ntyp, nx, lmax, lat2, rxyz2, types, znucl, cutoff)
    fpd_opt = get_fpdist(ntyp, types, fp1, fp2)
    return fpd_opt
    
    
def gd(v1, v2, iter_max, atol, contract, ntyp, nx, lmax, znucl, cutoff):
    lat1, rxyz1, types = readvasp(v1)
    lat2, rxyz2, types = readvasp(v2)
    print (rxyz1 - rxyz2)
    nat = len(rxyz2)
    rxyz2_right = rxyz2
    rxyz2_left = rxyz2
    fp1 = get_fp(contract, ntyp, nx, lmax, lat1, rxyz1, types, znucl, cutoff)
    fp2 = get_fp(contract, ntyp, nx, lmax, lat2, rxyz2, types, znucl, cutoff)
    fpd_init = get_fpdist(ntyp, types, fp1, fp2)
    print ('fpd_init', fpd_init)

    # dx = np.random.random((nat, 3))/10

    step_size = 0.03
    n_iter = 0
    dg = min_grad(nat, fp1, rxyz2, lat2, types, znucl, cutoff, contract, ntyp, nx, lmax)
    nrxyz = rxyz2.copy()
    fpdn = fpd_init
   
    while fpdn >= atol and n_iter <= iter_max:
        n_iter += 1
        nrxyz = nrxyz + dg*step_size
        nrxyz = checkdist(rxyz2, nrxyz)
        fpn = get_fp(contract, ntyp, nx, lmax, lat2, nrxyz, types, znucl, cutoff)
        fpdn = get_fpdist(ntyp, types, fp1, fpn)
        dg = min_grad(nat, fp1, nrxyz, lat2, types, znucl, cutoff, contract, ntyp, nx, lmax)
        # print (dg)
        print (n_iter, fpdn)
    print (nrxyz)
    return fpdn


def min_grad(nat, fp0, rxyz, lat, types, znucl, cutoff, contract, ntyp, nx, lmax):
    dgg = []
    for i in range(10):
        dx = np.random.random((nat, 3))/100
        dg, fpd = get_grad(dx, fp0, rxyz, lat, types, znucl, cutoff, contract, ntyp, nx, lmax)
        dgg.append((dg, fpd))
    sortdgg = sorted(dgg, key = lambda x:x[1])
    return sortdgg[0][0]



def get_grad(dx, fp0, rxyz, lat, types, znucl, cutoff, contract, ntyp, nx, lmax):

    rxyz_left = rxyz - dx
    rxyz_right = rxyz + dx
    # fp =  get_fp(contract, ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff)
    fp_left = get_fp(contract, ntyp, nx, lmax, lat, rxyz_left, types, znucl, cutoff)
    fp_right = get_fp(contract, ntyp, nx, lmax, lat, rxyz_right, types, znucl, cutoff)
    fpd_left = get_fpdist(ntyp, types, fp0, fp_left)
    fpd_right = get_fpdist(ntyp, types, fp0, fp_right)
    # print ('lf', fpd_left, fpd_right)
    fpd = fpd_right - fpd_left
    dg = fpd / (2*dx)
    for i in range(len(dg)):
        dg[i] = dg[i]/np.sqrt(np.dot(dg[i], dg[i]))
    return dg, fpd

def checkdist(rxyz0, rxyz):
    rxyz1 = rxyz.copy()
    for i in range(len(rxyz1)):
        d = rxyz0[i] - rxyz1[i]
        dist = np.sqrt(np.dot(d, d))
        if dist > 1.0:
            xd = 1.0/dist
            rxyz1[i] = rxyz0[i] - d*xd
            # print ('dist', dist)
    return rxyz1



