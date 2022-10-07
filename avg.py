import numpy as np
import sys
import os
import matplotlib.pyplot as plt

stress = float(sys.argv[1])

S1133 = -0.00000027659084042033574035	
S2233 = -0.00000027659084042033574037	
S3333 = 0.00000074054794816736705177	
l0 = 228.425019920042
a = 2.855312749000525                     # lattice parameter is the initial box length/80 CHECK THIS!!
P = a*np.array([[-1,1,1],[1,-1,1],[1,1,-1]])/2
D_half = np.identity(3) - 0.5*(np.array([[S1133,0,0],[0,S2233,0],[0,0,S3333]])*stress)
Pdeform = np.dot(np.dot(D_half,P),D_half)
#Pdeform = np.dot(P, np.identity(3) + (np.array([[S1133,0,0],[0,S2233,0],[0,0,S3333]])*stress))
Pdeforminv = np.linalg.inv(Pdeform)

d1 = np.loadtxt(f"CRA/{stress}/1/Fe_perfect_L80_pe_vol_box.dat")
d2 = np.loadtxt(f"CRA/{stress}/2/Fe_perfect_L80_pe_vol_box.dat")
d3 = np.loadtxt(f"CRA/{stress}/3/Fe_perfect_L80_pe_vol_box.dat")
m = np.mean([d1, d2, d3], axis=0)

#bcc_unit_cell= a*np.array([[1,0,0],[0,1,0],[0,0,1]])
#bcc_deformed = np.dot(np.dot(D_half,bcc_unit_cell),D_half)

w = np.zeros([3, d1.shape[0], 7])
Ldeform = np.zeros([3,d1.shape[0], 3,3])
for i in range(3):
    d = [d1,d2,d3][i]
    lx = d[:,5]
    ly = d[:,6]
    lz = d[:,7]
    Ldeform[i,:,0,0] = lx
    Ldeform[i,:,1,1] = ly
    Ldeform[i,:,2,2] = lz
    exx = (lx - l0)/l0
    eyy = (ly - l0)/l0
    ezz = (lz - l0)/l0
    strains = np.c_[exx, eyy, ezz]
    w11 = exx + S1133*stress
    w22 = eyy + S2233*stress
    w33 = ezz + S3333*stress
    rel_vol_den = np.c_[w11, w22, w33]
    w[i] = np.c_[d[:,2], strains, rel_vol_den]

Ldeform = np.mean(Ldeform, axis=0)
Ndeform = (Pdeforminv @ Ldeform)
breakpoint()
Ndeform = Ndeform.sum(axis=1)/2
np.savetxt(f"CRA/{stress}/Ndeform.dat",Ndeform)

wmean = np.mean(w, axis=0)
wsd = np.std(w, axis=0)

"""lx = m[:,5]
ly = m[:,6]
lz = m[:,7]
exx = (lx - l0)/l0
eyy = (ly - l0)/l0
ezz = (lz - l0)/l0
strains = np.c_[exx, eyy, ezz]
w11 = exx + S1133*stress
w22 = eyy + S2233*stress
w33 = ezz + S3333*stress
rel_vol_den = np.c_[w11, w22, w33]"""

np.savetxt(f"CRA/{stress}/Fe_perfect_L80_av_pe_vol_box.dat", m)
#np.savetxt(f"CRA/{stress}/Fe_perfect_L80_rel_vol_den.dat", np.c_[m[:,2], strains, rel_vol_den])
np.savetxt(f"CRA/{stress}/Fe_perfect_L80_rel_vol_den.dat", np.c_[wmean, wsd[:,4:]])

d1 = np.loadtxt(f"CRA/{stress}/1/Fe_perfect_L80_pe_voro_vms.dat")
d2 = np.loadtxt(f"CRA/{stress}/2/Fe_perfect_L80_pe_voro_vms.dat")
d3 = np.loadtxt(f"CRA/{stress}/3/Fe_perfect_L80_pe_voro_vms.dat")
m = np.mean([d1, d2, d3], axis=0)
np.savetxt(f"CRA/{stress}/Fe_perfect_L80_av_pe_voro_vms.dat", m)

d1 = np.loadtxt(f"CRA/{stress}/1/Fe_perfect_L80_DXA.dat", skiprows=1)
d2 = np.loadtxt(f"CRA/{stress}/2/Fe_perfect_L80_DXA.dat", skiprows=1)
d3 = np.loadtxt(f"CRA/{stress}/3/Fe_perfect_L80_DXA.dat", skiprows=1)
m = np.mean([d1, d2, d3], axis=0)
np.savetxt(f"CRA/{stress}/Fe_perfect_L80_av_DXA.dat", m)

d1 = np.loadtxt(f"CRA/{stress}/1/Fe_perfect_L80_defects.dat")
d2 = np.loadtxt(f"CRA/{stress}/2/Fe_perfect_L80_defects.dat")
d3 = np.loadtxt(f"CRA/{stress}/3/Fe_perfect_L80_defects.dat")
m = np.mean([d1, d2, d3], axis=0)
np.savetxt(f"CRA/{stress}/Fe_perfect_L80_defects.dat", m)

d1 = np.loadtxt(f"CRA/{stress}/1/C15hist.dat")
d2 = np.loadtxt(f"CRA/{stress}/2/C15hist.dat")
d3 = np.loadtxt(f"CRA/{stress}/3/C15hist.dat")
max_shape = max([d1.shape[1], d2.shape[1], d3.shape[1]])
d1 = np.c_[d1, np.nan*np.ones([d1.shape[0], max_shape - d1.shape[1]])]
d2 = np.c_[d2, np.nan*np.ones([d2.shape[0], max_shape - d2.shape[1]])]
d3 = np.c_[d3, np.nan*np.ones([d3.shape[0], max_shape - d3.shape[1]])]
m = np.nanmean([d1, d2, d3], axis=0)
np.savetxt(f"CRA/{stress}/av_C15_hist.dat", m)

os.system(f"cp CRA/{stress}/1/xedges.dat CRA/{stress}/av_C15_xedges.dat")

d1 = np.loadtxt(f"CRA/{stress}/1/yedges.dat")
d2 = np.loadtxt(f"CRA/{stress}/2/yedges.dat")
d3 = np.loadtxt(f"CRA/{stress}/3/yedges.dat")
m = [d1, d2, d3][np.argmax([d1.shape[0], d2.shape[0], d3.shape[0]])]
np.savetxt(f"CRA/{stress}/av_C15_yedges.dat", m)

d1 = np.loadtxt(f"CRA/{stress}/1/C15vol.dat")
d2 = np.loadtxt(f"CRA/{stress}/2/C15vol.dat")
d3 = np.loadtxt(f"CRA/{stress}/3/C15vol.dat")
m = np.mean([d1, d2, d3], axis=0)
np.savetxt(f"CRA/{stress}/avC15vol.dat", m)

print(f"{stress} bar done!")
