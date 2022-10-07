import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

os.system("mkdir -p figures/C15 figures/dxa figures/w figures/e figures/other figures/defects")

DPA = np.linspace(0, 3, 3001)
step = np.r_[np.linspace(1, 10, 10), np.linspace(20, 1000, 99), np.linspace(1050, 3000, 40)].astype(int)
stress = np.array([-10000.0, -5000.0, -2000.0, -1000.0, -500.0, -200.0, -100.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0])

w = np.zeros([stress.shape[0], DPA.shape[0], 9])
dxa = np.zeros([stress.shape[0], step.shape[0], 5])
pevol = np.zeros([stress.shape[0], DPA.shape[0], 2])
vorovms = np.zeros([stress.shape[0], DPA[1:].shape[0], 6])
defects = np.zeros([stress.shape[0], step.shape[0], 2])
Ndeform = np.zeros([stress.shape[0], DPA.shape[0], 3])

"""for s in range(stress.shape[0]):
    w[s] = np.loadtxt(f"CRA/{stress[s]}/Fe_perfect_L80_rel_vol_den.dat")[:,1:]
    dxa[s] = np.loadtxt(f"CRA/{stress[s]}/Fe_perfect_L80_av_DXA.dat")[:,1:]
    pevol[s] = np.loadtxt(f"CRA/{stress[s]}/Fe_perfect_L80_av_pe_vol_box.dat")[:,3:5]
    vorovms[s] = np.loadtxt(f"CRA/{stress[s]}/Fe_perfect_L80_av_pe_voro_vms.dat")[:,1:]
    defects[s] = np.loadtxt(f"CRA/{stress[s]}/Fe_perfect_L80_defects.dat")[:,1:]
    Ndeform[s] = np.loadtxt(f"CRA/{stress[s]}/Ndeform.dat")"""

"""np.save("data/Ndeform.npy", Ndeform)
np.save("data/w.npy", w)
np.save("data/dxa.npy", dxa)
np.save("data/pevol.npy", pevol)
np.save("data/vorovms.npy", vorovms)
np.save("data/defects.npy", defects)

for i in range(DPA.shape[0]):
    np.savetxt(f"data/Ndeform/{DPA[i]:.3f}DPA.dat", np.c_[stress, Ndeform[:,i]])
    np.savetxt(f"data/w/{DPA[i]:.3f}DPA.dat", np.c_[stress, w[:,i]])
    np.savetxt(f"data/pevol/{DPA[i]:.3f}DPA.dat", np.c_[stress, pevol[:,i]])

for i in range(DPA.shape[0]-1):
    np.savetxt(f"data/vorovms/{DPA[i+1]:.3f}DPA.dat", np.c_[stress, vorovms[:,i]])        # ENABLE THIS TO UPDATE ASCII FILES IF ADDING STRESS DATA POINTS

for i in range(step.shape[0]):
    np.savetxt(f"data/dxa/{step[i]/1000}DPA.dat", np.c_[stress, dxa[:,i]])
    np.savetxt(f"data/defects/{step[i]/1000}DPA.dat", np.c_[stress, defects[:,i]])"""

Ndeform = np.load("data/Ndeform.npy")
w = np.load("data/w.npy")
dxa = np.load("data/dxa.npy")
pevol = np.load("data/pevol.npy")
vorovms = np.load("data/vorovms.npy")
defects = np.load("data/defects.npy")

####  Strain  ####
##################
"""
e_names = {0: "$\epsilon_{xx}$", 2: "$\epsilon_{zz}$"}
for i in [0,2]:
    fig = plt.figure()
    plt.plot(stress, w[:,0,i], ".", label="0 DPA")
    plt.plot(stress, w[:,10,i], ".", label="0.01 DPA")
    plt.plot(stress, w[:,100,i], ".", label="0.1 DPA")
    plt.plot(stress, w[:,200,i], ".", label="0.2 DPA")
    plt.plot(stress, w[:,500,i], ".", label="0.5 DPA")
    plt.plot(stress, w[:,1000,i], ".", label="1.0 DPA")
    plt.plot(stress, w[:,-1,i], ".", label="3.0 DPA")
    plt.grid()
    #plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bars)")
    plt.ylabel(fr"{e_names[i]}", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/e/strain{i}vstresslin.png")
    plt.close()

    fig = plt.figure()
    plt.plot(DPA, w[0,:,i], label="-10000 bars")
    plt.plot(DPA, w[1,:,i], label="-5000 bars")
    plt.plot(DPA, w[3,:,i], label="-1000 bars")
    plt.plot(DPA, w[14,:,i], label="0 bars")
    plt.plot(DPA, w[-4,:,i], label="1000 bars")
    plt.plot(DPA, w[-2,:,i], label="5000 bars")
    plt.plot(DPA, w[-1,:,i], label="10000 bars")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.xscale("log")
    plt.xlabel("DPA")
    plt.ylabel(fr"{e_names[i]}", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/e/strain{i}vDPAlin.png")
    plt.close()

for i in [0, 1, 3, 14, -4, -2, -1]:
    fig = plt.figure()
    plt.plot(DPA, w[i,:,0], label=r"$\epsilon_{xx}$")
    plt.plot(DPA, w[i,:,2], label=r"$\epsilon_{zz}$")
    #plt.xscale("log")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("DPA")
    plt.ylabel(r"$\epsilon$", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/e/{stress[i]}barlin.png")
    plt.close()

for i in [0, 10, 100, -1]:
    fig = plt.figure()
    plt.plot(stress, w[:,i,0], ".", label=r"$\epsilon_{xx}$")
    plt.plot(stress, w[:,i,2], ".", label=r"$\epsilon_{zz}$")
    plt.grid()
    #plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bars)")
    plt.ylabel(r"$\epsilon$", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/e/{DPA[i]}DPAlin.png")
    plt.close()

####  Relaxation volume density  ####
#####################################
w_names = {3: "$\omega_{xx}$", 5: "$\omega_{zz}$"}
for i in [3,5]:
    fig = plt.figure()
    plt.plot(stress, w[:,0,i], ".", label="0 DPA")
    plt.plot(stress, w[:,10,i], ".", label="0.01 DPA")
    plt.plot(stress, w[:,100,i], ".", label="0.1 DPA")
    plt.plot(stress, w[:,200,i], ".", label="0.2 DPA")
    plt.plot(stress, w[:,500,i], ".", label="0.5 DPA")
    plt.plot(stress, w[:,1000,i], ".", label="1.0 DPA")
    plt.plot(stress, w[:,-1,i], ".", label="3.0 DPA")
    plt.grid()
    #plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bar)")
    plt.ylabel(fr"{w_names[i]}", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/w/w{i-3}vstresslin.png")
    plt.close()

    fig = plt.figure()
    plt.plot(DPA, w[0,:,i], label="-10000 bars")
    plt.plot(DPA, w[1,:,i], label="-5000 bars")  
    plt.plot(DPA, w[3,:,i], label="-1000 bars")
    plt.plot(DPA, w[14,:,i], label="0 bars")
    plt.plot(DPA, w[-4,:,i], label="1000 bars")
    plt.plot(DPA, w[-2,:,i], label="5000 bars")
    plt.plot(DPA, w[-1,:,i], label="10000 bars")
    #plt.xscale("log")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("DPA")
    plt.ylabel(fr"{w_names[i]}", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/w/w{i-3}vDPAlin.png")
    plt.close()

for i in [0, 1, 3, 14, -4, -2, -1]:
    fig = plt.figure()
    plt.plot(DPA, w[i,:,3], label=r"$\omega_{xx}$")
    plt.plot(DPA, w[i,:,5], label=r"$\omega_{zz}$")
    #plt.xscale("log")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("DPA")
    plt.ylabel(r"$\omega$", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/w/{stress[i]}barlin.png")
    plt.close()

for i in [0, 10, 100, -1]:
    fig = plt.figure()
    plt.plot(stress, w[:,i,3], ".", label=r"$\omega_{xx}$")
    plt.plot(stress, w[:,i,5], ".", label=r"$\omega_{zz}$")
    plt.grid()
    #plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bars)")
    plt.ylabel(r"$\omega$", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/w/{DPA[i]}DPAlin.png")
    plt.close()
"""
####  DXA  ####
###############
dxa_names = {0: "total", 1: "111", 2: "100", 3: "110", 4: "other"}
for i in range(5):
    fig = plt.figure()
    plt.plot(stress, dxa[:,0,i], "o", label=f"{step[0]/1000} DPA")
    plt.plot(stress, dxa[:,18,i], "o", label=f"{step[18]/1000} DPA")
    plt.plot(stress, dxa[:,108,i], "o", label=f"{step[108]/1000} DPA")
    plt.plot(stress, dxa[:,-1,i], "o", label=f"{step[-1]/1000} DPA")
    plt.grid()
    plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bars)")
    plt.ylabel(fr"Dislocation {dxa_names[i]} density ($\AA^{-2}$)")
    plt.tight_layout()
    plt.savefig(f"figures/dxa/{dxa_names[i]}vstress.png")
    plt.close()
    
    fig = plt.figure()
    plt.plot(step/1000, dxa[0,:,i], label="-10000 bars")
    plt.plot(step/1000, dxa[14,:,i], label="0 bars")
    plt.plot(step/1000, dxa[-1,:,i], label="10000 bars")
    #plt.xscale("log")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("DPA")
    plt.ylabel(fr"Dislocation {dxa_names[i]} density ($\AA^{-2}$)")
    plt.tight_layout()
    plt.savefig(f"figures/dxa/{dxa_names[i]}vDPA.png")
    plt.close()
    
for i in [0, 14, -1]:
    fig = plt.figure()
    plt.plot(step/1000, dxa[i,:,0], label="total")
    plt.plot(step/1000, dxa[i,:,1], label="111")
    plt.plot(step/1000, dxa[i,:,2], label="100")
    plt.plot(step/1000, dxa[i,:,3], label="110")
    plt.plot(step/1000, dxa[i,:,4], label="other")
    #plt.xscale("log")
    plt.grid()
    plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("DPA")
    plt.ylabel(r"Dislocation density ($\AA^{-2}$)")
    plt.tight_layout()
    plt.savefig(f"figures/dxa/{stress[i]}bar.png")
    plt.close()

for i in [0, 10, 100, -1]:
    fig = plt.figure()
    plt.plot(stress, dxa[:,i,0], "o", label="total")
    plt.plot(stress, dxa[:,i,1], "o", label="111")
    plt.plot(stress, dxa[:,i,2], "o", label="100")
    plt.plot(stress, dxa[:,i,3], "o", label="110")
    plt.plot(stress, dxa[:,i,4], "o", label="other")
    plt.grid()
    plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bars)")
    plt.ylabel(r"Dislocation density ($\AA^{-2}$)")
    plt.tight_layout()
    plt.savefig(f"figures/dxa/{DPA[i]}DPA.png")
    plt.close()
"""
####  Other ####
################
other_names = {0: "Average Potential Energy (eV)", 1: "S.D. Potential Energy (eV)", 2: "Average Voronoi Volume (\AA^3)", 
    3: "S.D. Voronoi Volume ($\AA^3$)", 4: "Average von Mises Stress (Mbar)", 5: "S.D. von Mises Stress (Mbar)"}
for i in range(6):
    fig = plt.figure()
    plt.plot(stress, vorovms[:,0,i], "o", label=f"0 DPA")
    plt.plot(stress, vorovms[:,10,i], "o", label=f"0.01 DPA")
    plt.plot(stress, vorovms[:,100,i], "o", label=f"0.1 DPA")
    plt.plot(stress, vorovms[:,-1,i], "o", label=f"3 DPA")
    plt.grid()
    plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bar)")
    plt.ylabel(rf"{other_names[i]}")
    plt.tight_layout()
    plt.savefig(fr"figures/other/other{i}vstress.png")
    plt.close()
    
    fig = plt.figure()
    plt.plot(DPA[1:], vorovms[0,:,i], label="-10000 bars")
    plt.plot(DPA[1:], vorovms[14,:,i], label="0 bars")
    plt.plot(DPA[1:], vorovms[-1,:,i], label="10000 bars")
    plt.xscale("log")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("DPA")
    plt.ylabel(fr"{other_names[i]}")
    plt.tight_layout()
    plt.savefig(f"figures/other/other{i}vDPA.png")
    plt.close()
    
for i in [0, 14, -1]:
    for j in range(6):
        fig = plt.figure()
        plt.plot(DPA[1:], vorovms[i,:,j])
        plt.grid()
        plt.xscale("log")
        plt.ylabel(fr"{other_names[j]}")
        plt.xlabel("DPA")
        plt.tight_layout()
        plt.savefig(f"figures/other/other{j}{stress[i]}bar.png")
        plt.close()

for i in [0, 10, 100, -1]:
    for j in range(6):
        fig = plt.figure()
        plt.plot(stress, vorovms[:,i,j], "o")
        plt.grid()
        plt.ylabel(fr"{other_names[j]}")
        plt.xscale("symlog")
        plt.xlabel("stress (bars)")
        plt.tight_layout()
        plt.savefig(f"figures/other/other{j}{str(DPA[i])}DPA.png")
        plt.close()
    
####  C15  ####
###############

for i in stress:
    hist = np.loadtxt(f"CRA/{i}/av_C15_hist.dat")
    hist = hist[:,~np.isnan(hist).all(axis=0)]
    X,Y=np.meshgrid(step/1000, np.arange(0, hist.shape[1], 1))
    fig = plt.figure()
    plt.pcolormesh(X, Y, hist.T, cmap="inferno")
    cbar = plt.colorbar()
    cbar.set_label("Count")
    plt.xscale("log")
    plt.xlabel("DPA")
    plt.ylabel("cluster size")
    plt.tight_layout()
    plt.savefig(f"figures/C15/{i}bar.png", bbox_inches='tight')
    plt.close()

    c15vol = np.loadtxt(f"CRA/{i}/avC15vol.dat")
    fig = plt.figure()
    plt.plot(step/1000, c15vol)
    plt.xlabel("DPA")
    plt.ylabel("Volume concentration (%)")
    plt.grid()
    plt.savefig(f"figures/C15/{i}barvol.png", bbox_inches="tight")
    plt.close()

####  Defects  ####
###################
for i in [0, 14, -1]:
    fig = plt.figure()
    plt.plot(step/1000, 100*defects[i, :, 0]/1024000, '.', label=f"vacancies at {stress[i]} bars")
    plt.plot(step/1000, 100*defects[i, :, 1]/1024000, '.', label=f"interstitials at {stress[i]} bars")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel("DPA")
    plt.ylabel("Percentage content")
    plt.tight_layout()
    plt.savefig(f"figures/defects/{stress[i]}bar.png")
    plt.close()
    
for i in [0, 18, 108, -1]:
    fig = plt.figure()
    plt.plot(stress, 100*defects[:, i, 0]/1024000, '.', label=f"vacancies at {step[i]/1000}DPA")
    plt.plot(stress, 100*defects[:, i, 1]/1024000, '.', label=f"interstitials at {step[i]/1000}DPA")
    plt.grid()
    plt.xscale("symlog")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.xlabel("DPA")
    plt.ylabel("Percentage content")
    plt.tight_layout()
    plt.savefig(f"figures/defects/{step[i]/1000}DPA.png")
    plt.close()
"""
#### Plastic Deformation ####
#############################
n_names = {0: "$n_{xx}$", 1:"$n_{yy}$", 2: "$n_{zz}$"}
for i in [0,1,2]:
    fig = plt.figure()
    plt.plot(stress, Ndeform[:,0,i], ".", label="0 DPA")
    plt.plot(stress, Ndeform[:,10,i], ".", label="0.01 DPA")
    plt.plot(stress, Ndeform[:,100,i], ".", label="0.1 DPA")
    plt.plot(stress, Ndeform[:,200,i], ".", label="0.2 DPA")
    plt.plot(stress, Ndeform[:,500,i], ".", label="0.5 DPA")
    plt.plot(stress, Ndeform[:,1000,i], ".", label="1.0 DPA")
    plt.plot(stress, Ndeform[:,-1,i], ".", label="3.0 DPA")
    plt.grid()
    plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bars)")
    plt.ylabel(fr"{n_names[i]}", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/Ndeform/n{i}vstress.png")
    plt.close()

    fig = plt.figure()
    plt.plot(DPA, Ndeform[0,:,i], label="-10000 bars")
    plt.plot(DPA, Ndeform[1,:,i], label="-5000 bars")
    plt.plot(DPA, Ndeform[3,:,i], label="-1000 bars")
    plt.plot(DPA, Ndeform[14,:,i], label="0 bars")
    plt.plot(DPA, Ndeform[-4,:,i], label="1000 bars")
    plt.plot(DPA, Ndeform[-2,:,i], label="5000 bars")
    plt.plot(DPA, Ndeform[-1,:,i], label="10000 bars")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xscale("log")
    plt.xlabel("DPA")
    plt.ylabel(fr"{n_names[i]}", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/Ndeform/n{i}vDPA.png")
    plt.close()

for i in [0, 1, 3, 14, -4, -2, -1]:
    fig = plt.figure()
    plt.plot(DPA, Ndeform[i,:,0], label=r"$n_{xx}$")
    plt.plot(DPA, Ndeform[i,:,1], label=r"$n_{yy}$")
    plt.plot(DPA, Ndeform[i,:,2], label=r"$n_{zz}$")
    #plt.xscale("log")
    plt.grid()
    plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("DPA")
    plt.ylabel("n", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/Ndeform/{stress[i]}barlin.png")
    plt.close()

for i in [0, 10, 100, -1]:
    fig = plt.figure()
    plt.plot(stress, Ndeform[:,i,0], ".", label=r"$n_{xx}$")
    plt.plot(stress, Ndeform[:,i,1], ".", label=r"$n_{yy}$")
    plt.plot(stress, Ndeform[:,i,2], ".", label=r"$n_{zz}$")
    plt.grid()
    plt.xscale("symlog")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("stress (bars)")
    plt.ylabel("n", fontsize=20)
    plt.tight_layout()
    plt.savefig(fr"figures/Ndeform/{DPA[i]}DPA.png")
    plt.close()