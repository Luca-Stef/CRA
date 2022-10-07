import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


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


def fstress(stress, m, c):
    return m*stress + c

def fdpa(dpa, c1, c2, c3):
    return c1*dpa/(c2 + c3*dpa) 
    #return c1 - c2*np.exp(-c3*dpa)
    #return c1*np.log(c2 + c3*dpa)/(c4 + c5*np.log(c6 + c7*dpa))

w_comp = sys.argv[1]
indep_var = sys.argv[2]
fixed_var = float(sys.argv[3])
log_or_lin = sys.argv[4]
stress = np.array([-10000.0, -5000.0, -2000.0, -1000.0, -500.0, -200.0, -100.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0])
DPA = np.linspace(0, 3, 3001)
w = np.load("data/w.npy")
p = []

if indep_var == "s":

    xdata = stress
    func = fstress
    fix_idx = np.where(DPA == fixed_var)[0][0]
    ydata = w[:,fix_idx,3] if w_comp == "x" else w[:,fix_idx,5]
    yerr = w[:,fix_idx,6] if w_comp == "x" else w[:,fix_idx,8]
    
    fig = plt.figure()
    #plt.plot(xdata, ydata, '.', label=f'data at {fixed_var} DPA')
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='none', label=f'data at {fixed_var} DPA')
    popt, pcov = curve_fit(func, xdata, ydata)
    print('Fitted parameters:')
    print(popt)
    err = (ydata - func(xdata, *popt))**2
    rms = np.sqrt(np.mean(err))
    plt.plot(xdata, func(xdata, *popt), 'g--', label=rf"${popt[0]:.2g}*stress + {popt[1]:.2g}$")
    plt.plot([], [], " ", label=f"rms = {rms:.2g}")
    plt.xlabel("Stress (bar)")
    if log_or_lin == "log": plt.xscale("symlog")
    plt.ylabel(r'$\omega_{zz}$' if w_comp == "z" else r"$\omega_{xx}$", fontsize=20) 
    plt.grid()
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    plt.savefig(f"figures/fits/w{w_comp}fitstress{fixed_var}dpa{log_or_lin}.png")
    
    fig = plt.figure()
    plt.plot(xdata, err*10000000, ".")
    plt.xlabel("Stress (bar)")
    if log_or_lin == "log": plt.xscale("symlog")
    plt.ylabel(r"Error $(\times 10^{-7})$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/fits/w{w_comp}fitstress{fixed_var}dpa_err{log_or_lin}.png")
    
    for idx in range(DPA.shape[0]):
        ydata = w[:,idx,3] if w_comp == "x" else w[:,idx,5]
        popt, pcov = curve_fit(func, xdata, ydata)
        p.append(popt)
    
    p = np.array(p)
    fig = plt.figure()
    plt.plot(DPA, p[:,0]*1000000)
    plt.xlabel("DPA")
    plt.ylabel(r"m ($\times 10^{-6}bar^{-1}$)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/fits/w{w_comp}fitstressm.png")
    
    fig = plt.figure()
    plt.plot(DPA, p[:,1])
    plt.xlabel("DPA")
    plt.ylabel("c")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/fits/w{w_comp}fitstressc.png")

else:
    
    xdata = DPA
    func = fdpa
    fix_idx = np.where(stress == fixed_var)[0][0]
    ydata = w[fix_idx,:,3] if w_comp == "x" else w[fix_idx,:,5]
    yerr = w[fix_idx,:,6] if w_comp == "x" else w[fix_idx,:,8]

    fig = plt.figure()
    plt.plot(xdata, ydata, 'b-', label=f'data at {fixed_var} bars')
    #plt.errorbar(xdata, ydata, yerr=yerr, fmt='none', label=f'data at {fixed_var} bars')
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=20000)
    print('Fitted parameters:')
    print(popt)
    err = (ydata - func(xdata, *popt))**2
    rms = np.sqrt(np.mean(err))
    plt.plot(xdata, func(xdata, *popt), 'g--', label=r'$\frac{%0.2g*DPA}{%0.2g + %0.2g*DPA}$' % tuple(popt))
    #plt.plot(xdata, func(xdata, *popt), 'g--', label=r'$%5.3f - %5.3f e^{-%5.3f*DPA}$' % tuple(popt))
    plt.plot([], [], " ", label=f"rms = {rms:.2g}")
    plt.xlabel('DPA')
    plt.ylabel(r"$\omega_{zz}$" if w_comp == "z" else r"$\omega_{xx}$", fontsize=20)
    plt.grid()
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    plt.savefig(f"figures/fits/w{w_comp}fitdpa{fixed_var}bar.png")    
    
    fig = plt.figure()
    plt.plot(xdata, err*10000000, ".")
    plt.xlabel("DPA")
    plt.xscale("symlog")
    plt.ylabel(r"Error $\times 10^{-7}$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/fits/w{w_comp}fitdpa{fixed_var}bar_err.png")