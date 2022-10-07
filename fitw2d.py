import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.ticker as mticker


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# The two-dimensional domain of the fit.
stress = np.array([-10000.0, -5000.0, -2000.0, -1000.0, -500.0, -200.0, -100.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 
                    0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0], dtype=np.float128)
DPA = np.linspace(0, 3, 3001, dtype=np.float128)
X, Y = np.meshgrid(DPA, stress)
comp = sys.argv[1]
w = np.load("data/w.npy")
if comp == "x":
    w = w[:,:,3]
elif comp == "z":
    w = w[:,:,5] 
    
def log_tick_formatter(val, pos=None):
    return r"$10^{{{:.0f}}}$".format(val)
    
def f(stress, dpa, c1, c2, c3, c4, c5, c6):
    #return c1 + c2*stress + c3*dpa + c4*stress*dpa + c5*stress**2 + c6*dpa**2
    return c1*stress*dpa/(c2 + c3*dpa) + c4*dpa/(c5 + c6*dpa)
    #return c1*stress*np.log(c2 + c3*dpa)/(c4 + c5*np.log(c6 + c7*dpa)) + c8*np.log(c9 + c10*dpa)/(c11 + c12*np.log(c13 + c14*dpa))

# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X, Y, w, ".")
plt.show()
#ax.yaxis.set_scale('symlog')
#plt.savefig("fig.png")

# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _f(M, *args):
    y, x = M
    arr = np.zeros(x.shape, dtype=np.float128)
    arr = f(x, y, *args)
    return arr

# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((X.ravel(), Y.ravel()))
# Do the fit, using our custom function which understands our
# flattened (ravelled) ordering of the data points.
if comp == "x":
    p0 = [2.55584274e-01, 9.60360120e+04, 1.31128354e+05, 1.80176087e+04, 1.53290126e+05, 1.66847778e+06]
elif comp == "z":
    p0 = [2.27148434e-04, -3.80197941e+01, -5.49861588e+01, 5.78762616e+02, 5.80882755e+03, 5.25296964e+04]

popt, pcov = curve_fit(_f, xdata, w.ravel(), p0, maxfev=50000)
fit = np.zeros(w.shape, dtype=np.float128)
fit = f(Y, X, *popt)
print(f'Fitted c1*stress*dpa/(c2 + c3*dpa) + c4*dpa/(c5 + c6*dpa) to {comp} data with parameters:')
print(popt)

err = np.sqrt((w - fit)**2)
rms = np.mean(err)
print('RMS residual =', rms)

# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_wireframe(X, Y/1000, fit, alpha=1)

#ax.plot_surface(np.log10(X), np.r_[-np.log10(-Y[:13]), np.log10(Y[13:])], fit)   # log scales 
#ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
#ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
  
ax.set_xlabel('DPA')
ax.set_ylabel('Stress (1000 bars)')
text = r"$\omega(\sigma, DPA) = m(DPA)\sigma + c(DPA)$" + "\n\n RMS residual = " + str(rms) 

if comp == "x":
    cset = ax.contourf(X, Y/1000, err, zdir='z', offset=-0.01, cmap="plasma")
    cbar = fig.colorbar(cset, ax=ax, shrink=0.9, pad=0.1)
    cbar.set_label('Error', rotation=270, labelpad=10)
    ax.set_zlim(-0.01, np.max(fit))
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\bar{\omega}_{11}$', fontsize=20)
    ax.zaxis.labelpad=10
    #ax.text(1,15,0.035, text, fontsize=10)
    ax.view_init(elev=22, azim=-131)

elif comp == "z":
    cset = ax.contourf(X, Y/1000, err, zdir='z', offset=-0.03, cmap="plasma")
    cbar = fig.colorbar(cset, ax=ax, shrink=0.9, pad=0.15)
    cbar.set_label('Error', rotation=270, labelpad=10)
    ax.set_zlim(-0.03, np.max(fit))
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\bar{\omega}_{33}$', fontsize=20)
    ax.zaxis.labelpad=10
    #ax.text(3,15,0.10, text, fontsize=10)
    ax.view_init(elev=29, azim=116)

#plt.show()
#plt.tight_layout()
plt.savefig(f"figures/fits/fitw{'x' if comp == 'x' else 'z'}.png", bbox_inches="tight")

# Plot the test data as a 2D image and the fit as overlaid contours.
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(w, origin='bottom', cmap='plasma',
          #extent=(x.min(), x.max(), y.min(), y.max()))
#ax.contour(X, Y, fit, colors='w')
#plt.show()