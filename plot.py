import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

DPA = np.linspace(0, 3, 3001)
step = np.r_[np.linspace(1, 10, 10), np.linspace(20, 1000, 99), np.linspace(1050, 3000, 40)].astype(int)
stress = np.array([-10000.0, -2000.0, -1000.0, -500.0, -200.0, -100.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 10000.0])
w = np.load("data/w.npy")
dxa = np.load("data/dxa.npy")
pevol = np.load("data/pevol.npy")
vorovms = np.load("data/vorovms.npy")
defects = np.load("data/defects.npy")

fig = plt.figure()
plt.plot(DPA[1:], vorovms[13,:,5], "o", label="")
plt.grid()
plt.xscale("log")
plt.legend()
plt.xlabel("DPA")
plt.ylabel(f"Standard deviation of von mises stress")
plt.savefig(f"fig.png")
plt.close()
