"""" Plot of steady state energy rate per Alfven energy against cross helicity"""
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
from vis._matplotlib import finalize_figure, import_pyplot


l_perp=[0.63, 0.65, 0.71, 0.78, 0.92, 0.98]
l_para=[1.12, 1.11, 1.27, 1.40,1.48, 1.42]
u=[0.77, 0.71, 0.90, 1.68,1.55, 2.21 ]
b=[1.09, 1.18, 1.11, 1.21,1.62, 2.04]
ch=[-0.01, 0.23, 0.54,0.73,0.92, 0.97]
num_rate=[0.217, 0.243, 0.23, 0.273, 0.280,0.428]

chi_A= np.zeros(len(u))
for i in range(len(chi_A)):
  chi_A[i]= l_para[i] * u[i]/l_perp[i]

e_plus= np.zeros(len(u))
for i in range(len(e_plus)):
    e_plus[i]= num_rate[i] / (1/2 * u[i] **2 +b[i] **2)

plt = import_pyplot(show=False)

fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
  
ax.scatter(ch, e_plus, label= "simulations", color= "tab:blue")
ax.scatter(ch, num_rate, label= "simulations (unscaled)", color= "tab:red")
ax.set_xlabel(r"$\sigma_c$", fontsize=18)
ax.set_ylabel(r"Heating Efficiency Rate", fontsize=18)
# ax.set_ylim(0, 0.3)
# ax.set_xlim(0, 10)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)

finalize_figure(fig, output_path= Path("imb_heating_rate.png"), plt=plt, show=False)


