"""" Plot of steady state energy rate per zplus energy against chi_A"""
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
from vis._matplotlib import finalize_figure, import_pyplot


chi_A= [0.65, 0.99, 1.63, 4.84, 8.08]
zplus= [0.43, 0.72, 1.27, 3.82, 6.36]
rate= [0.005, 0.019, 0.092, 0.615, 1.205]
e_plus= np.zeros(len(zplus))
for i in range(len(e_plus)):
    e_plus[i]= rate[i] / (1/4 * zplus[i] **2)

plt = import_pyplot(show=False)

fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
  
ax.scatter(chi_A, e_plus, label= "simulations", color= "tab:blue")
ax.scatter(chi_A, rate, label= "simulations (unscaled)", color= "tab:red")
ax.set_xlabel(r"$\chi_A$", fontsize=18)
ax.set_ylabel(r"Heating Efficiency Rate", fontsize=18)
# ax.set_ylim(0, 0.3)
# ax.set_xlim(0, 10)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)

finalize_figure(fig, output_path= Path("shear_heating_rate.png"), plt=plt, show=False)


