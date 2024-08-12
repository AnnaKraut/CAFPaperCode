import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

#################### Define Functions ####################

def GFmax(n_M,s_GFmax):
    return s_GFmax * n_M

def GFequil(n_D,GF_max,s_GFequil,D_crit):
    return GF_max / (1 + np.exp(-s_GFequil * (n_D - D_crit)))

def C50(n_G,C50_max,s_C50,GF_crit):
    return C50_max / (1 + np.exp(-s_C50 * (n_G - GF_crit)))

def NetGrowth(n_D,R_max,R_min,s_NetGrowth,C_50):
    return R_min + (R_max - R_min) / (1 + np.power((n_D / C_50),s_NetGrowth))

#Parameters

G_max = 27.1
s_GFequil = 65.7
D_crit = 0.0049
C50_max = 7.34
s_C50 = 0.0078
GF_crit = 379.4
R_max = 0.0147
R_min = -0.0079
s_NetGrowth = 0.93

#################### Combination therapy - EGF blocker ####################

G_max_range = np.linspace(12.1,87.1,6)
D_range = np.linspace(0.001,30,10000)
D_range2 = np.linspace(0.0001,100,1000)
G_range = np.linspace(0,150,1000)

plt.plot(np.full(1000,np.log10(0.86)),G_range,'k:')
plt.plot(np.log10(D_range),np.full(10000,G_max),'k:')
for G in G_max_range:
    plt.plot(np.log10(D_range),GFequil(D_range,G,s_GFequil,D_crit),'C1',alpha = 1 - (G-12.1)*0.8/75)
plt.plot(np.log10(D_range),GFequil(D_range,G_max,s_GFequil,D_crit),'C0',label='$\overline{G}$ equil. EGF conc. (dep. on $D$)')
plt.plot(np.log10(C50(G_range,C50_max,s_C50,GF_crit) * np.power(- R_max/R_min,1/s_NetGrowth)),G_range,'C0--',label='$D_{crit}$ CTX conc. (dep. on $G$)')
plt.xlabel('CTX concentration $D$ [μg/mL]',size=16)
plt.xticks([-3,-2,-1,np.log10(0.86),1],[0.001,0.01,0.1,0.86,10],size=16)
plt.ylabel('EGF concentration $G$',size=16)
plt.yticks([G_max],['$\overset{G^{\max}}{\downarrow}$'],size=23)
plt.legend(fontsize=13,loc='upper left')
plt.show()

def D_zero(n_G):
        return C50(n_G,C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)

# generate 2d grids for the x & y bounds (x=log10(n_D),y=n_M)
x, y = np.meshgrid(np.linspace(-3, 2, 1000), np.linspace(0, 400, 1000))

z = NetGrowth(pow(10,x),R_max,R_min,s_NetGrowth,C50(GFequil(pow(10,x),y,s_GFequil,D_crit),C50_max,s_C50,GF_crit))
divnorm=colors.TwoSlopeNorm(vmin=R_min, vcenter=0., vmax=0.015)

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='seismic', norm=divnorm)
ax.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(c, ax=ax,ticks=[-0.0075,-0.005,-0.0025,0,0.005,0.01,0.015])
cbar.set_label(label='CRC net growth rate [1/h]',size=13)
cbar.ax.set_yticklabels(labels=[-0.0075,-0.0050,-0.0025,0.0000,0.0050,0.0100,0.0150],size=13)
plt.xlabel('CTX concentration $D$ [μg/mL]',size=16)
plt.xticks([-3,-2,-1,np.log10(0.86),1,2],[0.001,0.01,0.1,0.86,10,100],size=16)
plt.ylabel('max. EGF conc. $G^{\max}$ [pg/mL]',size=16)
plt.yticks([0,50,100,200,300,400], size=16)
plt.plot(np.log10(D_range2),0*D_range2+G_max,'k')

plt.show()