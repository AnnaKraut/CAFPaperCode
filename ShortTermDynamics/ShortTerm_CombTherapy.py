import numpy as np
from scipy.integrate import odeint
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

s_GFmax = 0.35
s_GFequil = 5
D_crit = 0.7
C50_max = 10
s_C50 = 0.03
GF_crit = 300
R_max = 0.015
R_min = -0.005
s_NetGrowth = 2

#################### Combination therapy - Reduced stromal cells ####################

n_M_range = np.linspace(100,1600,6)
D_range = np.linspace(0.001,30,1000)
D_range2 = np.linspace(0.0001,100,1000)
G_range = np.linspace(0,600,1000)

plt.plot(np.full(1000,np.log10(D_crit)),G_range,'k:')
plt.plot(np.log10(D_range),np.full(1000,GFmax(1000,s_GFmax)),'k:')
for M in n_M_range:
    plt.plot(np.log10(D_range),GFequil(D_range,GFmax(M,s_GFmax),s_GFequil,D_crit),'C1',alpha = 1 - (M-100)*0.8/1500)
plt.plot(np.log10(D_range),GFequil(D_range,GFmax(1000,s_GFmax),s_GFequil,D_crit),'C0',label='$\overline{G}$ equil. GF conc. (dep. on $D$)')
plt.plot(np.log10(C50(G_range,C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)),G_range,'C0--',label='$D_{crit}$ drug conc. (dep. on $G$)')
plt.xlabel('drug concentration $D$',size=16)
plt.xticks([np.log10(D_crit)],['$\hat{D}$'],size=16)
plt.ylabel('growth factor conc. $G$',size=16)
plt.yticks([GFmax(1000,s_GFmax)],['$\overset{G^{\max}(S)}{\downarrow}$'],size=23)
plt.legend(loc='upper left',fontsize=13)
plt.show()

def D_zero(n_G):
        return C50(n_G,C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)
M_range = np.linspace(0,10000,1000)

# generate 2d grids for the x & y bounds (x=log10(n_D),y=n_M)
x, y = np.meshgrid(np.linspace(-4, 2, 1000), np.linspace(0, 10000, 1000))

z = NetGrowth(pow(10,x),R_max,R_min,s_NetGrowth,C50(GFequil(pow(10,x),GFmax(y,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit))

divnorm=colors.TwoSlopeNorm(vmin=R_min, vcenter=0., vmax=R_max)

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='seismic', norm=divnorm)
ax.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(c, ax=ax,ticks=[-0.005,0,0.015])
cbar.set_label(label='cancer growth rate',size=13)
cbar.ax.set_yticklabels(['neg','0','pos'],size=13)

plt.xlabel('drug concentration $D$',size=16)
plt.xticks([])
plt.ylabel('# stromal cells $S$',size=16)
plt.yticks([])
plt.plot(0*M_range+np.log10(D_zero(0)),M_range,'k',linestyle='dotted')
plt.plot(np.log10(D_zero(GFmax(M_range,s_GFmax))),M_range,'k',linestyle='dashed')
plt.plot(0*M_range+np.log10(D_crit),M_range,'k',linestyle='dashdot')
plt.plot(np.log10(D_range2),0*D_range2+1000,'k')

plt.show()

#################### Combination therapy - Reduced stromal cell sensititvity ####################

n_M = 1000
D_crit_range = np.linspace(0.1,1.1,6)
D_range = np.linspace(0.001,30,1000)
D_range2 = np.linspace(0.0001,100,1000)
G_range = np.linspace(0,500,1000)

plt.plot(np.full(1000,np.log10(D_crit)),G_range,'k:')
plt.plot(np.log10(D_range),np.full(1000,GFmax(1000,s_GFmax)),'k:')
for D in D_crit_range:
    plt.plot(np.log10(D_range),GFequil(D_range,GFmax(n_M,s_GFmax),s_GFequil,D),'C1',alpha = 0.2 + (D-0.1)*0.8)
plt.plot(np.log10(D_range),GFequil(D_range,GFmax(n_M,s_GFmax),s_GFequil,D_crit),'C0',label='$\overline{G}$ equil. GF conc. (dep. on $D$)')
plt.plot(np.log10(C50(G_range,C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)),G_range,'C0--',label='$D_{crit}$ drug conc. (dep. on $G$)')
plt.xlabel('drug concentration $D$',size=16)
plt.xticks([np.log10(D_crit)],[r'$\hat{D}\rightarrow$'],size=16)
plt.ylabel('growth factor conc. $G$',size=16)
plt.yticks([GFmax(1000,s_GFmax)],['$G^{\max}(S)$'],size=16)
plt.legend(fontsize=13)
plt.show()

# generate 2d grids for the x & y bounds (x=log10(n_D),y=log10(D_crit))
x, y = np.meshgrid(np.linspace(-4, 2, 1000), np.linspace(-4, 2, 1000))

z = NetGrowth(pow(10,x),R_max,R_min,s_NetGrowth,C50(GFequil(pow(10,x),GFmax(n_M,s_GFmax),s_GFequil,pow(10,y)),C50_max,s_C50,GF_crit))
divnorm=colors.TwoSlopeNorm(vmin=R_min, vcenter=0., vmax=R_max)

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='seismic', norm=divnorm)
ax.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(c, ax=ax,ticks=[-0.005,0,0.015])
cbar.set_label(label='cancer growth rate',size=13)
cbar.ax.set_yticklabels(['neg','0','pos'],size=13)
plt.xlabel('drug concentration $D$',size=16)
plt.xticks([])
plt.ylabel('threshold drug conc. $\hat{D}$',size=16)
plt.yticks([])
plt.plot(0*D_range2+np.log10(D_zero(0)),np.log10(D_range2),'k',linestyle='dotted')
plt.plot(0*D_range2+np.log10(D_zero(GFmax(n_M,s_GFmax))),np.log10(D_range2),'k',linestyle='dashed')
plt.plot(np.log10(D_range2),np.log10(D_range2),'k',linestyle='dashdot')
plt.plot(np.log10(D_range2),0*D_range2+np.log10(0.7),'k')

plt.show()

#################### Combination therapy - Reduced growth factor efficacy ####################

GF_crit_range = np.linspace(150,400,6)
D_range = np.linspace(0.0001,30,1000)
G_range = np.linspace(0,500,1000)
D_range2 = np.linspace(0.0001,100,1000)

plt.plot(np.log10(D_range),np.full(1000,GF_crit),'k:')
plt.plot(np.log10(D_range),GFequil(D_range,GFmax(n_M,s_GFmax),s_GFequil,D_crit),'C0',label='$\overline{G}$ equil. GF conc. (dep. on $D$))')
for G in GF_crit_range:
    plt.plot(np.log10(C50(G_range,C50_max,s_C50,G) * np.sqrt(- R_max/R_min)),G_range,'C1--',alpha = 0.2+(G-150)*0.8/250)
plt.plot(np.log10(C50(G_range,C50_max,s_C50,300) * np.sqrt(- R_max/R_min)),G_range,'C0--',label='$D_{crit}$ drug conc. (dep. on $G$)')
plt.xlabel('drug concentration $D$',size=16)
plt.xticks([],size=16)
plt.ylabel('growth factor conc. $G$',size=16)
plt.yticks([GF_crit],[r'$\overset{\hat{G}}{\uparrow}$'],size=23)
plt.legend(fontsize=13)
plt.show()

def D_zero(n_G,n_G_crit):
        return C50(n_G,C50_max,s_C50,n_G_crit) * np.sqrt(- R_max/R_min)

# generate 2d grids for the x & y bounds (x=log10(n_D),y=GF_crit)
x, y = np.meshgrid(np.linspace(-4, 2, 1000), np.linspace(0, 500, 1000))

z = NetGrowth(pow(10,x),R_max,R_min,s_NetGrowth,C50(GFequil(pow(10,x),GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,y))
divnorm=colors.TwoSlopeNorm(vmin=R_min, vcenter=0., vmax=R_max)

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='seismic', norm=divnorm)
ax.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(c, ax=ax,ticks=[-0.005,0,0.015])
cbar.set_label(label='cancer growth rate',size=13)
cbar.ax.set_yticklabels(['neg','0','pos'],size=13)
plt.xlabel('drug concentration $D$',size=16)
plt.xticks([])
plt.ylabel('threshold GF conc. $\hat{G}$',size=16)
plt.yticks([])
plt.plot(np.log10(D_zero(0,G_range)),G_range,'k',linestyle='dotted')
plt.plot(np.log10(D_zero(GFmax(n_M,s_GFmax),G_range)),G_range,'k',linestyle='dashed')
plt.plot(0*G_range+np.log10(D_crit),G_range,'k',linestyle='dashdot')
plt.plot(np.log10(D_range2),0*D_range2+6,'k')

plt.show()