import numpy as np
import matplotlib.pyplot as plt

#################### Define functions ####################

def GFmax(n_M,s_GFmax):
    return s_GFmax * n_M

def GFequil(n_D,GF_max,s_GFequil,D_crit):
    return GF_max / (1 + np.exp(-s_GFequil * (n_D - D_crit)))

def C50(n_G,C50_max,s_C50,GF_crit):
    return C50_max / (1 + np.exp(-s_C50 * (n_G - GF_crit)))

def NetGrowth(n_D,R_max,R_min,s_NetGrowth,C_50):
    return R_min + (R_max - R_min) / (1 + np.power((n_D / C_50),s_NetGrowth))

#################### Plot functions ####################

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

#Plot ranges

M_range = np.linspace(0,1500,1000)
D_range = np.linspace(0.001,10,1000)
D_range_log = np.linspace(-2,2,1000)
G_range = np.linspace(0,500,1000)

#Plot functions

plt.rcParams['figure.figsize']=[6,4]

plt.title('slope = s_G^max',size=16)
plt.plot(M_range,GFmax(M_range,s_GFmax))
plt.plot(M_range,0*M_range+350,'C0--')
plt.xticks([])
plt.xlabel('number of micro-environment cells',size=16)
plt.yticks([350],['n_G^max'],size=16)
plt.ylabel('max conc. growth factor',size=16)
plt.show()

#plt.title('slope = s_G',size=16)
plt.plot(D_range,np.full(1000,GFmax(50,s_GFmax)),'k:')
plt.plot(np.full(100,D_crit),np.linspace(0,GFmax(50,s_GFmax),100),'k:')
plt.plot(D_range,GFequil(D_range,GFmax(50,s_GFmax),s_GFequil,D_crit))
plt.xticks([D_crit],['$\hat{D}$'],size=16)
plt.xlabel('drug concentration $D$',size=16)
plt.yticks([GFmax(50,s_GFmax)],['$G^{\max}$'],size=16)
plt.ylabel('equil. growth factor conc. $\overline{G}$',size=16)
plt.show()

#plt.title('slope = s_C50',size=16)
plt.plot(G_range,np.full(1000,C50_max),'k:')
plt.plot(np.full(100,GF_crit),np.linspace(0,C50_max,100),'k:')
plt.plot(G_range,C50(G_range,C50_max,s_C50,GF_crit))
plt.xticks([GF_crit],['$\hat{G}$'],size=16)
plt.xlabel('growth factor concentration $G$',size=16)
plt.yticks([C50_max],['$D_{50}^{\max}$'],size=16)
plt.ylabel('$D_{50}$ of growth rate $r_C$',size=16)
plt.show()

#plt.title('slope = s',size=16)
plt.plot(D_range_log,np.full(1000,R_max),'k:')
plt.plot(D_range_log,np.full(1000,0),'k:')
plt.plot(D_range_log,np.full(1000,R_min),'k:')
plt.plot(np.full(100,np.log10(C50(250,C50_max,s_C50,GF_crit))),np.linspace(R_min,R_max,100),'k:')
plt.plot(np.full(100,np.log10(C50(250,C50_max,s_C50,GF_crit)*np.sqrt(- R_max/R_min))),np.linspace(R_min,R_max,100),'k:')
plt.plot(D_range_log,NetGrowth(pow(10,D_range_log),R_max,R_min,s_NetGrowth,C50(250,C50_max,s_C50,GF_crit)),'C0',label='low $G$')
plt.plot(D_range_log,NetGrowth(pow(10,D_range_log),R_max,R_min,s_NetGrowth,C50(350,C50_max,s_C50,GF_crit)),'C0--',label='high $G$')
plt.arrow(np.log10(C50(250,C50_max,s_C50,GF_crit))+0.05,0.005,np.log10(C50(350,C50_max,s_C50,GF_crit))-np.log10(C50(250,C50_max,s_C50,GF_crit))-0.1,0,length_includes_head=True,width = 0.00015,head_width=0.0007,head_length=0.12,color='C1')
plt.xticks([np.log10(C50(250,C50_max,s_C50,GF_crit)),np.log10(C50(250,C50_max,s_C50,GF_crit)*np.sqrt(- R_max/R_min))],['$D_{50}$     ','     $D_{crit}$'],size=16)
plt.xlabel('drug concentration $D$',size=16)
plt.yticks([R_min,0,R_max],['$r_C^{\min}$','$0$','$r_C^{\max}$'],size=16)
plt.ylabel('cancer growth rate $r_C$',size=16)
plt.legend(fontsize=13,loc='center left')
plt.show()

plt.plot(np.full(100,GFequil(0.4,GFmax(1000,s_GFmax),s_GFequil,D_crit)),np.linspace(0,6*GFequil(0.4,GFmax(1000,s_GFmax),s_GFequil,D_crit),100),'k:')
plt.plot(G_range,0.001*1000*np.maximum(6*GFequil(0.4,GFmax(1000,s_GFmax),s_GFequil,D_crit)-G_range,0),label='GF total secretion rate $b_G(\cdot,G)S$')
plt.plot(np.linspace(0,6*GFequil(0.4,GFmax(1000,s_GFmax),s_GFequil,D_crit)/5,100),5*np.linspace(0,6*GFequil(0.4,GFmax(1000,s_GFmax),s_GFequil,D_crit)/5,100),label='GF total decay rate $d_G\cdot G$')
plt.xticks([GFequil(0.4,GFmax(1000,s_GFmax),s_GFequil,D_crit),6*GFequil(0.4,GFmax(1000,s_GFmax),s_GFequil,D_crit)],['$\overline{G}$','$G^*$'],size=16)
plt.xlabel('growth factor concentration $G$',size=16)
plt.yticks([],[])
plt.legend(fontsize=13)
plt.show()