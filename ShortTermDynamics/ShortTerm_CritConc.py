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

#################### Plot Windows of (un)successful drug concentrations ####################

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

D_range = np.linspace(0.001,30,1000)
G_range = np.linspace(0,500,1000)
n_M=1000

plt.plot(np.log10(D_range),GFequil(D_range,GFmax(n_M,s_GFmax),s_GFequil,D_crit),label='$\overline{G}$ equil. GF conc. (dep. on $D$)')
plt.plot(np.log10(C50(G_range,C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)),G_range,'C0--',label='$D_{crit}$ drug conc. (dep. on $G$)')
plt.xticks([np.log10(0.35),np.log10(C50(GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min))],['$D^1_a$','$D_{crit}(\overline{G}(S,D^1_a))$'],size=16)
plt.xlabel('drug concentration $D$',size=16)
plt.yticks([GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit)],['$\overline{G}(S,D^1_a)$'],size=16)
plt.ylabel('growth factor conc. $G$',size=16)
plt.arrow(np.log10(0.35),-0.3,0,GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit)+0.3,length_includes_head=True,width = 0.02,head_width=0.1,head_length=20,color='k')
plt.arrow(np.log10(0.35),GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit),np.log10(C50(GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min))-np.log10(0.35),0,length_includes_head=True,width = 2.5,head_width=0,head_length=0,color='k')
plt.arrow(np.log10(C50(GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)),GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit),0,-GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit)-0.3,length_includes_head=True,width = 0.02,head_width=0.1,head_length=20,color='k')
plt.arrow(np.log10(0.35),-0.3,np.log10(C50(GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min))-np.log10(0.35),0,length_includes_head=True,width = 2.5,head_width=15,head_length=0.15,color='C1')
plt.text(x=np.log10(0.35)+0.05, y=16, s="1.",size=16)
plt.text(x=-1.3, y=GFequil(0.35,GFmax(n_M,s_GFmax),s_GFequil,D_crit)+5, s="2.",size=16)
plt.text(x=-1.3, y=-32, s="3.",size=16)
plt.legend(fontsize=13)
plt.show()

plt.plot(np.log10(D_range),GFequil(D_range,GFmax(n_M,s_GFmax),s_GFequil,D_crit),label='$\overline{G}$ equil. GF conc. (dep. on $D$)')
plt.plot(np.log10(C50(G_range,C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)),G_range,'C0--',label='$D_{crit}$ drug conc. (dep. on $G$)')
plt.xticks([np.log10(1.4),np.log10(C50(GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min))],['$D^2_a$','    $D_{crit}(\overline{G}(S,D^2_a))$'],size=16)
plt.xlabel('drug concentration $D$',size=16)
plt.yticks([GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit)],['$\overline{G}(S,D^2_a)$'],size=16)
plt.ylabel('growth factor conc. $G$',size=16)
plt.arrow(np.log10(1.4),-0.3,0,GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit)+0.3,length_includes_head=True,width = 0.02,head_width=0.1,head_length=20,color='k')
plt.arrow(np.log10(1.4),GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit),np.log10(C50(GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min))-np.log10(1.4),0,length_includes_head=True,width = 0,head_width=15,head_length=0,color='k')
plt.arrow(np.log10(C50(GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min)),GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit),0,-GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit)-0.3,length_includes_head=True,width = 0.02,head_width=0.1,head_length=20,color='k')
plt.arrow(np.log10(1.4),-0.3,np.log10(C50(GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min))-np.log10(1.4),0,length_includes_head=True,width = 2.5,head_width=15,head_length=0.15,color='C1')
plt.text(x=np.log10(1.4)+0.05, y=150, s="1.",size=16)
plt.text(x=0.55, y=GFequil(1.4,GFmax(n_M,s_GFmax),s_GFequil,D_crit)-35, s="2.",size=16)
plt.text(x=0.55, y=7, s="3.",size=16)
plt.legend(fontsize=13)
plt.show()


def D_zero(n_D):
    return C50(GFequil(n_D,GFmax(n_M,s_GFmax),s_GFequil,D_crit),C50_max,s_C50,GF_crit) * np.sqrt(- R_max/R_min) #sqrt bc s_NetGrowth=2

plt.axvspan(-3, np.log10(D_zero(0)), color='red', alpha=0.2, lw=0)
plt.axvspan(np.log10(D_zero(0)), np.log10(D_crit), color='blue', alpha=0.2, lw=0, label='succesful')
plt.axvspan(np.log10(D_crit), np.log10(D_zero(GFmax(n_M,s_GFmax))), color='red', alpha=0.2, lw=0, label='unsuccesful')
plt.axvspan(np.log10(D_zero(GFmax(n_M,s_GFmax))), np.log10(30), color='blue', alpha=0.2, lw=0)
plt.plot(np.log10(D_range),np.log10(D_range),'k:')
plt.plot(np.log10(D_range),np.log10(D_zero(D_range)))
plt.arrow(np.log10(0.35),-3.3,0,np.log10(D_zero(0.35))+3.3,length_includes_head=True,width = 0.02,head_width=0.1,head_length=0.4,color='C1')
plt.arrow(np.log10(1.4),-3.3,0,np.log10(D_zero(1.4))+3.3,length_includes_head=True,width = 0.02,head_width=0.1,head_length=0.4,color='C1')
plt.xticks([np.log10(D_zero(0)),np.log10(0.35),np.log10(D_crit),np.log10(1.4),np.log10(D_zero(GFmax(n_M,s_GFmax)))],['$D_{crit}(0)$','$D^1_a$','$\hat{D}$','$D^2_a$','$D_{crit}(G^{\max}(S))$'],size=14)
plt.xlabel('admin. drug conc. $D$',size=16)
plt.yticks([],size=16)
plt.ylabel('crit. drug conc. $D_{crit}(\overline{G}(S,D))$',size=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.38,1),fontsize=13)
plt.show()