import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import colors

#################### Define Parameters, Functions, ODEs ####################

class parameters:
  def __init__(self,
               GFmax_slope = 0.35,          # slope of maximal growth factor concentration
               GFequil_Dcrit = 0.7,         # critical drug dose for growth factor secretion
               GFequil_slope = 5,           # slope of growth factor concentration
               C50_max = 10,                # maximal C50 drug dose
               C50_Gcrit = 300,             # critical growth factor concentration for C50 shift
               C50_slope = 0.03,            # slop of C50 shift
               NetGrowth_Rmax = 0.015,      # maximal cancer net growth rate
               NetGrowth_Rmin = -0.005,     # minimal cancer net growth rate
               NetGrowth_slope = 2,         # slope of cancer net growth rate
               Ddecay = 0.03,               # drug decay rate (>0) 0.006
               Gdecay = 5,                 # growth factor decay rate (>0)
               Gsecrete = 0.0001):          # growth factor secretion rate
    self.GFmax_slope = GFmax_slope
    self.GFequil_Dcrit = GFequil_Dcrit
    self.GFequil_slope = GFequil_slope
    self.C50_max = C50_max
    self.C50_Gcrit = C50_Gcrit
    self.C50_slope = C50_slope
    self.NetGrowth_Rmax = NetGrowth_Rmax
    self.NetGrowth_Rmin = NetGrowth_Rmin
    self.NetGrowth_slope = NetGrowth_slope
    self.Ddecay = Ddecay
    self.Gdecay = Gdecay
    self.Gsecrete = Gsecrete

# Functions (parameter class needs to be argument of the functions)

def GFmax(n,t,para):              # maximal growth factor concentration
  s = para.GFmax_slope
  return s * n[1]

def GFequil(n,t,para):            # equilibrium growth factor concentration
  D_crit = para.GFequil_Dcrit
  s = para.GFequil_slope
  return GFmax(n,t,para) / (1 + np.exp(- s * (n[2] - D_crit)))

def C50(n,t,para):                # C50 drug dose
  C50_max = para.C50_max
  G_crit = para.C50_Gcrit
  s = para.C50_slope
  return C50_max / (1 + np.exp(- s * (n[3] - G_crit)))

def NetGrowth(n,t,para):          # cancer net growth rate
  R_max = para.NetGrowth_Rmax
  R_min = para.NetGrowth_Rmin
  s = para.NetGrowth_slope
  if n[2]>0:
    return R_min + (R_max - R_min) / (1 + np.power((n[2] / C50(n,t,para)),s))
  else:
    return R_max

# Differential equations; n = [C,M,D,G] (parameter class needs to be argument of the ODE)

def ODE(n,t,para):
  d_D = para.Ddecay
  d_G = para.Gdecay
  b_G = para.Gsecrete
  Gstar = (d_G + b_G * n[1]) / (b_G * n[1]) * GFequil(n,t,para)

  dCdt = n[0] * NetGrowth(n,t,para)
  dMdt = n[1] * 0
  dDdt = - n[2] * d_D
  dGdt = b_G * np.max(Gstar - n[3],0) * n[1] - d_G * n[3]
  #dGdt = b_G * (GFequil(n,t,para) - n[3]) * n[1] - d_G * n[3]
  return [dCdt,dMdt,dDdt,dGdt]

# Function that advances the ODE solution by a time of 'duration'

def advance_onestep(population,time,duration,para):
  time_new = np.linspace(time[-1],time[-1]+duration,np.ceil(duration*100).astype(int))    #time interval for solving
  population_new = odeint(ODE,population[-1],time_new,args=(para,))                       #solve ODE
  time = np.concatenate((time,time_new),axis=0)                                           #concatenate time intervals
  population = np.concatenate((population,population_new),axis=0)                         #concatenate solutions
  return population, time;

# Function that computes AUC for administration of a unit dose every n days for T days total and decay rate r (per h)

def AUC(T,n,r):
  A = 0.
  for i in range(int(np.floor(T/n))):
    A += (1-np.exp(-(r/24)*(T-i*n)))/r
  return A

# Function that computes administered dose for a given average concentration c, decay rate r (per h), days between doses n, time horizon T

def admdose(c,r,n,T):
  dose = c * T / AUC(T,n,r)
  return dose

#################### Plot agents over time ####################

days_max = 90
interval_days = 10
av_days = 30
adm_dose = 1.4

para = parameters()

n_0 = np.array([100000000.,1000.,0.,0.])                                      #initial state
n = np.array([n_0])
time = np.array([0])

cycles = int(np.floor(days_max/interval_days))
av_conc = adm_dose * AUC(days_max,interval_days,para.Ddecay) / days_max
print(av_conc)

for i in range(0,cycles):
    n[-1] += np.array([0,0,adm_dose,0])
    n,time = advance_onestep(n,time,24*interval_days,para)

n[-1] += np.array([0,0,adm_dose,0])
n,time = advance_onestep(n,time,24*days_max-time[-1],para)

plt.plot(time/24,n[:,0]/10000000,label='cancer cell number $C$ [10^7]',color='xkcd:soft blue')
plt.plot(time/24,n[:,2],label='drug conc. $D$ [μg/mL]',color='xkcd:coral')
plt.plot(time/24,n[:,3]/100,label='growth factor conc. $G$ [100 pg/mL]',color='xkcd:dull green')
plt.xlabel('time [d]',size=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=13,loc='upper right')
plt.show()

average = np.mean(n[int(24*100*(days_max-av_days)):-1][:,0])
print('average cancer cell number during last',av_days,'days = ',average)
average2 = np.mean(n[int(124*00*(days_max-interval_days)):-1][:,0])
print('average cancer cell number during last cycle = ',average2)

#################### Plot heat map ####################

def average_size(days_max,interval_days,av_days,av_conc):
  n_0 = np.array([100000000.,1000.,0.,0.])                                      #initial state
  n = np.array([n_0])
  time = np.array([0])

  cycles = int(np.floor(days_max/interval_days))
  dose = admdose(av_conc,para.Ddecay,interval_days,days_max)

  for i in range(0,cycles):
    n[-1] += np.array([0,0,dose,0])
    n,time = advance_onestep(n,time,24*interval_days,para)

  n[-1] += np.array([0,0,dose,0])
  n,time = advance_onestep(n,time,24*days_max-time[-1],para)

  average = np.mean(n[int(24*100*(days_max-av_days)):-1][:,0])
  return np.log10(average)

grid_size = 40
# generate 2d grids for the x & y bounds (x=log10(dose/day),y=interval days)
x, y = np.meshgrid(np.linspace(np.log10(0.05),np.log10(150), grid_size), np.linspace(1, 21, grid_size))

z = x+y
for i in range(grid_size):
  print(i)
  for j in range(grid_size):
    z[i][j]=average_size(180,y[i][0],30,pow(10,x[0][j]))

#z_min, z_max = z.min(), z.max()
divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=8., vmax=14)

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='seismic', norm=divnorm)
ax.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(c, ax=ax,ticks=[0,2,4,6,8,10,12,14])
cbar.set_label(label='avg. # of cancer cells $C$ ($\log_{10}$)',size=13)
cbar.ax.set_yticklabels(['<0','2','4','6','8','10','12','>14'],size=13)
plt.xlabel('avg. drug concentration $D$ [μg/mL]',size=16)
plt.xticks(ticks=[-1,0,1,2],labels=['0.1','1','10','100'],size=16)
plt.ylabel('time between doses $\\tau$ [d]',size=16)
plt.yticks([1,7,14,21],size=16)
plt.show()

#################### Plot horizontal cuts ####################

days_max = 180

color_list = ['C0','xkcd:sky blue','xkcd:teal']#,'xkcd:dark teal']
interval_days = [1,7,14]

plt.plot(np.linspace(np.log10(0.05),np.log10(150),100),np.full(100,8),'k--')
for j in range(3):
  for av_conc in np.linspace(np.log10(0.05),np.log10(500),40):
    cycles = int(np.floor(days_max/interval_days[j]))
    dose = admdose(pow(10,av_conc),para.Ddecay,interval_days[j],days_max)

    n_0 = np.array([100000000.,1000.,0,0])                                      #initial state
    n = np.array([n_0])
    time = np.array([0])

    for i in range(0,cycles):
      n[-1] += np.array([0,0,dose,0])
      n,time = advance_onestep(n,time,24*interval_days[j],para)

    n[-1] += np.array([0,0,dose,0])
    n,time = advance_onestep(n,time,24*days_max-time[-1],para)

    average = np.mean(n[int(24*100*(days_max-30)):-1][:,0])
    plt.scatter(av_conc,np.log10(average),color=color_list[j])
    #plt.scatter(dose_per_day,np.log10(n[-1][0]),color='C0')
  plt.scatter(av_conc,np.log10(average),color=color_list[j],label=interval_days[j])
  print('cycle length =',interval_days[j],'days')

plt.xlabel('avg. drug concentration $D$ [μg/mL]',size=16)
plt.xticks(ticks=[-1,0,1,2],labels=['0.1','1','10','100'],size=16)
plt.ylabel('avg. # of cancer cells $C$ ($\log_{10}$)',size=16)
plt.yticks(size=16)
plt.legend(title='time betw. doses $\\tau$ [d]',fontsize=13,title_fontsize=13,loc='upper right')
plt.show()

#################### Plot vertical cuts ####################

days_max = 180

color_list = ['C0','xkcd:sky blue','xkcd:teal','xkcd:dark teal']
av_conc = [0.1,3,50]

plt.plot(np.linspace(1,21,100),np.full(100,8),'k--')
for j in range(3):
  for interval_days in np.linspace(1,21,40):
    cycles = int(np.floor(days_max/interval_days))
    dose = admdose(av_conc[j],para.Ddecay,interval_days,days_max)

    n_0 = np.array([100000000.,1000.,0,0])                                      #initial state
    n = np.array([n_0])
    time = np.array([0])

    for i in range(0,cycles):
      n[-1] += np.array([0,0,dose,0])
      n,time = advance_onestep(n,time,24*interval_days,para)

    n[-1] += np.array([0,0,dose,0])
    n,time = advance_onestep(n,time,24*days_max-time[-1],para)

    average = np.mean(n[int(24*100*(days_max-30)):-1][:,0])
    plt.scatter(interval_days,np.log10(average),color=color_list[j])
  plt.scatter(interval_days,np.log10(average),color=color_list[j],label=av_conc[j])
  print('average concentration =',av_conc[j])
plt.xlabel('time between doses $\\tau$ [d]',size=16)
plt.xticks(ticks=[1,7,14,21],size=16)
plt.ylabel('avg. # of cancer cell ($\log_{10}$)',size=16)
plt.yticks(size=16)
plt.legend(title='avg. drug conc. $D$ [μg/mL]',fontsize=13,title_fontsize=13,loc='upper right')
plt.show()

#################### Plot log drug concentration ####################

days_max = 50
# D_crit(0)
D1 = np.log10(para.C50_max / (1 + np.exp(- para.C50_slope * (0 - para.C50_Gcrit)))*np.sqrt(-para.NetGrowth_Rmax/para.NetGrowth_Rmin))#np.log10(0.00292)
# D^
D2 = np.log10(para.GFequil_Dcrit) #np.log10(0.75)
# D_crit(G^max)
D3 = np.log10(para.C50_max / (1 + np.exp(- para.C50_slope * (para.GFmax_slope * 1000 - para.C50_Gcrit)))*np.sqrt(-para.NetGrowth_Rmax/para.NetGrowth_Rmin))#np.log10(14.16)

color_list = ['C0','xkcd:sky blue','xkcd:teal','xkcd:dark teal']
interval_days = [1,7,14,21]

av_conc = 0.1
#plt.plot(np.linspace(0,50,100),np.full(100,-1),'k--')
for j in range(4):
  cycles = int(np.floor(days_max/interval_days[j]))
  dose = dose = admdose(av_conc,para.Ddecay,interval_days[j],days_max)

  n_0 = np.array([100000000.,1000.,0,0])                                      #initial state
  n = np.array([n_0])
  time = np.array([0])

  for i in range(0,cycles):
    n[-1] += np.array([0,0,dose,0])
    n,time = advance_onestep(n,time,24*interval_days[j],para)

  n[-1] += np.array([0,0,dose,0])
  n,time = advance_onestep(n,time,24*days_max-time[-1],para)

  plt.plot(time/24,np.log10(n[:,2]),color=color_list[j],label=interval_days[j])
plt.xlabel('time [d]',size=16)
plt.xticks(size=16)
plt.yticks([D1,-1,D2,D3],['$D_{crit}(0)$',0.1,'$\hat{D}$','$D_{crit}(G^{\max}(S))$'],size=16)
plt.ylabel('D [μg/mL]',size=16)
plt.axhspan(-7, D1, color='red', alpha=0.2, lw=0)
plt.axhspan(D1, D2, color='blue', alpha=0.2, lw=0)
plt.axhspan(D2, D3, color='red', alpha=0.2, lw=0)
plt.axhspan(D3, 3.5, color='blue', alpha=0.2, lw=0)
plt.legend(title='time betw. doses $\\tau$ [d]',fontsize=13,title_fontsize=13,loc='lower right')
plt.title('average drug concentration 0.1 μg/mL',size=16)
plt.show()

daily_dose = 3
for j in range(4):
  cycles = int(np.floor(days_max/interval_days[j]))
  dose = daily_dose * AUC(days_max,1,para.Ddecay) / AUC(days_max,interval_days[j],para.Ddecay)

  n_0 = np.array([100000000.,1000.,0,0])                                      #initial state
  n = np.array([n_0])
  time = np.array([0])

  for i in range(0,cycles):
    n[-1] += np.array([0,0,dose,0])
    n,time = advance_onestep(n,time,24*interval_days[j],para)

  n[-1] += np.array([0,0,dose,0])
  n,time = advance_onestep(n,time,24*days_max-time[-1],para)

  plt.plot(time/24,np.log10(n[:,2]),color=color_list[j],label=interval_days[j])
plt.xlabel('time [d]',size=16)
plt.xticks(size=16)
plt.yticks([D1,D2,np.log10(3),D3],['$D_{crit}(0)$','$\hat{D}$',3,'$D_{crit}(G^{\max}(S))$'],size=16)
plt.ylabel('D [μg/mL]',size=16)
plt.axhspan(-7, D1, color='red', alpha=0.2, lw=0)
plt.axhspan(D1, D2, color='blue', alpha=0.2, lw=0)
plt.axhspan(D2, D3, color='red', alpha=0.2, lw=0)
plt.axhspan(D3, 3.5, color='blue', alpha=0.2, lw=0)
plt.legend(title='time betw. doses $\\tau$ [d]',fontsize=13,title_fontsize=13,loc='lower right')
plt.title('average drug concentration 3 μg/mL',size=16)
plt.show()

daily_dose = 50
for j in range(4):
  cycles = int(np.floor(days_max/interval_days[j]))
  dose = daily_dose * AUC(days_max,1,para.Ddecay) / AUC(days_max,interval_days[j],para.Ddecay)

  n_0 = np.array([100000000.,1000.,0,0])                                      #initial state
  n = np.array([n_0])
  time = np.array([0])

  for i in range(0,cycles):
    n[-1] += np.array([0,0,dose,0])
    n,time = advance_onestep(n,time,24*interval_days[j],para)

  n[-1] += np.array([0,0,dose,0])
  n,time = advance_onestep(n,time,24*days_max-time[-1],para)

  plt.plot(time/24,np.log10(n[:,2]),color=color_list[j],label=interval_days[j])
plt.xlabel('time [d]',size=16)
plt.xticks(size=16)
plt.yticks([D1,D2,D3,np.log10(50)],['$D_{crit}(0)$','$\hat{D}$','$D_{crit}(G^{\max}(S))$',50],size=16)
plt.ylabel('D [μg/mL]',size=16)
plt.axhspan(-7, D1, color='red', alpha=0.2, lw=0)
plt.axhspan(D1, D2, color='blue', alpha=0.2, lw=0)
plt.axhspan(D2, D3, color='red', alpha=0.2, lw=0)
plt.axhspan(D3, 3.5, color='blue', alpha=0.2, lw=0)
plt.legend(title='time betw. doses $\\tau$ [d]',fontsize=13,title_fontsize=13,loc='lower right')
plt.title('average drug concentration 50 μg/mL',size=16)
plt.show()