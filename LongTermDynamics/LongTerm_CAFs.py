import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import colors

#################### Define Parameters, Functions, ODEs ####################

class parameters:
  def __init__(self,
               GFequil_max = 27.1,             # maximal growth factor concentration
               GFequil_Dcrit = 0.0049,         # critical drug dose for growth factor secretion
               GFequil_slope = 65.7,           # slope of growth factor concentration
               C50_max = 7.34,                 # maximal C50 drug dose
               C50_Gcrit = 379.4,              # critical growth factor concentration for C50 shift
               C50_slope = 0.0078,             # slop of C50 shift
               NetGrowth_Rmax = 0.0147,        # maximal cancer net growth rate
               NetGrowth_Rmin = -0.0079,       # minimal cancer net growth rate
               NetGrowth_slope = 0.93,         # slope of cancer net growth rate
               Ddecay = 0.006,                 # drug decay rate (>0)
               Gdecay = 5,                     # growth factor decay rate (>0)
               Gsecrete = 0.001,               # growth factor secretion rate
               Bdecay = 0.004,                 # blocker decay rate (>0)
               Bblock = 1):                    # blocker effect rate
    self.GFequil_max = GFequil_max
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
    self.Bdecay = Bdecay
    self.Bblock = Bblock

# Functions (parameter class needs to be argument of the functions)

def GFequil(n,t,para):            # equilibrium growth factor concentration
  GF_max = para.GFequil_max
  D_crit = para.GFequil_Dcrit
  s = para.GFequil_slope
  return GF_max / (1 + np.exp(- s * (n[2] - D_crit)))

def C50(n,t,para):                # C50 drug dose
  C50_max = para.C50_max
  G_crit = para.C50_Gcrit
  s = para.C50_slope
  Bblock = para.Bblock
  return C50_max / (1 + np.exp(- s * (n[3] * np.max(1 - n[4] * Bblock,0) - G_crit)))

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
  d_B = para.Bdecay
  Gstar = (d_G + b_G * n[1]) / (b_G * n[1]) * GFequil(n,t,para)

  dCdt = n[0] * NetGrowth(n,t,para)
  dMdt = n[1] * 0
  dDdt = - n[2] * d_D
  dGdt = b_G * np.max(Gstar - n[3],0) * n[1] - d_G * n[3]
  dBdt = - n[4] * d_B
  return [dCdt,dMdt,dDdt,dGdt,dBdt]

# Function that advances the ODE solution by a time of 'duration'

def advance_onestep(population,time,duration,para):
  time_new = np.linspace(time[-1],time[-1]+duration,np.ceil(duration*100).astype(int))    #time interval for solving
  population_new = odeint(ODE,population[-1],time_new,args=(para,))                       #solve ODE
  time = np.concatenate((time,time_new),axis=0)                                           #concatenate time intervals
  population = np.concatenate((population,population_new),axis=0)                         #concatenate solutions
  return population, time;

# Function that computes AUC for administration of a unit dose every n days for T days total and decay rate r

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
interval_days = 7
av_days = 30
adm_dose = 1

para = parameters()

n_0 = np.array([100000000.,1000.,0.,0.,0.])                                      #initial state
n = np.array([n_0])
time = np.array([0])

cycles = int(np.floor(days_max/interval_days))
av_conc = adm_dose * AUC(days_max,interval_days,para.Ddecay) / days_max
print(av_conc)

for i in range(0,cycles):
    n[-1] += np.array([0,0,adm_dose,0,0])
    n,time = advance_onestep(n,time,24*interval_days,para)

n[-1] += np.array([0,0,adm_dose,0,0])
n,time = advance_onestep(n,time,24*days_max-time[-1],para)

plt.plot(time/24,n[:,0]/10000000,label='CRC cell number $C$ [10^7]',color='xkcd:soft blue')
plt.plot(time/24,n[:,2],label='CTX concentration $D$ [μg/mL]',color='xkcd:coral')
plt.plot(time/24,n[:,3]/10,label='EGF concentration $G$ [10 pg/mL]',color='xkcd:dull green')
plt.xlabel('time [d]',size=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylim(-1,41)
plt.legend(loc='upper left',fontsize=13)
plt.show()

average = np.mean(n[int(24*100*(days_max-av_days)):-1][:,0])
print('average cancer cell number during last',av_days,'days = ',average)
average2 = np.mean(n[int(24*100*(days_max-interval_days)):-1][:,0])
print('average cancer cell number during last cycle = ',average2)

#################### Plot heat map - end size ####################

def average_size(days_max,interval_days,av_conc):
  n_0 = np.array([100000000.,1000.,0.,0.,0.])                                      #initial state
  n = np.array([n_0])
  time = np.array([0])

  cycles = int(np.floor(days_max/interval_days))
  dose = admdose(av_conc,para.Ddecay,interval_days,days_max)

  for i in range(0,cycles):
    n[-1] += np.array([0,0,dose,0,0])
    n,time = advance_onestep(n,time,24*interval_days,para)

  n[-1] += np.array([0,0,dose,0,0])
  n,time = advance_onestep(n,time,24*days_max-time[-1],para)

  average = np.mean(n[int(24*100*(days_max-30)):-1][:,0])
  return np.log10(average)

# generate 2d grids for the x & y bounds (x=log10(dose/day),y=interval days)
grid_size = 40

x, y = np.meshgrid(np.linspace(np.log10(0.1), np.log10(5.5), grid_size), np.linspace(1, 28, grid_size))

z = x+y
for i in range(grid_size):
  print(i)
  for j in range(grid_size):
    z[i][j]=average_size(180,y[i][0],pow(10,x[0][j]))

divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=8., vmax=14)

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='seismic', norm=divnorm)
ax.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(c, ax=ax,ticks=[0,2,4,6,8,10,12,14])
cbar.set_label(label='avg. # of cancer cells $C$ ($\log_{10}$)',size=13)
cbar.ax.set_yticklabels(['<0','2','4','6','8','10','12','>14'],size=13)
plt.xlabel('average CTX concentration $D$ [μg/mL]',size=16)
plt.xticks(ticks=[-1,np.log10(0.5),0,np.log10(5)],labels=['0.1','0.5','1','5'],size=16)
plt.yticks(ticks=[1,7,14,21,28],size=16)
plt.ylabel('time between doses $\\tau$ [d]',size=16)

plt.show()"""

#################### Plot heat map - max size ####################

"""def peak_size(days_max,interval_days,av_conc):
  n_0 = np.array([100000000.,1000.,0.,0.,0.])                                      #initial state
  n = np.array([n_0])
  time = np.array([0])

  cycles = int(np.floor(days_max/interval_days))
  dose = admdose(av_conc,para.Ddecay,interval_days,days_max)

  for i in range(0,cycles):
    n[-1] += np.array([0,0,dose,0,0])
    n,time = advance_onestep(n,time,24*interval_days,para)

  n[-1] += np.array([0,0,dose,0,0])
  n,time = advance_onestep(n,time,24*days_max-time[-1],para)

  peak = max(n[:,0])
  return np.log10(peak)

# generate 2d grids for the x & y bounds (x=log10(dose/day),y=interval days)
grid_size = 40

x, y = np.meshgrid(np.linspace(np.log10(0.1), np.log10(5.5), grid_size), np.linspace(1, 28, grid_size))

z = x+y
for i in range(grid_size):
  print(i)
  for j in range(grid_size):
    z[i][j]=peak_size(180,y[i][0],pow(10,x[0][j]))

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='Reds', vmin=8., vmax=9.)
ax.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(c, ax=ax,ticks=[8,8.5,9])
cbar.set_label(label='max. # of CRC cells $C$ ($\log_{10}$)',size=13)
cbar.ax.set_yticklabels(['8','8.5','>9'],size=13)
plt.xlabel('average CTX concentration $D$ [μg/mL]',size=16)
plt.xticks(ticks=[-1,np.log10(0.5),0,np.log10(5)],labels=['0.1','0.5','1','5'],size=16)
plt.yticks(ticks=[1,7,14,21,28],size=16)
plt.ylabel('time between doses $\\tau$ [d]',size=16)

plt.show()

#################### Plot vertical cuts ####################

days_max = 180

color_list = ['C0','xkcd:sky blue','xkcd:teal','xkcd:dark teal']
av_conc = [0.4,0.6,1.1]

plt.plot(np.linspace(1,21,100),np.full(100,8),'k--')
for j in range(3):
  av = np.array([])
  for interval_days in np.linspace(1,21,41):
    cycles = int(np.floor(days_max/interval_days))
    dose = admdose(av_conc[j],para.Ddecay,interval_days,days_max)

    n_0 = np.array([100000000.,1000.,0,0,0])                                      #initial state
    n = np.array([n_0])
    time = np.array([0])

    for i in range(0,cycles):
      n[-1] += np.array([0,0,dose,0,0])
      n,time = advance_onestep(n,time,24*interval_days,para)

    n[-1] += np.array([0,0,dose,0,0])
    n,time = advance_onestep(n,time,24*days_max-time[-1],para)

    average = np.mean(n[int(24*100*(days_max-30)):-1][:,0])
    plt.scatter(interval_days,np.log10(average),color=color_list[j])
    av = np.concatenate((av,[average]),axis=0)
  plt.scatter(interval_days,np.log10(average),color=color_list[j],label=av_conc[j])
  plt.scatter(np.linspace(1,21,41)[np.argmin(av)],np.log10(min(av)),color='r')
  print('average concentration =',av_conc[j])
plt.xlabel('time between doses $\\tau$ [d]',size=16)
plt.xticks(ticks=[1,7,14,21],size=16)
plt.ylabel('avg. # of CRC cells $C$ ($\log_{10}$)',size=16)
plt.yticks(size=16)
plt.legend(title='avg. CTX conc. $D$ [μg/mL]',fontsize=13,title_fontsize=13,loc='upper right')
plt.show()