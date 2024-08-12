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

#################### Plot agents over time - without blocker ####################

days_max = 90
interval_days = 7
av_days = 30
adm_dose = 0.85
adm_dose_aEGF = 0
delay_aEGF = 0

para = parameters()

n_0 = np.array([100000000.,1000.,0.,0.,0.])                                      #initial state
n = np.array([n_0])
time = np.array([0])

cycles = int(np.floor(days_max/interval_days))
av_conc = adm_dose * AUC(days_max,interval_days,para.Ddecay) / days_max
print(av_conc)
av_conc_aEGF = adm_dose * AUC(days_max,interval_days,para.Bdecay) / days_max
print(av_conc_aEGF)

for i in range(0,cycles):
    n[-1] += np.array([0,0,adm_dose,0,0])
    n,time = advance_onestep(n,time,24*delay_aEGF,para)
    n[-1] += np.array([0,0,0,0,adm_dose_aEGF])
    n,time = advance_onestep(n,time,24*(interval_days-delay_aEGF),para)

n[-1] += np.array([0,0,adm_dose,0,0])
n,time = advance_onestep(n,time,min([24*days_max-time[-1],24*delay_aEGF]),para)
if time[-1]<24*days_max:
    n[-1] += np.array([0,0,0,0,adm_dose_aEGF])
n,time = advance_onestep(n,time,24*days_max-time[-1],para)

plt.plot(time/24,n[:,0]/10000000,label='CRC cell number $C$ [10^7]',color='xkcd:soft blue')
plt.plot(time/24,n[:,2],label='CTX concentration $D$ [μg/mL]',color='xkcd:coral')
plt.plot(time/24,n[:,3]/10,label='EGF concentration $G$ [10 pg/mL]',color='xkcd:dull green')
plt.plot(time/24,n[:,4],label='aEGF  concentration $B$ [μg/mL]',color='xkcd:pale purple')
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


#################### Plot agents over time - with blocker ####################

days_max = 90
interval_days = 7
av_days = 30
adm_dose = 0.85
adm_dose_aEGF = 0.45
delay_aEGF = 0

para = parameters()

n_0 = np.array([100000000.,1000.,0.,0.,0.])                                      #initial state
n = np.array([n_0])
time = np.array([0])

cycles = int(np.floor(days_max/interval_days))
av_conc = adm_dose * AUC(days_max,interval_days,para.Ddecay) / days_max
print(av_conc)
av_conc_aEGF = adm_dose * AUC(days_max,interval_days,para.Bdecay) / days_max
print(av_conc_aEGF)

for i in range(0,cycles):
    n[-1] += np.array([0,0,adm_dose,0,0])
    n,time = advance_onestep(n,time,24*delay_aEGF,para)
    n[-1] += np.array([0,0,0,0,adm_dose_aEGF])
    n,time = advance_onestep(n,time,24*(interval_days-delay_aEGF),para)

n[-1] += np.array([0,0,adm_dose,0,0])
n,time = advance_onestep(n,time,min([24*days_max-time[-1],24*delay_aEGF]),para)
if time[-1]<24*days_max:
    n[-1] += np.array([0,0,0,0,adm_dose_aEGF])
n,time = advance_onestep(n,time,24*days_max-time[-1],para)

plt.plot(time/24,n[:,0]/10000000,label='CRC cell number $C$ [10^7]',color='xkcd:soft blue')
plt.plot(time/24,n[:,2],label='CTX concentration $D$ [μg/mL]',color='xkcd:coral')
plt.plot(time/24,n[:,3]/10,label='EGF concentration $G$ [10 pg/mL]',color='xkcd:dull green')
plt.plot(time/24,n[:,4],label='aEGF  concentration $B$ [μg/mL]',color='xkcd:pale purple')
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

#################### Plot varying aEGF delay ####################

para = parameters()
days_max = 180
interval_days = 7
adm_dose = 0.85
av_days = 30

colors = ['C0','xkcd:sky blue','xkcd:teal','xkcd:dark teal','C1']
av_conc_aEGF = [0.01,0.2,0.5,0.8]

plt.plot(np.linspace(0,7,100),np.full(100,8),'k--')
for j in range(4):
  av = np.array([])
  for delay_aEGF in np.linspace(0,7,15):
    cycles = int(np.floor(days_max/interval_days))
    dose_aEGF = admdose(av_conc_aEGF[j],para.Bdecay,7,180)

    n_0 = np.array([100000000.,1000.,0,0,0])                                      #initial state
    n = np.array([n_0])
    time = np.array([0])

    for i in range(0,cycles):
      n[-1] += np.array([0,0,adm_dose,0,0])
      n,time = advance_onestep(n,time,24*delay_aEGF,para)
      n[-1] += np.array([0,0,0,0,dose_aEGF])
      n,time = advance_onestep(n,time,24*(7-delay_aEGF),para)

    n[-1] += np.array([0,0,adm_dose,0,0])
    n,time = advance_onestep(n,time,min([24*days_max-time[-1],24*delay_aEGF]),para)
    if time[-1]<24*days_max:
      n[-1] += np.array([0,0,0,0,dose_aEGF])
    n,time = advance_onestep(n,time,24*days_max-time[-1],para)

    average = np.mean(n[int(24*100*(days_max-av_days)):-1][:,0])
    plt.scatter(delay_aEGF,np.log10(average),color=colors[j])
    av = np.concatenate((av,[average]),axis=0)
  plt.scatter(delay_aEGF,np.log10(average),color=colors[j],label=av_conc_aEGF[j])
  plt.scatter(np.linspace(0,7,15)[np.argmin(av)],np.log10(min(av)),color='r')
  print('av conc. aEGF =',av_conc_aEGF[j])
plt.xlabel('aEGF adm. delay [d]',size=16)
plt.xticks(ticks=[0,1,2,3,4,5,6,7],size=16)
plt.yticks([5,6,7,8,9,10],size=16)
plt.ylim(5,10.2)
plt.ylabel('avg. # of CRC cells $C$ ($\log_{10}$)',size=16)
plt.legend(title='avg. aEGF conc. $B$ [μg/mL]',fontsize=13,title_fontsize=13,loc='lower right')
plt.show()