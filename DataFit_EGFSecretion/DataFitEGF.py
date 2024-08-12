import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#################### plot time series data ####################

time1 = np.genfromtxt('EGF_timeseries_20190525.dat',usecols=(0),skip_header=1)
conc1 = np.genfromtxt('EGF_timeseries_20190525.dat',usecols=(1),skip_header=1)
time2 = np.genfromtxt('EGF_timeseries_20190611.dat',usecols=(0),skip_header=1)
conc2 = np.genfromtxt('EGF_timeseries_20190611.dat',usecols=(1),skip_header=1)

plt.bar(time1-1.5,conc1,label='CAF12905',width=4)
plt.bar(time2+1.5,conc2,label='CAF13000',width=4)
plt.legend(loc='lower right',title='CAF cell line',fontsize=13,title_fontsize=13)   
plt.xlabel('time [h]',size=16)
plt.xticks(size=16)
plt.ylabel('EGF conc. $G$ [pg/mL]',size=16)
plt.yticks(size=16)
plt.show()

#################### fit parameters for EGF secretion ####################

data_CAF13000_20180808 = np.array([[0.,9.795],[0.05,17.841],[0.1,20.396],[1.,21.727],[5.,25.892]])
data_CAF12905_20180721 = np.array([[0.,11.686],[0.05,28.582],[0.1,27.042],[1.,30.202],[5.,36.667]])
data_CAF12905_20180901 = np.array([[0.,17.315],[0.02,26.884],[0.4,8.393],[1.,30.419],[5.,33.57]])
data_CAF12911_20180709 = np.array([[0.,6.004],[0.1,21.462],[1.,22.464],[5.,20.393],[10.,22.965]])

dataset = [data_CAF13000_20180808,data_CAF12905_20180721,data_CAF12905_20180901,data_CAF12911_20180709]

number_CAF13000_20180808 = np.array([34112,48009,38809,36513,31235])
number_CAF12905_20180721 = np.array([29250,30835,33330,33947,33670])
number_CAF12905_20180901 = np.array([10895,12056,13206,14713,12130])
number_CAF12911_20180709 = np.array([24913,23645,24954,21348,22913])

numbers = [np.mean(number_CAF13000_20180808),
           np.mean(number_CAF12905_20180721),
           np.mean(number_CAF12905_20180901),
           np.mean(number_CAF12911_20180709)]

def GFequil(D,GF_max,s,D_crit):
  return GF_max / (1 + np.exp(-s * (D - D_crit)))

def simultaneous_error(para):
  s = para[0]
  D_crit = para[1]
  E=0
  for i in range(4):
    for j in range(np.size(dataset[i],0)):
      fctvalue = GFequil(dataset[i][j][0],para[i+2],s,D_crit)
      E += np.power((dataset[i][j][1]-fctvalue),2)
  return E

def simultaneous_relative_error(para):
  s = para[0]
  D_crit = para[1]
  E=0
  for i in range(4):
    for j in range(np.size(dataset[i],0)):
      fctvalue = GFequil(dataset[i][j][0],para[i+2],s,D_crit)
      if fctvalue != 0:
        E += np.power((dataset[i][j][1]-fctvalue)/fctvalue,2)
      else:
        E += 10
  return E

para0 = [10,0.01,25,30,30,20]

result = opt.minimize(simultaneous_error,para0,bounds=((0,None),(0,None),(0,None),(0,None),(0,None),(0,None)))
params = result.x

result_rel = opt.minimize(simultaneous_relative_error,para0,bounds=((0,None),(0,None),(0,None),(0,None),(0,None),(0,None)))
params_rel = result_rel.x

print('Order: Initial guess, least square fit, relative least square fit')
print('Slope: ',para0[0],params[0],params_rel[0])
print('D_crit: ',para0[1],params[1],params_rel[1])
print('Max0: ',para0[2],params[2],params_rel[2])
print('Max1: ',para0[3],params[3],params_rel[3])
print('Max2: ',para0[4],params[4],params_rel[4])
print('Max3: ',para0[5],params[5],params_rel[5])
print('Least square error: ',simultaneous_error(para0),simultaneous_error(params),simultaneous_error(params_rel))
print('Relative least square error: ',simultaneous_relative_error(para0),simultaneous_relative_error(params),simultaneous_relative_error(params_rel))

D_range=np.linspace(0,10,100000)
C = ('C0','C1','C2','C3')

for i in range(4):
  dataset[i][0][0] = 0.00001

D_range_adj=np.linspace(0.00001,10,100000)

#plt.rcParams['figure.figsize']=[8,6]
for i in range(4):
  plt.scatter(np.log10(dataset[i][:,0]),dataset[i][:,1],c=C[i],label='_nolegend_')
  #plt.plot(np.log10(D_range_adj),GFequil(D_range,params[i+2],params[0],params[1]),c=C[i])
  plt.plot(np.log10(D_range_adj),GFequil(D_range,params_rel[i+2],params_rel[0],params_rel[1]),C[i])#+'--')

  plt.xticks([-5,-3,-2,-1,0,1],['0','0.001','0.01','0.1','1','10'],size=16)
  plt.xlabel('CTX concentration $D$ [μg/mL]',size=16)
  plt.ylabel('equil. EGF conc. $\overline{G}$ [pg/mL]',size=16)
  plt.yticks(size=16)
plt.legend(['CAF13000','CAF12905','CAF12905','CAF12911'],fontsize=13,title='CAF cell line',title_fontsize=13)
plt.show()