import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

birth0 = np.array([[0,0.01502],[0.001,0.01455],[0.01,0.01341],[0.1,0.0],[1,0.0],[10,0.0]])
birth1 = np.array([[0,0.01396],[0.001,0.01724],[0.01,0.01703],[0.1,0.00420],[1,0.00102],[10,0.0]])
birth2 = np.array([[0,0.01435],[0.001,0.01654],[0.01,0.01406],[0.1,0.01141],[1,0.00621],[10,0.0]])
birth3 = np.array([[0,0.01551],[0.001,0.01657],[0.01,0.01744],[0.1,0.01627],[1,0.00621],[10,0.00070]])
birth4 = np.array([[0,0.01644],[0.001,0.01506],[0.01,0.01277],[0.1,0.01362],[1,0.00735],[10,0.0]])
birth5 = np.array([[0,0.01621],[0.001,0.01735],[0.01,0.01540],[0.1,0.01641],[1,0.01271],[10,0.00195]])
birth6 = np.array([[0,0.01847],[0.001,0.01806],[0.01,0.01851],[0.1,0.01546],[1,0.01172],[10,0.00543]])
birth7 = np.array([[0,0.01828],[0.001,0.01919],[0.01,0.01588],[0.1,0.01645],[1,0.01485],[10,0.00655]])

death0 = np.array([[0,0.00144],[0.001,0.00114],[0.01,0.00120],[0.1,0.00520],[1,0.00665],[10,0.00733]])
death1 = np.array([[0,0.00072],[0.001,0.00094],[0.01,0.00112],[0.1,0.00101],[1,0.00327],[10,0.00481]])
death2 = np.array([[0,0.00025],[0.001,0.00141],[0.01,0.00102],[0.1,0.00098],[1,0.00274],[10,0.00463]])
death3 = np.array([[0,0.00111],[0.001,0.00130],[0.01,0.00139],[0.1,0.00038],[1,0.00271],[10,0.00307]])
death4 = np.array([[0,0.00123],[0.001,0.00132],[0.01,0.00133],[0.1,0.00024],[1,0.00123],[10,0.00278]])
death5 = np.array([[0,0.00187],[0.001,0.00167],[0.01,0.00120],[0.1,0.00104],[1,0.00054],[10,0.00217]])
death6 = np.array([[0,0.00307],[0.001,0.00254],[0.01,0.00367],[0.1,0.00048],[1,0.00106],[10,0.00467]])
death7 = np.array([[0,0.00291],[0.001,0.00264],[0.01,0.00191],[0.1,0.00092],[1,0.00067],[10,0.00365]])

dataset_birth = [birth0,birth1,birth2,birth3,birth4,birth5,birth6,birth7]
dataset_death = [death0,death1,death2,death3,death4,death5,death6,death7]

def NetGrowth(D,R_max,R_min,s,C_50):
    return R_min + (R_max - R_min) / (1 + np.power((D / C_50),s))

def simultaneous_error(para):
  R_max = para[0]
  R_min = para[1]
  s = para[2]
  E=0
  for i in range(8):
    for j in range(np.size(dataset_birth[i],0)):
      fctvalue = NetGrowth(dataset_birth[i][j][0],R_max,R_min,s,para[i+3])
      E += np.power((dataset_birth[i][j][1]-dataset_death[i][j][1]-fctvalue),2)
  return E

para0 = [0.015,-0.007,2,1,1,2,3,4,5,6,7]
print(simultaneous_error(para0))

result = opt.minimize(simultaneous_error,para0,method='L-BFGS-B',bounds=((0,0.02),(-0.01,0),(0,10),(0.00001,15),(0.00001,15),(0.00001,15),(0.00001,15),(0.00001,15),(0.00001,15),(0.00001,15),(0.00001,15)))
params = result.x
print(params)
print(simultaneous_error(params))


D_range = np.array([0,0.001,0.01,0.1,1,10])
D_range_adj = np.array([0.00001,0.001,0.01,0.1,1,10])
D_range_full_adj = np.linspace(0.00001,10,100000)

for i in range(8):
  plt.scatter(np.log10(D_range_adj),dataset_birth[i][:,1]-dataset_death[i][:,1])
  G_value = 100 * i
  plt.plot(np.log10(D_range_full_adj),NetGrowth(D_range_full_adj,params[0],params[1],params[2],params[i+3]),label=G_value)
plt.xticks([-5,-3,-2,-1,0,1],['0','0.001','0.01','0.1','1','10'],size=16)
plt.xlabel('CTX concentration $D$ [μg/mL]',size=15)
plt.ylabel('CRC net growth rate $r_C$ [1/h]',size=15)
plt.yticks(size=16)
plt.legend(title='EGF conc. $G$ [pg/mL]',fontsize=13,title_fontsize=13)
plt.show()


def C50(G,C50_max,s,GF_crit):
    return C50_max / (1 + np.exp(-s * (G - GF_crit)))

G_range = np.linspace(0,700,8)
G_range_full = np.linspace(0,1000,1000)

para_50,cov = opt.curve_fit(f=C50,xdata=G_range,ydata=params[3:11],p0 = [15,0.01,700],bounds = ([0,0,0],[20,1,1000]))
print(para_50)

plt.scatter(G_range,params[3:11])
plt.plot(G_range_full,C50(G_range_full,para_50[0],para_50[1],para_50[2]))
plt.xlabel('EGF concentration $G$ [pg/mL]',size = 15)
plt.xticks(size=16)
plt.ylabel('$D_{50}$ CTX conc. of $r_C$ [μg/mL]',size = 15)
plt.yticks(size=16)
plt.show()