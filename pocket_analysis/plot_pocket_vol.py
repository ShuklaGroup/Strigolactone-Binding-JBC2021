import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import glob
import pickle

plt.rc('savefig', dpi=300)
matplotlib.rc('font',family='Helvetica-Normal',size=20)

host_files = sorted(glob.glob("/home/jiming/Storage/AtD14_binding_analysis/pocket_calc/POVME_output_manual_sphere/*tabbed*"))
parasite_files = sorted(glob.glob("/home/jiming/Documents/Striga/ShHTL7_binding/pocket_calc/POVME_output_manual_sphere/*tabbed*"))

host = []
for f in host_files:
    host.extend(np.loadtxt(f)[:,1])

host = np.array(host)
host_msm_weights = pickle.load(open("/home/jiming/Storage/AtD14_binding_analysis/msm_final_fixed/msm_weights.pkl",'rb'))
weights_subsampled = []
for w in host_msm_weights:
    weights_subsampled.append(w[::20])
host_weights_use = np.concatenate(weights_subsampled)

parasite = []
for f in parasite_files:
    parasite.extend(np.loadtxt(f)[:,1])

parasite = np.array(parasite)

print(np.min(host))
print(np.max(host))
print(np.min(parasite))
print(np.max(parasite))

parasite_msm_weights = pickle.load(open("/home/jiming/Documents/Striga/ShHTL7_binding/msm_final_fixed/msm_weights.pkl",'rb'))

weights_subsampled = []
for w in parasite_msm_weights:
    weights_subsampled.append(w[::20])
parasite_weights_use = np.concatenate(weights_subsampled)
print(np.shape(parasite))
print(np.shape(parasite_weights_use))

hist_host, bins = np.histogram(host, bins=500, range=(0,1000), weights=host_weights_use)
hist_parasite, bins = np.histogram(parasite, bins=500, range=(0,1000), weights=parasite_weights_use)

normed_host = hist_host/float(np.sum(hist_host))
normed_parasite = hist_parasite/float(np.sum(hist_parasite))
rng = 0.5*(bins[0:-1] + bins[1:])

print(rng)

#Crystal volume markers
P_crys_host = normed_host[rng==215]
P_crys_parasite = 0.5*(normed_parasite[rng==357]+normed_parasite[rng==359])
print(P_crys_host)
print(P_crys_parasite)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.plot(rng, normed_host, label="AtD14", color='b')
plt.plot([215, 215], [0, 0.025], color='b')
plt.plot(rng, normed_parasite, label="ShHTL7", color='r')
plt.plot([358, 358], [0, 0.025], color='r')

plt.xlim(0,800)
plt.ylim(0, 0.025)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
plt.xlabel("Pocket Volume ($\AA^3$)")
plt.ylabel("Probability Density")
plt.legend(fancybox=False, framealpha=0.0)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
fig.tight_layout()
plt.savefig("pocket_vol_weighted_withCrys.png", transparent=True)

def calc_kl(dist1, dist2, dx):
    dist1_masked = np.ma.masked_where(dist1==0, dist1)
    dist2_masked = np.ma.masked_where(dist2==0, dist2)
    kl = dx*np.sum(dist1_masked*np.log(dist1_masked/dist2_masked))
    return kl

def calc_avg(x, prob):
    #dx = x[1] - x[0] #Assumes uniform spacing over interval
    avg = np.sum(x*prob)
    return avg

def calc_sd(x, avg, prob):
    var = np.sum((x-avg)**2*prob)
    sd = np.sqrt(var)
    return sd

avg_host = calc_avg(rng, normed_host)
avg_parasite = calc_avg(rng, normed_parasite)
print(avg_host)
print(avg_parasite)

sd_host = calc_sd(rng, avg_host, normed_host)
sd_parasite = calc_sd(rng, avg_parasite, normed_parasite)
print(sd_host)
print(sd_parasite)
