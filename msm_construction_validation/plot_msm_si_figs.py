"""Plot implied timescale, CK tests, and cross-validation scores"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyemma
import pickle

plt.rc('savefig',dpi=500)

def plot_its(dtrajs, max_its):

    its_object = pyemma.msm.its(dtrajs, errors='bayes', lags=600, nits=10)
    ax = pyemma.plots.plot_implied_timescales(its_object, units='ns', dt=0.07, show_mean=True, show_mle=False)
    plt.ylim(1e-1, max_its)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.savefig("its_plot_hq.png", transparent=True)

def plot_cktest(msm_object, n_states, save_prefix="cktest"):

    cktest = msm_object.cktest(n_states, mlags=range(4))
    fig, ax = pyemma.plots.plot_cktest(cktest)
    fig.savefig("%s_%d_state_hq.png"%(save_prefix, n_states), transparent=True)

def plot_cv_scores(score_results, n_tica=5, cls_per_tica=17):

    plt.figure()
    fig, ax = plt.subplots()

    for i in range(n_tica):
        plot_data = np.zeros((cls_per_tica, 3))
        scores = score_results[i*cls_per_tica:(i+1)*cls_per_tica]
        tica_label = scores[0][0]
        for j in range(len(scores)):
            plot_data[j,0] = scores[j][1]
            plot_data[j,1] = np.mean(np.exp(scores[j][2]))
            plot_data[j,2] = np.std(scores[j][2])*plot_data[j,1]

        #plt.plot(plot_data[:,0], plot_data[:,1])
        plt.errorbar(plot_data[:,0], plot_data[:,1], plot_data[:,2], capsize=2, label="%d TICs"%tica_label)
        plt.xlim(100,500)
        plt.ylim(60,160)

    plt.legend(fancybox=True, frameon=True, edgecolor='k', fontsize=14)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.xlabel("Number of Clusters")
    plt.ylabel("exp(GMRQ)")

    plt.savefig("cv_scores_hq.png", transparent=True)

def plot_microstate_cktest(dtrajs, msm_object, lag_time, mlags=range(4), save_prefix="microstate_cktest"):

    Tij = msm_object.P
    lags_list = []
    Tij_n_list = []
    Tij_est_list = []
    Tij_n_traces = []
    Tij_est_traces = []

    for mlag in mlags:
        lags_list.append(mlag*lag_time)
        Tij_n = np.linalg.matrix_power(Tij, mlag)
        Tij_n_traces.append(np.log(np.trace(Tij_n)))
        if mlag == 0:
            Tij_est = pyemma.msm.estimate_markov_model(dtrajs, 1, dt_traj='70 ps', score_method='VAMP1', score_k=10).P
        elif mlag == 1:
            Tij_est = Tij
        else:
            Tij_est = pyemma.msm.estimate_markov_model(dtrajs, mlag*lag_time, dt_traj='70 ps', score_method='VAMP1', score_k=10).P
        Tij_est_list.append(Tij_est)
        Tij_est_traces.append(np.log(np.trace(Tij_est)))

    print(Tij_n_traces)
    print(Tij_est_traces)

    plt.figure()

    fig, ax = plt.subplots()

    step_rng = np.array(mlags)*286
    plt.semilogy(np.array(mlags)*286, Tij_n_traces, color='b', linestyle='--', label="Predict")
    step_rng[0] = 1
    plt.semilogy(np.array(mlags)*286, Tij_est_traces, color='k', label="Estimate")
    plt.xlim(0,3*286)
    plt.ylim(1,1e3)
    plt.legend(fancybox=True, frameon=True, edgecolor='k')

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.xlabel("Lag Times (steps)")
    plt.ylabel("Trace(Tij)")
    fig.tight_layout()

    plt.savefig("cktest_micro.png", transparent=True)

if __name__=="__main__":

    matplotlib.rc('font',family='Helvetica-Normal',size=20)
    dtrajs = pickle.load(open("dtrajs.pkl", 'rb'))
    plot_its(dtrajs, 1e6)

    matplotlib.rc('font',family='Helvetica-Normal',size=16)
    score_results = pickle.load(open("score_results.pkl", 'rb'))
    plot_cv_scores(score_results)

    matplotlib.rc('font',family='Helvetica-Normal',size=14)
    msm = pickle.load(open("msm_object.pkl",'rb'))
    plot_cktest(msm, 5)

    matplotlib.rc('font',family='Helvetica-Normal',size=20)
    plot_microstate_cktest(dtrajs, msm, 286)
