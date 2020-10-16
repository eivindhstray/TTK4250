# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# %% local
from gaussparams import GaussParams
from mixturedata import MixtureParameters
import estimationstatistics as estats
import dynamicmodels
import measurementmodels
import ekf
import imm
import pda

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )

plot_save_path = "./plots/joyride_ekf/"

# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].squeeze()
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]
T_mean = np.mean(Ts)

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

# %% play measurement movie. Remember that you can cross out the window
play_movie = False
play_slice = slice(0, K)
if play_movie:
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    fig2, ax2 = plt.subplots(num=2, clear=True)
    sh = ax2.scatter(np.nan, np.nan)
    th = ax2.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = 0.1
    # sets a pause in between time steps if it goes to fast
    
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)

# %% setup and track
# sensor
sigma_z = 17
clutter_intensity = 1e-5
PD = 0.9
gate_size = 5

# dynamic models
sigma_a_CV = 2
sigma_a_CT = 2
sigma_omega = 0.03 #* np.pi

mean_init = Xgt[0]
mean_init = np.append(mean_init, 0.1)
cov_init = np.diag([2*sigma_z, 2*sigma_z, 2, 2, 0.1])

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))

trackers = []
trackers.append(pda.PDA(ekf_filters[0], clutter_intensity, PD, gate_size)) # EKF CV
trackers.append(pda.PDA(ekf_filters[1], clutter_intensity, PD, gate_size)) # EKF CT

names = ["CV_EKF", "CT_EKF"]

init_ekf_state = GaussParams(mean_init, cov_init)

# NEES = np.zeros(K)
# NEESpos = np.zeros(K)
# NEESvel = np.zeros(K)

tracker_update_init = [init_ekf_state, init_ekf_state]
tracker_update_list = np.empty((len(trackers), len(Xgt)), dtype=GaussParams)
tracker_predict_list = np.empty((len(trackers), len(Xgt)), dtype=GaussParams)
tracker_estimate_list = np.empty((len(trackers), len(Xgt)), dtype=GaussParams)
# estimate
Ts = np.insert(Ts,0, 0., axis=0)

x_hat = np.empty((len(trackers), len(Xgt), 5))
prob_hat = np.empty((len(trackers), len(Xgt), 2))

NEES = np.empty((len(trackers), len(Xgt)))
NEESpos = np.empty((len(trackers), len(Xgt)))
NEESvel = np.empty((len(trackers), len(Xgt)))

for i, (tracker, name) in enumerate(zip(trackers, names)):
    print("Running: ",name)
    for k, (Zk, x_true_k, Tsk) in enumerate(zip(Z, Xgt, Ts)):
        if k == 0:
            tracker_predict = tracker.predict(tracker_update_init[i], Tsk)
        else:
            tracker_predict = tracker.predict(tracker_update, Tsk)
        tracker_update = tracker.update(Zk, tracker_predict)

        # You can look at the prediction estimate as well
        tracker_estimate = tracker.estimate(tracker_update)

        NEES[i][k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(4))
        NEESpos[i][k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2))
        NEESvel[i][k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2, 4))

        tracker_predict_list[i][k]= tracker_predict
        tracker_update_list[i][k] = tracker_update
        tracker_estimate_list[i][k] = tracker_estimate

    x_hat[i] = np.array([est.mean for est in tracker_estimate_list[i]])

# calculate performance metrics
posRMSE = np.empty((len(trackers)), dtype=float)
velRMSE = np.empty((len(trackers)), dtype=float)
peak_pos_deviation = np.empty((len(trackers)), dtype=float)
peak_vel_deviation = np.empty((len(trackers)), dtype=float)

for i,_ in enumerate(trackers):
    poserr = np.linalg.norm(x_hat[i,:, :2] - Xgt[:, :2], axis=1)
    velerr = np.linalg.norm(x_hat[i,:, 2:4] - Xgt[:, 2:4], axis=1)
    posRMSE[i] = np.sqrt(
        np.mean(poserr ** 2)
    )  # not true RMSE (which is over monte carlo simulations)
    velRMSE[i] = np.sqrt(np.mean(velerr ** 2))
    # not true RMSE (which is over monte carlo simulations)
    peak_pos_deviation[i] = poserr.max()
    peak_vel_deviation[i] = velerr.max()

#Consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)

print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# %% plots
# trajectory
fig3, axs3 = plt.subplots(1, 2, num=2, clear=True)
for i in range(posRMSE.shape[0]):
    if i == 0:
        axs3[i].plot(*x_hat[0].T[:2], label=r"$Predicted Track CV$")
    else:
        axs3[i].plot(*x_hat[1].T[:2], label=r"$Predicted track CT$")

    axs3[i].plot(*Xgt.T[:2], label=r"$Ground truth$")
    axs3[i].legend()
    axs3[i].set_title(
         f"RMSE(pos, vel) = ({posRMSE[i]:.3f}, {velRMSE[i]:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation[i]:.3f}, {peak_vel_deviation[i]:.3f})"
    )
     


# NEES
for i in range(NEESpos.shape[0]):
    fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
    axs4[0].plot(np.arange(K) * T_mean, NEESpos[i,:])
    axs4[0].plot([0, (K - 1) * T_mean], np.repeat(CI2[None], 2, 0), "--r")
    axs4[0].set_ylabel("NEES pos")
    inCIpos = np.mean((CI2[0] <= NEESpos[i,:]) * (NEESpos[i,:] <= CI2[1]))
    axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

    axs4[1].plot(np.arange(K) * T_mean, NEESvel[i,:])
    axs4[1].plot([0, (K - 1) * T_mean], np.repeat(CI2[None], 2, 0), "--r")
    axs4[1].set_ylabel("NEES vel")
    inCIvel = np.mean((CI2[0] <= NEESvel[i,:]) * (NEESvel[i,:] <= CI2[1]))
    axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

    axs4[2].plot(np.arange(K) * T_mean, NEES[i,:])
    axs4[2].plot([0, (K - 1) * T_mean], np.repeat(CI4[None], 2, 0), "--r")
    axs4[2].set_ylabel("NEES")
    inCI = np.mean((CI4[0] <= NEES[i,:]) * (NEES[i,:] <= CI4[1]))
    axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")


    plt.show()

# errors
for i,_ in enumerate(trackers):
    fig5, axs5 = plt.subplots(2, num=5, clear=True) 
    axs5[0].set_title(names[i])
    axs5[0].plot(np.arange(K) * T_mean, np.linalg.norm(x_hat[i][:, :2] - Xgt[:, :2], axis=1))
    axs5[0].set_ylabel("position error")

    axs5[1].plot(np.arange(K) * T_mean, np.linalg.norm(x_hat[i][:, 2:4] - Xgt[:, 2:4], axis=1))
    axs5[1].set_ylabel("velocity error")

   

    plt.show()