# %% Imports
from gaussparams import GaussParams
import measurementmodels
import dynamicmodels
import ekf
import estimationstatistics as estats
import scipy
import scipy.stats
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:  # see if tqdm is available, otherwise define it as a dummy
    try:  # Ipython seem to require different tqdm... try..except seem to be the easiest way to check
        __IPYTHON__  # should only be defined for ipython
        from tqdm.notebook import tqdm
    except NameError:
        from tqdm import tqdm

except Exception as e:

    print(
        f'got "{type(e).__name__}: {e}": continuing without tqdm\n\t-->install tqdm to get progress bars'
    )

    def tqdm(iterable, *args, **kwargs):
        return iterable


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


# %% get and plot the data
data_path = "data_for_ekf.mat"

usePregen = True  # choose between own generated data and pregenerated

if usePregen:
    loadData: dict = scipy.io.loadmat(data_path)
    K: int = int(loadData["K"])  # The number of time steps
    Ts: float = float(loadData["Ts"])  # The sampling time
    Xgt: np.ndarray = loadData["Xgt"].T  # grounexutd truth
    Z: np.ndarray = loadData["Z"].T  # the measurements
else:
    from sample_CT_trajectory import sample_CT_trajectory

    # np.random.seed(10)  # random seed can be set for repeatability

    # inital state distribution
    x0 = np.array([0, 0, 1, 1, 0])
    P0 = np.diag([50, 50, 10, 10, np.pi / 4]) ** 2

    # model parameters
    sigma_a_true = 0.25
    sigma_omega_true = np.pi / 15
    sigma_z_true = 3

    # sampling interval a lenght
    K = 1000
    Ts = 0.1

    # get data
    Xgt, Z = sample_CT_trajectory(
        K, Ts, x0, P0, sigma_a_true, sigma_omega_true, sigma_z_true
    )

# show ground truth and measurements
fig, ax = plt.subplots(num=1, clear=True)
ax.scatter(*Z.T, color="C0", marker=".")
ax.plot(*Xgt.T[:2], color="C1")
ax.set_title("Data")

# show turnrate
fig2, ax2 = plt.subplots(num=2, clear=True)
ax2.plot(Xgt.T[4])
ax2.set_xlabel("time step")
ax2.set_ylabel("turn rate")


# %% a: tune by hand and comment

# set parameters
sigma_a = 2.6
sigma_z = 3.1

# create the model and estimator object
dynmod = dynamicmodels.WhitenoiseAccelleration(sigma_a)
measmod = measurementmodels.CartesianPosition(sigma_z)
ekf_filter = ekf.EKF(dynmod, measmod)
print(ekf_filter)

# Optimal init for model
mean = np.array([*Z[1], *(Z[1] - Z[0]) / Ts])
cov11 = sigma_z ** 2 * np.eye(2)
cov12 = sigma_z ** 2 * np.eye(2) / Ts
cov22 = (2 * sigma_z ** 2 / Ts ** 2 + sigma_a ** 2 * Ts / 3) * np.eye(2)
cov = np.block([[cov11, cov12], [cov12.T, cov22]])
init_ekfstate = GaussParams(mean, cov)

ekfpred_list = []
ekfupd_list = []
ekfupd = init_ekfstate
NIS = np.empty(K)
NEES_pred = np.empty(K)
NEES_upd = np.empty(K)
dists_pred = np.empty((K, 2))
dists_upd = np.empty((K, 2))
# estimate
for k, (zk, x_true_k) in enumerate(zip(Z[2:], Xgt[2:])):
    ekfpred = ekf_filter.predict(ekfupd, Ts)
    ekfupd = ekf_filter.update(zk, ekfpred)

    NIS[k] = ekf_filter.NIS(zk, ekfpred)
    NEES_pred[k] = estats.NEES(*ekfpred, x_true_k)
    NEES_upd[k] = estats.NEES(*ekfupd, x_true_k)

    diff_pred = ekfpred.mean - x_true_k[:4]
    diff_upd = ekfupd.mean - x_true_k[:4]
    dists_pred[k] = np.linalg.norm(diff_pred[:2]), np.linalg.norm(diff_pred[2:])
    dists_pred[k] = np.linalg.norm(diff_upd[:2]), np.linalg.norm(diff_upd[2:])

    ekfpred_list.append(ekfpred)
    ekfupd_list.append(ekfupd)


x_bar = np.array([pred.mean for pred in ekfpred_list])
P_bar = np.array([pred.cov for pred in ekfpred_list])

x_hat = np.array([upd.mean for upd in ekfupd_list])
P_hat = np.array([upd.cov for upd in ekfupd_list])

# %% Calculate average performance metrics
RMSE_pred = np.sqrt((dists_pred ** 2).mean(axis=0))
RMSE_upd = np.sqrt((dists_upd ** 2).mean(axis=0))

fig3, ax3 = plt.subplots(num=3, clear=True)

ax3.plot(*Xgt.T[:2])
ax3.plot(*x_hat.T[:2])
RMSEs_str = ", ".join(f"{v:.2f}" for v in (*RMSE_pred, *RMSE_upd))
ax3.set_title(
    rf"$\sigma_a = {sigma_a}$, $\sigma_z= {sigma_z}$,"
    + f"\nRMSE(p_p, p_v, u_p, u_v) = ({RMSEs_str})"
)

# %% Task 5 b and c

# % parameters for the parameter grid
n_vals = 20
sigma_a_low = 0.5
sigma_a_high = 10
sigma_z_low = 0.3
sigma_z_high = 12

# % set the grid on logscale(not mandatory)
sigma_a_list = np.logspace(
    np.log10(sigma_a_low), np.log10(sigma_a_high), n_vals, base=10
)
sigma_z_list = np.logspace(
    np.log10(sigma_z_low), np.log10(sigma_z_high), n_vals, base=10
)


NIS = np.empty((n_vals, n_vals, K - 2))
NEES_pred = np.zeros((n_vals, n_vals, K - 2))
NEES_upd = np.zeros((n_vals, n_vals, K - 2))
dists_pred = np.zeros((n_vals, n_vals, K - 2, 2))
dists_upd = np.zeros((n_vals, n_vals, K - 2, 2))
# %% run through the grid and estimate
for i, sigma_a in tqdm(enumerate(sigma_a_list), "sigma_a", leave=None, total=n_vals):
    dynmod = dynamicmodels.WhitenoiseAccelleration(sigma_a)
    for j, sigma_z in tqdm(
        enumerate(sigma_z_list), "sigma_z", leave=None, total=n_vals
    ):
        measmod = measurementmodels.CartesianPosition(sigma_z)
        ekf_filter = ekf.EKF(dynmod, measmod)
        # Optimal init according to our model.
        mean = np.array([*Z[1], *(Z[1] - Z[0]) / Ts])
        cov11 = sigma_z ** 2 * np.eye(2)
        cov12 = sigma_z ** 2 * np.eye(2) / Ts
        cov22 = (2 * sigma_z ** 2 / Ts ** 2 + sigma_a ** 2 * Ts / 3) * np.eye(2)
        cov = np.block([[cov11, cov12], [cov12.T, cov22]])
        ekfupd = GaussParams(mean, cov)
        # estimate
        for k, (zk, x_true_k) in enumerate(zip(Z[2:], Xgt[2:])):
            ekfpred = ekf_filter.predict(ekfupd, Ts)
            ekfupd = ekf_filter.update(zk, ekfpred)

            NIS[i, j, k] = ekf_filter.NIS(zk, ekfpred)
            NEES_pred[i, j, k] = estats.NEES(*ekfpred, x_true_k)
            NEES_upd[i, j, k] = estats.NEES(*ekfupd, x_true_k)

            diff_pred = ekfpred.mean - x_true_k[:4]
            diff_upd = ekfupd.mean - x_true_k[:4]
            dists_pred[i, j, k] = (
                np.linalg.norm(diff_pred[:2]),
                np.linalg.norm(diff_pred[2:]),
            )
            dists_pred[i, j, k] = (
                np.linalg.norm(diff_upd[:2]),
                np.linalg.norm(diff_upd[2:]),
            )

# %% calculate averages
RMSE_pred = np.sqrt((dists_pred ** 2).mean(axis=2))
RMSE_upd = np.sqrt((dists_upd ** 2).mean(axis=2))
ANEES_pred = NEES_pred.mean(axis=2)
ANEES_upd = NEES_upd.mean(axis=2)
ANIS = NIS.mean(axis=2)

# %% interpolate ANEES/NIS
ANIS_spline = scipy.interpolate.RectBivariateSpline(sigma_a_list, sigma_z_list, ANIS)
ANEES_pred_spline = scipy.interpolate.RectBivariateSpline(
    sigma_a_list, sigma_z_list, ANEES_pred
)
ANEES_upd_spline = scipy.interpolate.RectBivariateSpline(
    sigma_a_list, sigma_z_list, ANEES_upd
)

n_eval = 100
mesh_a, mesh_z = np.meshgrid(
    np.linspace(sigma_a_low, sigma_a_high, n_eval),
    np.linspace(sigma_z_low, sigma_z_high, n_eval),
)
ANIS_eval = ANIS_spline(mesh_a.ravel(), mesh_z.ravel(), grid=False).reshape(
    mesh_a.shape
)
ANEES_pred_eval = ANEES_pred_spline(mesh_a.ravel(), mesh_z.ravel(), grid=False).reshape(
    mesh_a.shape
)
ANEES_upd_eval = ANEES_upd_spline(mesh_a.ravel(), mesh_z.ravel(), grid=False).reshape(
    mesh_a.shape
)  # %% find confidence regions for NIS and plot

# %% confidence plots
confprob = 0.9
CINIS = np.array(scipy.stats.chi2.interval(0.9, 2 * K)) / K

# plot
fig4 = plt.figure(4, clear=True)
ax4 = plt.gca(projection="3d")
ax4.plot_surface(mesh_a, mesh_z, ANIS_eval, alpha=0.9)
ax4.contour(
    mesh_a, mesh_z, ANIS_eval, [1, 1.5, *CINIS, 2.5, 3], offset=0
)  # , extend3d=True, colors='yellow')
ax4.set_xlabel(r"$\sigma_a$")
ax4.set_ylabel(r"$\sigma_z$")
ax4.set_zlabel("ANIS")
ax4.set_zlim(0, 10)
ax4.view_init(30, 20)

# %% find confidence regions for NEES and plot
confprob = 0.9
CINEES = np.array(scipy.stats.chi2.interval(0.9, 4 * K)) / K
print(CINEES)

# plot
fig5 = plt.figure(5, clear=True)
ax5s = [
    fig5.add_subplot(1, 2, 1, projection="3d"),
    fig5.add_subplot(1, 2, 2, projection="3d"),
]
ax5s[0].plot_surface(mesh_a, mesh_z, ANEES_pred_eval, alpha=0.9)
ax5s[0].contour(
    mesh_a, mesh_z, ANEES_pred_eval, [3, 3.5, *CINEES, 4.5, 5], offset=0,
)
ax5s[0].set_xlabel(r"$\sigma_a$")
ax5s[0].set_ylabel(r"$\sigma_z$")
ax5s[0].set_zlabel("ANEES_pred")
ax5s[0].set_zlim(0, 50)
ax5s[0].view_init(40, 30)

ax5s[1].plot_surface(mesh_a, mesh_z, ANEES_upd_eval, alpha=0.9)
ax5s[1].contour(
    mesh_a, mesh_z, ANEES_upd_eval, [3, 3.5, *CINEES, 4.5, 5], offset=0,
)
ax5s[1].set_xlabel(r"$\sigma_a$")
ax5s[1].set_ylabel(r"$\sigma_z$")
ax5s[1].set_zlabel("ANEES_upd")
ax5s[1].set_zlim(0, 50)
ax5s[1].view_init(40, 30)

# %% see the intersection of NIS and NEESes
fig6, ax6 = plt.subplots(num=6, clear=True)
cont_upd = ax6.contour(mesh_a, mesh_z, ANEES_upd_eval, CINEES, colors=["C0", "C1"])
cont_pred = ax6.contour(mesh_a, mesh_z, ANEES_pred_eval, CINEES, colors=["C2", "C3"])
cont_nis = ax6.contour(mesh_a, mesh_z, ANIS_eval, CINIS, colors=["C4", "C5"])

for cs, l in zip([cont_upd, cont_pred, cont_nis], ["NEESupd", "NEESpred", "NIS"]):
    for c, hl in zip(cs.collections, ["low", "high"]):
        c.set_label(l + "_" + hl)
ax6.legend()
ax6.set_xlabel(r"$\sigma_a$")
ax6.set_ylabel(r"$\sigma_z$")

# %% show all the plots
plt.show()

# %%
# figure(6)
# clf
# grid on
# subplot(2, 1, 1)
# surfc(qs, rs, Ainnov(: , : , 1)')
# zlabel('mean(\nu_1)')
# xlabel('q')
# ylabel('r')
# subplot(2, 1, 2)

# surfc(qs, rs, Ainnov(:, : , 2)')
# zlabel('mean(\nu_2)')
# xlabel('q')
# ylabel('r')

# figure(7)
# clf
# surfc(qs, rs, abs(Ainnov(: , : , 1)') + abs(Ainnov(:, :, 2)'))
# zlabel('|mean(\nu_1)| + |mean(\nu_2)|')
# xlabel('q')
# ylabel('r')

# figure(7)
# clf
# contourlevels = [-50, -15, -5, 0, 5, 15, 50]
# cmlvls = (-3: 3)*3 + 32
# innovCorr = reshape(innovCorr, [Nvals, Nvals+1, 11, 2, 2])
# for k = 0:
#     5
#     for i = 1:
#         2
#         for j = 1:
#             2
#             subplot(6, 4, 4*k + (i-1)*2 + j)
#             hold on
#             zlim([-100, 100])
#             caxis([-100, 100])
#             surfc(qs, rs, innovCorr(:, : , 5 + k, i, j)')
#             contour3(qs, rs, innovCorr(:, : , 5 + k, i, j)', contourlevels,'r')
#             contour3(qs, rs, innovCorr(:, : , 5 + k, i, j)', [0, 0],'g')
#             view([45, 45])
#             zlabel(sprintf('R_{%d, %d}(%d)', i, j, k))
#             xlabel('q')
#             ylabel('r')
#         end
#     end
# end
# suptitle('autocrosscorrelation of innovation')

# figure(9)
# clf

# subplot(1, 3, 1)
# hold on
# surfc(qs, rs, MSI', 'FaceColor', 'interp')
# mMSI = min(MSI(:))
# contour3(qs, rs, MSI', logspace(log10(mMSI*1.05),log10(mMSI*2),5), 'g')
# view([45, 45])
# caxis([0, 100])
# xlabel('q')
# ylabel('r')
# zlabel('MSI')

# subplot(1, 3, 2)
# hold on
# surfc(qs, rs, posMSE', 'FaceColor', 'interp')
# mpMSE = min(posMSE(:))
# contour3(qs, rs, posMSE', logspace(log10(mpMSE*1.05),log10(mpMSE*2),5), 'g')
# view([45, 45])
# caxis([0, 50])
# xlabel('q')
# ylabel('r')
# zlabel('posMSE')

# subplot(1, 3, 3)
# hold on
# surfc(qs, rs, velMSE', 'FaceColor', 'interp')
# mvMSE = min(velMSE(:))
# contour3(qs, rs, velMSE', logspace(log10(mvMSE*1.05),log10(mvMSE*2),5), 'g')
# view([45, 45])
# caxis([0, 50])
# xlabel('q')
# ylabel('r')
# zlabel('velMSE')

# %%

# %%
