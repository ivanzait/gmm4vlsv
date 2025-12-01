import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
import os

def unpack_results(results, filename, N_pop):
    means = results[filename]['means'][N_pop]
    covariances = results[filename]['covariances'][N_pop]
    weights = results[filename]['weights'][N_pop]
    bic = results[filename]['bic'][N_pop]
    aic = results[filename]['aic'][N_pop]
    return means, covariances, weights, bic, aic


def calculate_thermal_energy(results, N_pop):
    E_th_k = np.zeros(len(results))
    for tt in range(len(results)):    
        filename = list(results.keys())[tt]  # or pick manually
        means, covariances, weights, bic, aic = unpack_results(results, filename, N_pop=N_pop)    
        E_th_n = np.zeros(N_pop)
        for i in range(N_pop):                
            vals, vecs = np.linalg.eigh(covariances[i])        
            idx = np.argsort(vals)[::-1]
            sorted_vals = vals[idx]
            v_th_x, v_th_y, v_th_z = np.sqrt(sorted_vals)
            mass_ion = 1.6726219e-27  # kg (proton mass)    
            E_th_n[i] = 0.5 * mass_ion * (v_th_x**2 + v_th_y**2 + v_th_z**2)        
            E_th_k[tt] += weights[i] * E_th_n[i]        
    return E_th_k        
    #print(f"Thermal energy for population {i}: { int(E_th_k[tt]/1.60218e-19)} eV")




def plot_vdf_with_gmm(results, filename, vdf, shape, lims, N_pop=2, save_fig=False, img_size=(1024,1024)):
    
    t_ind =  filename.split('_')[-3]
    # --- Velocity axes
    vx = np.linspace(lims[0], lims[3], shape[0])
    vy = np.linspace(lims[1], lims[4], shape[1])
    vz = np.linspace(lims[2], lims[5], shape[2])
    
    # --- 2D projections of the VDF ---
    vdf_xy = np.sum(vdf, axis=2).T
    vdf_xz = np.sum(vdf, axis=1).T
    vdf_yz = np.sum(vdf, axis=0).T

    # --- Create figure with fixed size in inches to match desired pixels ---
    dpi = 100  # adjust so that figsize*dpi = img_size
    fig_w, fig_h = img_size[0]/dpi, img_size[1]/dpi
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    sz = 0.25  # size of each square axis (width = height)
    spacing = 0.06  # small horizontal spacing between squares
    bottom = 0.2   # distance from bottom
    left_start = 0.1  # left margin
    # Compute positions
    ax_xy = plt.axes([left_start, bottom, sz, sz])
    ax_xz = plt.axes([left_start + sz + spacing, bottom, sz, sz])
    ax_yz = plt.axes([left_start + 2*(sz + spacing), bottom, sz, sz])

    # --- Plot VDF projections ---
    ax_xy.pcolor(vx, vy, vdf_xy, cmap='jet', shading='auto', norm=LogNorm(vmin=1e-16, vmax=np.max(vdf_xy)))
    ax_xz.pcolor(vx, vz, vdf_xz, cmap='jet', shading='auto', norm=LogNorm(vmin=1e-16, vmax=np.max(vdf_xz)))
    ax_yz.pcolor(vy, vz, vdf_yz, cmap='jet', shading='auto', norm=LogNorm(vmin=1e-16, vmax=np.max(vdf_yz)))

    # --- GMM ellipses ---
    means = results[filename]['means'][N_pop]
    covs  = results[filename]['covariances'][N_pop]

    for i in range(len(means)):
        mean_vx, mean_vy, mean_vz = means[i]
        mean_pairs = [[mean_vx, mean_vy], [mean_vx, mean_vz], [mean_vy, mean_vz]]
        cov_xy = covs[i][:2, :2]
        cov_xz = covs[i][[0,2], :][:, [0,2]]
        cov_yz = covs[i][1:, 1:]

        for cov, ax, mean in zip([cov_xy, cov_xz, cov_yz],
                                 [ax_xy, ax_xz, ax_yz],
                                 mean_pairs):
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            major_val, minor_val = vals[order]
            major_vec = vecs[:, order[0]]
            width, height = 2*np.sqrt(major_val), 2*np.sqrt(minor_val)
            angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

            ellipse = Ellipse(mean, width, height, angle=angle,
                              edgecolor='black', facecolor='none', lw=2)
            ax.add_patch(ellipse)
            ax.plot(mean[0], mean[1], 'bX', markersize=8)

    # --- Axis labels and titles ---
    ax_xy.set_xlabel('Vx')
    ax_xy.set_ylabel('Vy')
    ax_xy.set_title('Vx–Vy')

    ax_xz.set_xlabel('Vx')
    ax_xz.set_ylabel('Vz')
    ax_xz.set_title('Vx–Vz')

    ax_yz.set_xlabel('Vy')
    ax_yz.set_ylabel('Vz')
    ax_yz.set_title('Vy–Vz')
    plt.suptitle(f"GMM populations (N={N_pop}) for {filename}", y=0.5, x=0.44)
    
    
    folder_path = "gmm_vdf_plots"
    os.makedirs(folder_path, exist_ok=True)
    if save_fig:
        plt.savefig(os.path.join(folder_path, f"gmm_vdf_{t_ind}.png"),
                    dpi=dpi, bbox_inches=None, pad_inches=0.05)
    
    plt.show()


def plot_gmm(results, filename, N_pop=2, save_fig=False, img_size=(1024,1024)):
    
    t_ind =  filename.split('_')[-3]
    # # --- Velocity axes
    # vx = np.linspace(lims[0], lims[3], shape[0])
    # vy = np.linspace(lims[1], lims[4], shape[1])
    # vz = np.linspace(lims[2], lims[5], shape[2])
    
    # # --- 2D projections of the VDF ---
    # vdf_xy = np.sum(vdf, axis=2).T
    # vdf_xz = np.sum(vdf, axis=1).T
    # vdf_yz = np.sum(vdf, axis=0).T

    # --- Create figure with fixed size in inches to match desired pixels ---
    dpi = 100  # adjust so that figsize*dpi = img_size
    fig_w, fig_h = img_size[0]/dpi, img_size[1]/dpi
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    sz = 0.25  # size of each square axis (width = height)
    spacing = 0.06  # small horizontal spacing between squares
    bottom = 0.2   # distance from bottom
    left_start = 0.1  # left margin
    # Compute positions
    ax_xy = plt.axes([left_start, bottom, sz, sz])
    ax_xz = plt.axes([left_start + sz + spacing, bottom, sz, sz])
    ax_yz = plt.axes([left_start + 2*(sz + spacing), bottom, sz, sz])

    # --- Plot VDF projections ---
    # ax_xy.pcolor(vx, vy, vdf_xy, cmap='jet', shading='auto', norm=LogNorm(vmin=1e-16, vmax=np.max(vdf_xy)))
    # ax_xz.pcolor(vx, vz, vdf_xz, cmap='jet', shading='auto', norm=LogNorm(vmin=1e-16, vmax=np.max(vdf_xz)))
    # ax_yz.pcolor(vy, vz, vdf_yz, cmap='jet', shading='auto', norm=LogNorm(vmin=1e-16, vmax=np.max(vdf_yz)))

    # # --- GMM ellipses ---
    means = results[filename]['means'][N_pop]
    covs  = results[filename]['covariances'][N_pop]

    for i in range(len(means)):
        mean_vx, mean_vy, mean_vz = means[i]
        mean_pairs = [[mean_vx, mean_vy], [mean_vx, mean_vz], [mean_vy, mean_vz]]
        cov_xy = covs[i][:2, :2]
        cov_xz = covs[i][[0,2], :][:, [0,2]]
        cov_yz = covs[i][1:, 1:]

        for cov, ax, mean in zip([cov_xy, cov_xz, cov_yz],
                                 [ax_xy, ax_xz, ax_yz],
                                 mean_pairs):
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            major_val, minor_val = vals[order]
            major_vec = vecs[:, order[0]]
            width, height = 2*np.sqrt(major_val), 2*np.sqrt(minor_val)
            angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

            ellipse = Ellipse(mean, width, height, angle=angle,
                              edgecolor='black', facecolor='none', lw=2)
            ax.add_patch(ellipse)
            ax.plot(mean[0], mean[1], 'bX', markersize=8)

    # --- Axis labels and titles ---
    ax_xy.set_xlabel('Vx')
    ax_xy.set_ylabel('Vy')
    ax_xy.set_title('Vx–Vy')

    ax_xz.set_xlabel('Vx')
    ax_xz.set_ylabel('Vz')
    ax_xz.set_title('Vx–Vz')

    ax_yz.set_xlabel('Vy')
    ax_yz.set_ylabel('Vz')
    ax_yz.set_title('Vy–Vz')
    plt.suptitle(f"GMM populations (N={N_pop}) for {filename}", y=0.5, x=0.44)
    
    
    folder_path = "gmm_vdf_plots"
    os.makedirs(folder_path, exist_ok=True)
    if save_fig:
        plt.savefig(os.path.join(folder_path, f"gmm_vdf_{t_ind}.png"),
                    dpi=dpi, bbox_inches=None, pad_inches=0.05)
    
    plt.show()
