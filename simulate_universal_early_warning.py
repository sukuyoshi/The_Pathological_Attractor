import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.universal_disease_dynamics import UniversalDiseaseDynamics

def run_early_warning_simulation():
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    diseases = ["diabetes", "cancer", "sjogren", "atopic"]
    t_span = (0.0, 120.0)
    t_eval = np.linspace(0.0, 120.0, 1201)
    dt = t_eval[1] - t_eval[0]
    
    y0_healthy = [0.88, 0.02, 0.02, 1.35]
    
    data_dict = {"time_days": t_eval}
    lead_times = {}
    
    plt.figure(figsize=(16, 12))
    
    for idx, dis in enumerate(diseases):
        system = UniversalDiseaseDynamics(disease_type=dis)
        
        def transition_dynamics(t, y):
            stress_ramp = system.sigma_ext * (1.0 + 2.5 * (t / 60.0)**1.5)
            trigger_cyt = 0.040 * (t / 50.0)**1.5
            res = system.vector_field(t, y, sigma_ext=stress_ramp)
            res[2] += trigger_cyt
            return res
            
        sol = solve_ivp(
            transition_dynamics,
            t_span,
            y0_healthy,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )
        
        n_pts = len(sol.t)
        omega1_arr = np.zeros(n_pts)
        theta_arr = np.zeros(n_pts)
        
        window_pts = 35
        for k in range(n_pts):
            start_idx = max(0, k - window_pts + 1)
            if k - start_idx < 8:
                omega1_arr[k] = 0.10
                theta_arr[k] = -0.20
                continue
                
            y_slice = sol.y[:, start_idx : k + 1]
            y_mean = np.mean(y_slice, axis=1, keepdims=True)
            y_normed = y_slice - y_mean
            
            y0_mat = y_normed[:, :-1]
            y1_mat = y_normed[:, 1:]
            
            reg_eye = 1e-4 * np.eye(4)
            cov0 = y0_mat @ y0_mat.T + reg_eye
            cross_cov = y1_mat @ y0_mat.T
            
            try:
                a_mat = cross_cov @ np.linalg.inv(cov0)
                eig_vals = np.linalg.eigvals(a_mat)
                cont_eigs = np.log(np.maximum(1e-6, np.abs(eig_vals))) / dt
                omega1_arr[k] = - float(np.max(cont_eigs))
                theta_arr[k] = float(np.sum(cont_eigs))
            except Exception:
                omega1_arr[k] = omega1_arr[k - 1]
                theta_arr[k] = theta_arr[k - 1]
                
        smooth_span = 25
        kernel = np.ones(smooth_span) / smooth_span
        omega1_smooth = np.convolve(omega1_arr, kernel, mode="same")
        
        data_dict[f"{dis}_P"] = sol.y[0]
        data_dict[f"{dis}_D"] = sol.y[1]
        data_dict[f"{dis}_Icyt"] = sol.y[2]
        data_dict[f"{dis}_Ireg"] = sol.y[3]
        data_dict[f"{dis}_omega1"] = omega1_smooth
        data_dict[f"{dis}_theta"] = theta_arr
        
        warn_mask = (omega1_smooth < 0.045) & (sol.t > 15.0)
        warning_day = float(sol.t[np.argmax(warn_mask)]) if np.any(warn_mask) else None
        
        symptom_mask = (sol.y[1] > 0.25) & (sol.t > 15.0)
        symptom_day = float(sol.t[np.argmax(symptom_mask)]) if np.any(symptom_mask) else None
        
        lead_time = (symptom_day - warning_day) if (warning_day and symptom_day) else 0.0
        lead_times[dis] = {
            "warning_day": warning_day,
            "symptom_day": symptom_day,
            "lead_time": lead_time
        }
        
        plt.subplot(2, 2, idx + 1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        l1 = ax1.plot(sol.t, sol.y[0], "b-", linewidth=2.2, label="P (Parenchymal Integrity)")
        l2 = ax1.plot(sol.t, sol.y[1], "r-", linewidth=2.2, label="D (Tissue Damage / Lesion)")
        l3 = ax2.plot(sol.t, omega1_smooth, "g--", linewidth=2.0, label="Omega_1 (Spectral Gap)")
        
        ax1.axhline(0.25, color="darkred", linestyle=":", label="Clinical Threshold (D=0.25)")
        ax2.axhline(0.045, color="green", linestyle=":", label="Softening Alert (Omega_1 < 0.045)")
        
        if warning_day and symptom_day:
            ax1.axvspan(warning_day, symptom_day, color="yellow", alpha=0.20, label=f"Pre-Symptomatic Window: {lead_time:.1f} Days")
            ax1.axvline(warning_day, color="green", linestyle="--", linewidth=1.8)
            ax1.axvline(symptom_day, color="darkred", linestyle="--", linewidth=1.8)
            
        ax1.set_xlabel("Time (Days)", fontsize=11)
        ax1.set_ylabel("Tissue State (P, D)", fontsize=11, color="blue")
        ax2.set_ylabel("Koopman Spectral Gap Omega_1", fontsize=11, color="green")
        plt.title(f"{system.name}: Pre-Transition Spectral Softening", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=8)
        
        print(f"=== {dis.upper()} ({system.name}) ===")
        print(f"  Omega_1 Softening Alert Day: {warning_day:.1f}")
        print(f"  Clinical Damage Onset Day:   {symptom_day:.1f}")
        print(f"  Actionable Lead Time:        {lead_time:.1f} days ahead of clinical symptoms")

    plt.tight_layout()
    plot_path = os.path.join(res_dir, "universal_early_warning_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    df = pd.DataFrame(data_dict)
    csv_path = os.path.join(res_dir, "universal_early_warning_simulation.csv")
    df.to_csv(csv_path, index=False)
    
    print("Universal Early Warning Simulation Completed Successfully.")
    print(f"Results saved to {csv_path} and {plot_path}")

if __name__ == "__main__":
    run_early_warning_simulation()