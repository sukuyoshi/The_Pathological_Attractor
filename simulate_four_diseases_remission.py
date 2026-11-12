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

def run_simulation():
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    diseases = ["diabetes", "cancer", "sjogren", "atopic"]
    t_span = (0.0, 180.0)
    t_eval = np.linspace(0.0, 180.0, 1801)
    
    pulse_start = 20.0
    pulse_end = 70.0
    
    initial_chronic_states = {
        "diabetes": [0.12, 0.55, 1.10, 0.05],
        "cancer": [0.10, 0.65, 1.20, 0.04],
        "sjogren": [0.11, 0.58, 1.15, 0.05],
        "atopic": [0.14, 0.52, 1.05, 0.06]
    }
    
    data_dict = {"time_days": t_eval}
    
    plt.figure(figsize=(16, 12))
    
    for idx, dis in enumerate(diseases):
        system = UniversalDiseaseDynamics(disease_type=dis)
        y0 = initial_chronic_states[dis]
        
        def chronic_dynamics(t, y):
            return system.vector_field(t, y)
            
        def pulse_dynamics(t, y):
            if pulse_start <= t <= pulse_end:
                u_rec = system.pulse_u_rec
                u_reg = system.pulse_u_reg
                u_anti = system.pulse_u_anti
            else:
                u_rec = 0.0
                u_reg = 0.0
                u_anti = 0.0
            return system.vector_field(t, y, u_rec=u_rec, u_reg=u_reg, u_anti=u_anti)
            
        sol_chronic = solve_ivp(
            chronic_dynamics,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )
        
        sol_pulse = solve_ivp(
            pulse_dynamics,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8
        )
        
        data_dict[f"{dis}_P_chronic"] = sol_chronic.y[0]
        data_dict[f"{dis}_D_chronic"] = sol_chronic.y[1]
        data_dict[f"{dis}_Icyt_chronic"] = sol_chronic.y[2]
        data_dict[f"{dis}_Ireg_chronic"] = sol_chronic.y[3]
        
        data_dict[f"{dis}_P_remission"] = sol_pulse.y[0]
        data_dict[f"{dis}_D_remission"] = sol_pulse.y[1]
        data_dict[f"{dis}_Icyt_remission"] = sol_pulse.y[2]
        data_dict[f"{dis}_Ireg_remission"] = sol_pulse.y[3]
        
        plt.subplot(2, 2, idx + 1)
        plt.plot(t_eval, sol_chronic.y[0], "r--", linewidth=1.8, label="P (Chronic Untreated)")
        plt.plot(t_eval, sol_chronic.y[1], "m--", linewidth=1.8, label="D (Chronic Untreated)")
        plt.plot(t_eval, sol_pulse.y[0], "b-", linewidth=2.5, label="P (Curative Remission)")
        plt.plot(t_eval, sol_pulse.y[1], "g-", linewidth=2.2, label="D (Cleared / Normalized)")
        plt.plot(t_eval, sol_pulse.y[2], "orange", linestyle="-.", linewidth=1.5, label="I_cyt (Suppressed)")
        plt.plot(t_eval, sol_pulse.y[3], "cyan", linestyle=":", linewidth=1.8, label="I_reg (Restored Tolerance)")
        
        plt.axvspan(pulse_start, pulse_end, color="gray", alpha=0.15, label="Pulse Intervention Window")
        
        plt.xlabel("Time (Days)", fontsize=11)
        plt.ylabel("Normalized Cell / Immune Density", fontsize=11)
        plt.title(f"{system.name}: Saddle-Node Separatrix Traversal", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="center right", fontsize=8)
        
        day20_idx = int(pulse_start / 0.1)
        day70_idx = int(pulse_end / 0.1)
        print(f"=== {dis.upper()} ({system.name}) ===")
        print(f"Baseline Chronic (Day 20): P={sol_pulse.y[0, day20_idx]:.4f}, D={sol_pulse.y[1, day20_idx]:.4f}, Icyt={sol_pulse.y[2, day20_idx]:.4f}, Ireg={sol_pulse.y[3, day20_idx]:.4f}")
        print(f"End of Pulse (Day 70):     P={sol_pulse.y[0, day70_idx]:.4f}, D={sol_pulse.y[1, day70_idx]:.4f}, Icyt={sol_pulse.y[2, day70_idx]:.4f}, Ireg={sol_pulse.y[3, day70_idx]:.4f}")
        print(f"Final Outcome (Day 180):   P={sol_pulse.y[0, -1]:.4f}, D={sol_pulse.y[1, -1]:.4f}, Icyt={sol_pulse.y[2, -1]:.4f}, Ireg={sol_pulse.y[3, -1]:.4f}")
        print(f"Untreated State (Day 180): P={sol_chronic.y[0, -1]:.4f}, D={sol_chronic.y[1, -1]:.4f}, Icyt={sol_chronic.y[2, -1]:.4f}, Ireg={sol_chronic.y[3, -1]:.4f}")
        
    plt.tight_layout()
    plot_path = os.path.join(res_dir, "four_diseases_phase_portraits.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    df = pd.DataFrame(data_dict)
    csv_path = os.path.join(res_dir, "four_diseases_remission_simulation.csv")
    df.to_csv(csv_path, index=False)
    
    print("Four Diseases Remission Simulation Completed Successfully.")
    print(f"Results saved to {csv_path} and {plot_path}")

if __name__ == "__main__":
    run_simulation()