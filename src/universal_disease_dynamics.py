import numpy as np

class UniversalDiseaseDynamics:
    DISEASE_PRESETS = {
        "diabetes": {
            "name": "Type 1 & Type 2 Diabetes Mellitus",
            "P_label": "Functional Beta-Cell Mass (P)",
            "D_label": "Dedifferentiated / Senescent Beta-Cells (D)",
            "I_cyt_label": "Ectopic Fat Load / Autoimmune T-Cells (I_cyt)",
            "I_reg_label": "Insulin Sensitivity / Regulatory Tregs (I_reg)",
            "u_label": "Caloric Restriction & GLP-1 Pulse",
            "K_tissue": 1.0,
            "K_cyt": 2.0,
            "K_reg": 2.0,
            "r_P": 0.48,
            "r_D": 0.18,
            "k_rec0": 0.05,
            "gamma_rec": 1.00,
            "alpha_I": 0.50,
            "delta_P": 0.02,
            "k_ind0": 0.005,
            "gamma_ind_I": 0.45,
            "gamma_ind_sigma": 0.35,
            "kappa_kill": 0.50,
            "delta_D": 0.08,
            "s_cyt": 0.015,
            "rho_cyt": 0.60,
            "mu_reg": 0.85,
            "d_cyt": 0.20,
            "gamma_anti": 0.80,
            "s_reg": 0.03,
            "rho_reg": 0.55,
            "mu_cyt": 0.75,
            "d_reg": 0.15,
            "sigma_ext": 0.45,
            "pulse_u_rec": 1.5,
            "pulse_u_reg": 1.2,
            "pulse_u_anti": 1.0
        },
        "cancer": {
            "name": "Oncological Tumor Microenvironment",
            "P_label": "Healthy Somatic Parenchyma (P)",
            "D_label": "Cancer Stem Cells / Malignant Mass (D)",
            "I_cyt_label": "Immunosuppressive Stroma / Chronic Inflammation (I_cyt)",
            "I_reg_label": "Cytotoxic T / NK Surveillance (I_reg)",
            "u_label": "Epigenetic Differentiation & ICI Pulse",
            "K_tissue": 1.0,
            "K_cyt": 2.0,
            "K_reg": 2.0,
            "r_P": 0.45,
            "r_D": 0.28,
            "k_rec0": 0.04,
            "gamma_rec": 1.20,
            "alpha_I": 0.55,
            "delta_P": 0.025,
            "k_ind0": 0.008,
            "gamma_ind_I": 0.50,
            "gamma_ind_sigma": 0.40,
            "kappa_kill": 0.60,
            "delta_D": 0.06,
            "s_cyt": 0.020,
            "rho_cyt": 0.65,
            "mu_reg": 0.80,
            "d_cyt": 0.18,
            "gamma_anti": 0.85,
            "s_reg": 0.025,
            "rho_reg": 0.50,
            "mu_cyt": 0.80,
            "d_reg": 0.16,
            "sigma_ext": 0.55,
            "pulse_u_rec": 1.8,
            "pulse_u_reg": 1.5,
            "pulse_u_anti": 1.2
        },
        "sjogren": {
            "name": "Sjogren Syndrome (Autoimmune Exocrinopathy)",
            "P_label": "Functional Exocrine Gland Epithelium (P)",
            "D_label": "Atrophic / Inflamed Glandular Tissue (D)",
            "I_cyt_label": "Autoreactive B / T Cell Clones (I_cyt)",
            "I_reg_label": "Antigen-Specific Tolerance / Tregs (I_reg)",
            "u_label": "Tolerance Induction & B-Cell Depletion Pulse",
            "K_tissue": 1.0,
            "K_cyt": 2.0,
            "K_reg": 2.0,
            "r_P": 0.46,
            "r_D": 0.16,
            "k_rec0": 0.05,
            "gamma_rec": 1.10,
            "alpha_I": 0.55,
            "delta_P": 0.02,
            "k_ind0": 0.006,
            "gamma_ind_I": 0.48,
            "gamma_ind_sigma": 0.30,
            "kappa_kill": 0.55,
            "delta_D": 0.07,
            "s_cyt": 0.018,
            "rho_cyt": 0.62,
            "mu_reg": 0.85,
            "d_cyt": 0.19,
            "gamma_anti": 0.80,
            "s_reg": 0.030,
            "rho_reg": 0.52,
            "mu_cyt": 0.78,
            "d_reg": 0.15,
            "sigma_ext": 0.40,
            "pulse_u_rec": 1.4,
            "pulse_u_reg": 1.6,
            "pulse_u_anti": 1.3
        },
        "atopic": {
            "name": "Atopic Dermatitis (Epithelial Barrier Disease)",
            "P_label": "Intact Epidermal Barrier / Claudins (P)",
            "D_label": "Disrupted Stratum Corneum / Lesions (D)",
            "I_cyt_label": "Th2 Inflammatory Cascade (IL-4/13/TSLP) (I_cyt)",
            "I_reg_label": "Cutaneous Tregs & Tissue Healing (I_reg)",
            "u_label": "Intensive Anti-IL-4/13 & Barrier Repair Pulse",
            "K_tissue": 1.0,
            "K_cyt": 2.0,
            "K_reg": 2.0,
            "r_P": 0.55,
            "r_D": 0.22,
            "k_rec0": 0.06,
            "gamma_rec": 1.30,
            "alpha_I": 0.60,
            "delta_P": 0.03,
            "k_ind0": 0.008,
            "gamma_ind_I": 0.52,
            "gamma_ind_sigma": 0.38,
            "kappa_kill": 0.50,
            "delta_D": 0.08,
            "s_cyt": 0.022,
            "rho_cyt": 0.64,
            "mu_reg": 0.82,
            "d_cyt": 0.21,
            "gamma_anti": 0.90,
            "s_reg": 0.035,
            "rho_reg": 0.58,
            "mu_cyt": 0.72,
            "d_reg": 0.14,
            "sigma_ext": 0.50,
            "pulse_u_rec": 1.6,
            "pulse_u_reg": 1.3,
            "pulse_u_anti": 1.5
        }
    }

    def __init__(self, disease_type="diabetes", **custom_kwargs):
        self.disease_type = disease_type
        params = self.DISEASE_PRESETS.get(disease_type, self.DISEASE_PRESETS["diabetes"]).copy()
        params.update(custom_kwargs)
        for key, val in params.items():
            setattr(self, key, val)

    def vector_field(self, t, state, u_rec=0.0, u_reg=0.0, u_anti=0.0, sigma_ext=None):
        P, D, I_cyt, I_reg = state
        
        P = max(0.0, float(P))
        D = max(0.0, float(D))
        I_cyt = max(0.001, float(I_cyt))
        I_reg = max(0.001, float(I_reg))
        
        if sigma_ext is None:
            sigma_ext = self.sigma_ext
            
        space_tissue = max(0.0, 1.0 - (P + D) / self.K_tissue)
        space_cyt = max(0.0, 1.0 - I_cyt / self.K_cyt)
        space_reg = max(0.0, 1.0 - I_reg / self.K_reg)
        
        k_rec = self.k_rec0 + self.gamma_rec * u_rec
        
        k_ind = (
            self.k_ind0
            + self.gamma_ind_I * (I_cyt**2 / (0.20 + I_cyt**2))
            + self.gamma_ind_sigma * (sigma_ext / (0.50 + sigma_ext)) * (I_cyt / (0.30 + I_cyt))
        )
        
        dP_dt = (
            self.r_P * P * space_tissue
            + k_rec * D
            - self.alpha_I * (I_cyt / (0.30 + I_cyt)) * P
            - self.delta_P * P
        )
        
        dD_dt = (
            self.r_D * D * space_tissue
            + k_ind * P
            - k_rec * D
            - self.kappa_kill * (I_reg / (0.30 + I_reg)) * D
            - self.delta_D * D
        )
        
        stim_cyt = (
            self.s_cyt
            + self.rho_cyt * (D / (0.20 + D)) * I_cyt * space_cyt
        )
        supp_cyt = self.mu_reg * (I_reg**2 / (0.20 + I_reg**2)) * I_cyt
        clear_cyt = self.d_cyt * I_cyt + self.gamma_anti * u_anti * I_cyt
        dI_cyt_dt = stim_cyt - supp_cyt - clear_cyt
        
        stim_reg = (
            self.s_reg
            + self.rho_reg * (P / (0.30 + P)) * I_reg * space_reg
        )
        exhaust_reg = self.mu_cyt * (I_cyt**2 / (0.20 + I_cyt**2)) * I_reg
        clear_reg = self.d_reg * I_reg
        dI_reg_dt = stim_reg - exhaust_reg - clear_reg + u_reg
        
        return np.array([dP_dt, dD_dt, dI_cyt_dt, dI_reg_dt])

    def jacobian(self, state, u_rec=0.0, u_reg=0.0, u_anti=0.0, sigma_ext=None):
        eps = 1e-6
        base_f = self.vector_field(0.0, state, u_rec, u_reg, u_anti, sigma_ext)
        jac = np.zeros((4, 4))
        for j in range(4):
            perturbed = np.array(state, dtype=float)
            perturbed[j] += eps
            pert_f = self.vector_field(0.0, perturbed, u_rec, u_reg, u_anti, sigma_ext)
            jac[:, j] = (pert_f - base_f) / eps
        return jac
