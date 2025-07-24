"""
This code makes the augmented EECs using the model's data.
"""

import numpy as np
import pandas as pd
import ROOT
from ROOT import TFile, TH1D, TH2D, TLorentzVector, gROOT
import math
import os
import random
from spec_new import *
from const import *

# --- Global Settings ---
gROOT.SetBatch(True)
MAX_DR = 0.5
PT_MIN = 0.0
PT_MAX = 2.0
RANDOM_SEED = 22


# --- Helper Functions ---
def map_ang_mpitopi(x):
    """Maps an angle to the range [-pi, pi]. Works on scalars and numpy arrays."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def delta_r(eta1, phi1, eta2, phi2):
    """Calculates delta_r between two sets of particles. Works on scalars and numpy arrays."""
    dphi = map_ang_mpitopi(phi1 - phi2)
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

# --- Data Loading Functions ---
def load_hadron_particles(filepath, jet_eta=None, jet_phi=None):
    try:
        hadron_path = os.path.join(filepath, "hadron.dat")
        if not os.path.exists(hadron_path): return pd.DataFrame()
        
        hadron_df = pd.read_csv(hadron_path, sep='\s+', names=["pid", "px","py","pz","E"], engine='python')
        if hadron_df.empty: return pd.DataFrame()

        hadron_df["pt"] = np.sqrt(hadron_df["px"]**2 + hadron_df["py"]**2)
        hadron_df["p"] = np.sqrt(hadron_df["pt"]**2 + hadron_df["pz"]**2)
        hadron_df["phi"] = map_ang_mpitopi(np.arctan2(hadron_df["py"], hadron_df["px"]))
        
        p_minus_pz = hadron_df["p"] - hadron_df["pz"]
        valid_eta = p_minus_pz > 1e-9
        hadron_df["eta"] = np.nan
        hadron_df.loc[valid_eta, "eta"] = 0.5 * np.log((hadron_df.loc[valid_eta, "p"] + hadron_df.loc[valid_eta, "pz"]) / p_minus_pz[valid_eta])
        hadron_df.dropna(subset=['eta'], inplace=True)

        charged_pids = [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13]
        charged_particles_df = hadron_df[np.abs(hadron_df["pid"]).isin(charged_pids)].copy()

        if jet_eta is not None and jet_phi is not None and MAX_DR is not None:
            dr_to_jet = delta_r(charged_particles_df["eta"], charged_particles_df["phi"], jet_eta, jet_phi)
            charged_particles_df = charged_particles_df[dr_to_jet < MAX_DR]

        charged_particles_df["source"] = "signal_hadron"
        charged_particles_df["weight"] = 1.0
        charged_particles_df.rename(columns={'E': 'energy'}, inplace=True)
        final_cols = ["pt", "eta", "phi", "pid", "source", "weight", "energy"]
        return charged_particles_df[[col for col in final_cols if col in charged_particles_df.columns]]

    except Exception as e:
        print(f"Error processing hadron.dat in {filepath}: {e}")
        return pd.DataFrame()
def load_spec_particles(filepath, jet_eta=None, jet_phi=None, cone_radius=MAX_DR, pt_min=PT_MIN, pt_max=PT_MAX):
    """Load spec particles from dNdEtaPtdPtdPhi_Charged.dat with correct weights and pt filter, then apply cone cut if requested."""
    try:
        event_charged_dat = os.path.join(filepath, "dNdEtaPtdPtdPhi_Charged.dat")
        if not os.path.exists(event_charged_dat):
            print(f"dNdEtaPtdPtdPhi_Charged.dat not found in {filepath}")
            return pd.DataFrame()

        spec_data = np.loadtxt(event_charged_dat).reshape(NY, NPT, NPHI) / (HBARC**3.0)

        eta_width = np.empty_like(Y)
        eta_width[1:-1] = (Y[2:] - Y[:-2]) / 2.0
        eta_width[0] = (Y[1] - Y[0]) / 2.0
        eta_width[-1] = (Y[-1] - Y[-2]) / 2.0
        pt_width = gala15w * INVP
        phi_width = np.pi * np.concatenate((gaulew48, gaulew48[::-1]))

        # Make 3D grids for all variables and widths
        eta_grid, pt_grid, phi_grid = np.meshgrid(Y, PT, PHI, indexing='ij')
        eta_width_grid, pt_width_grid, phi_width_grid = np.meshgrid(eta_width, pt_width, phi_width, indexing='ij')

        # dN = dN/dEta/ptdPt/dPhi * dEta * dPt * dPhi * pt
        particle_counts = spec_data * eta_width_grid * pt_width_grid * phi_width_grid * pt_grid

        # Flatten everything for DataFrame
        eta_flat = eta_grid.flatten()
        pt_flat = pt_grid.flatten()
        phi_flat = map_ang_mpitopi(phi_grid.flatten())
        particle_counts_flat = particle_counts.flatten()
        eta_width_flat = eta_width_grid.flatten()
        phi_width_flat = phi_width_grid.flatten()

        # Mask for nonzero and pt range
        mask = (particle_counts_flat > 0) & (pt_flat >= pt_min) & (pt_flat <= pt_max)
        if not np.any(mask):
            print("No spec particles found in pt range.")
            return pd.DataFrame()

        particles_df = pd.DataFrame({
            "pt": pt_flat[mask],
            "eta": eta_flat[mask],
            "phi": phi_flat[mask],
            "weight": particle_counts_flat[mask],
            "eta_width": eta_width_flat[mask],
            "phi_width": phi_width_flat[mask]
        })

        # Apply cone cut if requested
        if jet_eta is not None and jet_phi is not None and cone_radius is not None:
            dr_to_jet = delta_r(particles_df["eta"], particles_df["phi"], jet_eta, jet_phi)
            particles_df = particles_df[dr_to_jet < cone_radius]

        return particles_df

    except Exception as e:
        print(f"Error processing spec data: {e}")
        return pd.DataFrame()
        
def load_newconrecom_particles(filepath, jet_eta, jet_phi):
    try:
        newcon_path = os.path.join(filepath, "NewConRecom.dat")
        if not os.path.exists(newcon_path): return pd.DataFrame()

        newcon_df = pd.read_csv(newcon_path, sep='\s+', names=["pid","px","py","pz","E","mass","x","y","z","t"], engine='python')
        if newcon_df.empty: return pd.DataFrame()

        newcon_df["pt"] = np.sqrt(newcon_df["px"]**2 + newcon_df["py"]**2)
        newcon_df["p"] = np.sqrt(newcon_df["pt"]**2 + newcon_df["pz"]**2)
        newcon_df["phi"] = map_ang_mpitopi(np.arctan2(newcon_df["py"], newcon_df["px"]))
        
        p_minus_pz = newcon_df["p"] - newcon_df["pz"]
        valid_eta = p_minus_pz > 1e-9
        newcon_df["eta"] = np.nan
        newcon_df.loc[valid_eta, "eta"] = 0.5 * np.log((newcon_df.loc[valid_eta, "p"] + newcon_df.loc[valid_eta, "pz"]) / p_minus_pz[valid_eta])
        newcon_df.dropna(subset=['eta'], inplace=True)
        
        charged_pids = [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13]
        charged_particles_df = newcon_df[np.abs(newcon_df["pid"]).isin(charged_pids)].copy()

        dr_to_jet = delta_r(charged_particles_df["eta"], charged_particles_df["phi"], jet_eta, jet_phi)
        final_df = charged_particles_df[dr_to_jet < MAX_DR].copy()

        final_df["source"] = "newcon"
        final_df["weight"] = 1.0
        return final_df[["pt", "eta", "phi", "source", "weight"]]

    except Exception as e:
        print(f"Error processing NewConRecom.dat in {filepath}: {e}")
        return pd.DataFrame()

def select_equal_particles(df1, df2):
    """If DFs have different lengths, randomly downsample the larger one."""
    if df1.empty or df2.empty: return df1.copy(), df2.copy()
    n1, n2 = len(df1), len(df2)
    if n1 == n2:
        return df1.copy(), df2.copy()
    elif n1 > n2:
        return df1.sample(n=n2, random_state=RANDOM_SEED).copy(), df2.copy()
    else:
        return df1.copy(), df2.sample(n=n1, random_state=RANDOM_SEED).copy()

def rotate_particles_to_match_axes(particles_df, old_axis_eta, old_axis_phi, new_axis_eta, new_axis_phi):
    expected_cols = ["pt", "eta", "phi", "pid", "source", "weight", "energy"]
    if particles_df.empty:
        return pd.DataFrame(columns=expected_cols)
    deta = new_axis_eta - old_axis_eta
    dphi = map_ang_mpitopi(new_axis_phi - old_axis_phi)
    rotated_df = particles_df.copy()
    rotated_df["eta"] += deta
    rotated_df["phi"] = map_ang_mpitopi(rotated_df["phi"] + dphi)
    return rotated_df

def find_matching_pp_event(pb_gamma_pt, available_pp_events):
    tolerance = 0.05 * pb_gamma_pt
    for i, (pp_idx, pp_gamma_pt_val) in enumerate(available_pp_events):
        if abs(pp_gamma_pt_val - pb_gamma_pt) < tolerance:
            return i, pp_idx, pp_gamma_pt_val
    return -1, -1, None

def apply_ratio_weights_df(df, ratio_hist, jet_eta, jet_phi):
    """Applies ratio weights to a particle DataFrame using a ROOT histogram."""
    if df.empty or not ratio_hist: return df, 0, 0
    
    deta = df["eta"].to_numpy() - jet_eta
    dphi = map_ang_mpitopi(df["phi"].to_numpy() - jet_phi)
    
    # Get weights from histogram
    ratio_weights = np.zeros(len(df))
    for i, (de, dp) in enumerate(zip(deta, dphi)):
        bin_x = ratio_hist.GetXaxis().FindBin(de)
        bin_y = ratio_hist.GetYaxis().FindBin(dp)
        weight = ratio_hist.GetBinContent(bin_x, bin_y)
        ratio_weights[i] = weight if weight > 0 else 1.0
    
    modified_df = df.copy()
    old_weights = modified_df["weight"].sum()
    modified_df["weight"] *= ratio_weights
    new_weights = modified_df["weight"].sum()
    
    return modified_df, old_weights, new_weights

def particles_to_arrays(particles_df):
    return (
        particles_df["eta"].to_numpy(dtype=np.float64),
        particles_df["phi"].to_numpy(dtype=np.float64),
        particles_df["pt"].to_numpy(dtype=np.float64),
        particles_df.get("weight", pd.Series(np.ones(len(particles_df)), index=particles_df.index)).to_numpy(dtype=np.float64)
    )

def fill_histogram_pairs(hist, eta, phi, pt, weight, jet_pt, n_exp, dr_max):
    n = len(eta)
    if n < 2:
        return
    idx_i, idx_j = np.triu_indices(n, k=1)
    deta = eta[idx_i] - eta[idx_j]
    dphi = map_ang_mpitopi(phi[idx_i] - phi[idx_j])
    dr = np.sqrt(deta**2 + dphi**2)
    mask = dr < dr_max
    dr = dr[mask]
    pt_i = pt[idx_i][mask]
    pt_j = pt[idx_j][mask]
    w_i = weight[idx_i][mask]
    w_j = weight[idx_j][mask]
    pair_weight = ((pt_i * pt_j) / (jet_pt**2))**n_exp * w_i * w_j
    hist.FillN(len(dr), dr, pair_weight)

def fill_histogram_cross_pairs(hist, eta1, phi1, pt1, weight1, eta2, phi2, pt2, weight2, jet_pt, n_exp, dr_max):
    n1 = len(eta1)
    n2 = len(eta2)
    if n1 == 0 or n2 == 0:
        return
    deta = eta1[:, None] - eta2[None, :]
    dphi = map_ang_mpitopi(phi1[:, None] - phi2[None, :])
    dr = np.sqrt(deta**2 + dphi**2)
    mask = dr < dr_max
    pt1_2d = pt1[:, None]
    pt2_2d = pt2[None, :]
    w1_2d = weight1[:, None]
    w2_2d = weight2[None, :]
    pair_weight = ((pt1_2d * pt2_2d) / (jet_pt**2))**n_exp * w1_2d * w2_2d
    dr_flat = dr[mask]
    pair_weight_flat = pair_weight[mask]
    hist.FillN(len(dr_flat), dr_flat, pair_weight_flat)

# --- Main Analysis Function ---
def energy_energy_correlator_pbpbpp(
    pbpb_datapath, pp_datapath, hydro_datapath, outfilename, 
    start_file, end_file, pp_start, pp_end
):
    # --- STEP 1: INITIALIZATION ---
    print("Pre-scanning PP events...")
    available_pp_events = []
    for pp_idx in range(pp_start, pp_end):
        gamma_file = os.path.join(pp_datapath, f"event{pp_idx}", "gamma.dat")
        if not os.path.exists(gamma_file): continue
        try:
            gamma_df = pd.read_csv(gamma_file, sep='\s+', names=["px","py","pz","E"], engine='python')
            if not gamma_df.empty: available_pp_events.append((pp_idx, np.sqrt(gamma_df["px"][0]**2 + gamma_df["py"][0]**2)))
        except Exception: continue
    print(f"Found {len(available_pp_events)} valid PP events.")
    
    # --- HISTOGRAM SETUP ---
    gamma_pt_bins = [(0, 500)]
    p_ch_T_cuts = [0.0, 1.0]
    n_exponents = [0, 1]
    delta_r_min, delta_r_max, n_bins = 0.01, 1, 40
    log_min, log_max = np.log10(delta_r_min), np.log10(delta_r_max)
    delta_r_bins = np.logspace(log_min, log_max, n_bins + 1)
    
    n_eta_bins, n_phi_bins = 24, 20
    eta_min, eta_max = -6.0, 6.0
    phi_min, phi_max = -np.pi, np.pi

    hists = {}
    d2_hists = {}
    
    hist_definitions = {
        "pbpb_signal": "PbPb EEC", "pbpb_sm1": "PbPb S-M1", "pbpb_m1m1": "PbPb M1-M1", "pbpb_m1m2": "PbPb M1-M2",
        "pbpb_spec_m1": "PbPb Spec-M1", "pbpb_newcon_cone": "PbPb NewCon Cone",
        "pp_signal": "PP EEC", "pp_sm1": "PP S-M1", "pp_m1m1": "PP M1-M1", "pp_m1m2": "PP M1-M2",
        "pp_hadron_cone": "PP Hadron Cone",
    }
    
    for prefix, title_base in hist_definitions.items():
        hists[prefix] = {}
        for n_exp in n_exponents:
            for pt_min, pt_max in gamma_pt_bins:
                for p_cut in p_ch_T_cuts:
                    key = (n_exp, pt_min, pt_max, p_cut)
                    name = f"{prefix}_eec_n{n_exp}_jetpt{pt_min}to{pt_max}_pch{p_cut}"
                    title = f"{title_base} n={n_exp}, pT {pt_min}-{pt_max}, p_ch>{p_cut};R_L;EEC"
                    hists[prefix][key] = ROOT.TH1D(name, title, n_bins, delta_r_bins)
                    hists[prefix][key].Sumw2()
    
    # Initialize 2D histograms and jet counts
    pbpb_jet_counts = ROOT.TH1D("pbpb_jet_counts", "PbPb Jet Counts", len(gamma_pt_bins), 0, len(gamma_pt_bins))
    pp_jet_counts = ROOT.TH1D("pp_jet_counts", "PP Jet Counts", len(gamma_pt_bins), 0, len(gamma_pt_bins))
    for i, (pt_min, pt_max) in enumerate(gamma_pt_bins):
        label = f"{pt_min}-{pt_max} GeV"
        pbpb_jet_counts.GetXaxis().SetBinLabel(i+1, label)
        pp_jet_counts.GetXaxis().SetBinLabel(i+1, label)

    # Create 2D histograms for ratio calculation (no pt cut)
    for pt_min, pt_max in gamma_pt_bins:
        key = (pt_min, pt_max)
        d2_hists[key] = {
            'pbpb_spec': ROOT.TH2D(f"pbpb_spec_2d_{pt_min}_{pt_max}", "PbPb Spec;#Delta#eta;#Delta#phi", 
                                   n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max),
            'hydro': ROOT.TH2D(f"hydro_2d_{pt_min}_{pt_max}", "Hydro;#Delta#eta;#Delta#phi", 
                               n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max),
            'ratio': ROOT.TH2D(f"ratio_2d_{pt_min}_{pt_max}", "Ratio (PbPb/Hydro);#Delta#eta;#Delta#phi", 
                               n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max),
        }
        d2_hists[key]['pbpb_spec'].Sumw2()
        d2_hists[key]['hydro'].Sumw2()

    # --- PHASE 1: Accumulate data for Ratio Map (no pt cut) ---
    print("\n--- Starting Phase 1: Accumulating data for ratio map ---")
    for evt_idx in range(start_file, end_file):
        if evt_idx % 1 == 0:
            print(f"Phase 1: Processing event {evt_idx}")
        
        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        try:
            hadjet_df = pd.read_csv(os.path.join(pbpb_path, "hadjet.dat"), sep='\s+', 
                                   names=["jet_no","px","py","pz","E","eta"], engine='python')
            gamma_df = pd.read_csv(os.path.join(pbpb_path, "gamma.dat"), sep='\s+', 
                                  names=["px","py","pz","E"], engine='python')
            if hadjet_df.empty or gamma_df.empty: continue
        except Exception: continue
        
        jet_row = hadjet_df.iloc[0]
        jet_pt = np.sqrt(jet_row["px"]**2 + jet_row["py"]**2)
        jet_eta = jet_row["eta"]
        jet_phi = map_ang_mpitopi(np.arctan2(jet_row["py"], jet_row["px"]))
        pb_gamma_pt = np.sqrt(gamma_df["px"].iloc[0]**2 + gamma_df["py"].iloc[0]**2)
        
        active_pt_bins = [(pt_min, pt_max) for pt_min, pt_max in gamma_pt_bins if pt_min <= pb_gamma_pt < pt_max]
        if not active_pt_bins: continue

        # Load spec particles without cone cut for ratio calculation
        pbpb_spec_df = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=PT_MIN, pt_max=PT_MAX)
        hydro_path = os.path.join(hydro_datapath, f"event{evt_idx % 400}")
        hydro_df = load_spec_particles(hydro_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=PT_MIN, pt_max=PT_MAX)

        for pt_min_bin, pt_max_bin in active_pt_bins:
            key = (pt_min_bin, pt_max_bin)
            
            # Fill PbPb spec 2D hist (no pt cut)
            if not pbpb_spec_df.empty:
                deta = (pbpb_spec_df["eta"] - jet_eta).to_numpy()
                dphi = map_ang_mpitopi(pbpb_spec_df["phi"] - jet_phi).to_numpy()
                w = pbpb_spec_df["weight"].to_numpy()
                d2_hists[key]['pbpb_spec'].FillN(len(pbpb_spec_df), deta, dphi, w)
            
            # Fill Hydro 2D hist (no pt cut)
            if not hydro_df.empty:
                deta = (hydro_df["eta"] - jet_eta).to_numpy()
                dphi = map_ang_mpitopi(hydro_df["phi"] - jet_phi).to_numpy()
                w = hydro_df["weight"].to_numpy()
                d2_hists[key]['hydro'].FillN(len(hydro_df), deta, dphi, w)

    # --- PHASE 1.5: Create Ratio Maps ---
    print("\n--- Calculating ratio histograms ---")
    for key, h_dict in d2_hists.items():
        h_dict['ratio'].Divide(h_dict['pbpb_spec'], h_dict['hydro'])

    # --- PHASE 2: Perform EEC Calculation ---
    print("\n--- Starting Phase 2: EEC Calculation with ratio weights ---")
    processed_pb_events, processed_pp_events = 0, 0
    available_pp_events_for_loop2 = list(available_pp_events)

    for evt_idx in range(start_file, end_file):
        if evt_idx % 1 == 0:
            print(f"Phase 2: Processing event pair for PbPb event {evt_idx}")
        
        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        try:
            hadjet_df = pd.read_csv(os.path.join(pbpb_path, "hadjet.dat"), sep='\s+', 
                                   names=["jet_no","px","py","pz","E","eta"], engine='python')
            gamma_df = pd.read_csv(os.path.join(pbpb_path, "gamma.dat"), sep='\s+', 
                                  names=["px","py","pz","E"], engine='python')
            if hadjet_df.empty or gamma_df.empty: continue
        except Exception: continue
        
        jet_row = hadjet_df.iloc[0]
        jet_pt = np.sqrt(jet_row["px"]**2 + jet_row["py"]**2)
        jet_eta = jet_row["eta"]
        jet_phi = map_ang_mpitopi(np.arctan2(jet_row["py"], jet_row["px"]))
        pb_gamma_pt = np.sqrt(gamma_df["px"].iloc[0]**2 + gamma_df["py"].iloc[0]**2)
        
        active_pt_bins = [(pt_min, pt_max) for pt_min, pt_max in gamma_pt_bins if pt_min <= pb_gamma_pt < pt_max]
        if not active_pt_bins: continue

        match_idx, matched_pp_idx, _ = find_matching_pp_event(pb_gamma_pt, available_pp_events_for_loop2)
        if match_idx == -1: continue
        available_pp_events_for_loop2.pop(match_idx)
        
        pp_path = os.path.join(pp_datapath, f"event{matched_pp_idx}")
        try:
            pp_hadjet_df = pd.read_csv(os.path.join(pp_path, "hadjet.dat"), sep='\s+', 
                                      names=["jno","px","py","pz","E","eta"], engine='python')
            pp_jet_row = pp_hadjet_df.iloc[0]
            pp_jet_pt = np.sqrt(pp_jet_row["px"]**2 + pp_jet_row["py"]**2)
            pp_jet_eta = pp_jet_row["eta"]
            pp_jet_phi = map_ang_mpitopi(np.arctan2(pp_jet_row["py"], pp_jet_row["px"]))
        except Exception: continue

        # Load particle sets with cone cuts
        pbpb_newcon_df = load_newconrecom_particles(pbpb_path, jet_eta, jet_phi)
        pp_hadron_df = load_hadron_particles(pp_path, pp_jet_eta, pp_jet_phi)
        #pbpb_newcon_df, pp_hadron_df = select_equal_particles(pbpb_newcon_df, pp_hadron_df)
        pbpb_newcon_df, pp_hadron_df = pbpb_newcon_df, pp_hadron_df
        pbpb_spec_df = load_spec_particles(pbpb_path, jet_eta, jet_phi, cone_radius=MAX_DR, pt_min=PT_MIN, pt_max=PT_MAX)
        pbpb_signal_df = pd.concat([pbpb_newcon_df, pbpb_spec_df], ignore_index=True)

        m1_idx, m2_idx = random.sample(list(set(range(400)) - {evt_idx % 400}), 2)
        hydro_m1_df = load_spec_particles(os.path.join(hydro_datapath, f"event{m1_idx}"), jet_eta, jet_phi, cone_radius=MAX_DR, pt_min=PT_MIN, pt_max=PT_MAX)
        hydro_m2_df = load_spec_particles(os.path.join(hydro_datapath, f"event{m2_idx}"), jet_eta, jet_phi, cone_radius=MAX_DR, pt_min=PT_MIN, pt_max=PT_MAX)

        rotated_hadron_df = rotate_particles_to_match_axes(pp_hadron_df, pp_jet_eta, pp_jet_phi, jet_eta, jet_phi)
        if rotated_hadron_df.empty or "pt" not in rotated_hadron_df.columns:
            print(f"Skipping event {evt_idx}: rotated_hadron_df is empty or missing 'pt' column.")
            continue
        pp_ue_hydro_df = load_spec_particles(os.path.join(hydro_datapath, f"event{evt_idx % 400}"), jet_eta, jet_phi, cone_radius=MAX_DR, pt_min=PT_MIN, pt_max=PT_MAX)
        pp_signal_df = pd.concat([rotated_hadron_df, pp_ue_hydro_df], ignore_index=True)

        # Get ratio histogram for this pt bin
        ratio_hist = d2_hists[active_pt_bins[0]]['ratio']

        for p_cut in p_ch_T_cuts:
            # Apply weights to hydro backgrounds
            hydro_m1_w_df, _, _ = apply_ratio_weights_df(hydro_m1_df, ratio_hist, jet_eta, jet_phi)
            print("applied ratio weights to m1")
            hydro_m2_w_df, _, _ = apply_ratio_weights_df(hydro_m2_df, ratio_hist, jet_eta, jet_phi)
            print("applied ratio weights to m2")
            # Filter all dataframes by pT cut
            pbpb_signal_f = pbpb_signal_df[pbpb_signal_df["pt"] >= p_cut]
            hydro_m1_f = hydro_m1_w_df[hydro_m1_w_df["pt"] >= p_cut]
            hydro_m2_f = hydro_m2_w_df[hydro_m2_w_df["pt"] >= p_cut]
            pbpb_newcon_f = pbpb_newcon_df[pbpb_newcon_df["pt"] >= p_cut]
            pbpb_spec_f = pbpb_spec_df[pbpb_spec_df["pt"] >= p_cut]
            
            pp_signal_f = pp_signal_df[pp_signal_df["pt"] >= p_cut]
            rotated_hadron_f = rotated_hadron_df[rotated_hadron_df["pt"] >= p_cut]
            pp_m1_f = hydro_m1_f  # Use same weighted hydro backgrounds
            pp_m2_f = hydro_m2_f

            # --- Fill PbPb Histograms ---
            if len(pbpb_signal_f) >= 2:
                eta, phi, pt, weight = particles_to_arrays(pbpb_signal_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_pairs(hists["pbpb_signal"][key], eta, phi, pt, weight, jet_pt, n_exp, delta_r_max)

            if len(pbpb_signal_f) > 0 and len(hydro_m1_f) > 0:
                eta1, phi1, pt1, weight1 = particles_to_arrays(pbpb_signal_f)
                eta2, phi2, pt2, weight2 = particles_to_arrays(hydro_m1_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_cross_pairs(hists["pbpb_sm1"][key], eta1, phi1, pt1, weight1, 
                                                 eta2, phi2, pt2, weight2, jet_pt, n_exp, delta_r_max)

            if len(hydro_m1_f) >= 2:
                eta, phi, pt, weight = particles_to_arrays(hydro_m1_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_pairs(hists["pbpb_m1m1"][key], eta, phi, pt, weight, jet_pt, n_exp, delta_r_max)

            if len(hydro_m1_f) > 0 and len(hydro_m2_f) > 0:
                eta1, phi1, pt1, weight1 = particles_to_arrays(hydro_m1_f)
                eta2, phi2, pt2, weight2 = particles_to_arrays(hydro_m2_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_cross_pairs(hists["pbpb_m1m2"][key], eta1, phi1, pt1, weight1, 
                                                 eta2, phi2, pt2, weight2, jet_pt, n_exp, delta_r_max)

            if len(pbpb_spec_f) > 0 and len(hydro_m1_f) > 0:
                eta1, phi1, pt1, weight1 = particles_to_arrays(pbpb_spec_f)
                eta2, phi2, pt2, weight2 = particles_to_arrays(hydro_m1_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_cross_pairs(hists["pbpb_spec_m1"][key], eta1, phi1, pt1, weight1, 
                                                 eta2, phi2, pt2, weight2, jet_pt, n_exp, delta_r_max)

            if len(pbpb_newcon_f) >= 2:
                eta, phi, pt, weight = particles_to_arrays(pbpb_newcon_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_pairs(hists["pbpb_newcon_cone"][key], eta, phi, pt, weight, jet_pt, n_exp, delta_r_max)
            
            # --- Fill PP Histograms ---
            if len(pp_signal_f) >= 2:
                eta, phi, pt, weight = particles_to_arrays(pp_signal_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_pairs(hists["pp_signal"][key], eta, phi, pt, weight, pp_jet_pt, n_exp, delta_r_max)

            if len(pp_signal_f) > 0 and len(pp_m1_f) > 0:
                eta1, phi1, pt1, weight1 = particles_to_arrays(pp_signal_f)
                eta2, phi2, pt2, weight2 = particles_to_arrays(pp_m1_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_cross_pairs(hists["pp_sm1"][key], eta1, phi1, pt1, weight1, 
                                                 eta2, phi2, pt2, weight2, pp_jet_pt, n_exp, delta_r_max)

            if len(pp_m1_f) >= 2:
                eta, phi, pt, weight = particles_to_arrays(pp_m1_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_pairs(hists["pp_m1m1"][key], eta, phi, pt, weight, pp_jet_pt, n_exp, delta_r_max)

            if len(pp_m1_f) > 0 and len(pp_m2_f) > 0:
                eta1, phi1, pt1, weight1 = particles_to_arrays(pp_m1_f)
                eta2, phi2, pt2, weight2 = particles_to_arrays(pp_m2_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_cross_pairs(hists["pp_m1m2"][key], eta1, phi1, pt1, weight1, 
                                                 eta2, phi2, pt2, weight2, pp_jet_pt, n_exp, delta_r_max)

            if len(rotated_hadron_f) >= 2:
                eta, phi, pt, weight = particles_to_arrays(rotated_hadron_f)
                for n_exp in n_exponents:
                    for pt_bin in active_pt_bins:
                        key = (n_exp, pt_bin[0], pt_bin[1], p_cut)
                        fill_histogram_pairs(hists["pp_hadron_cone"][key], eta, phi, pt, weight, pp_jet_pt, n_exp, delta_r_max)
        
        # Update jet counts
        for i, (pt_min, pt_max) in enumerate(gamma_pt_bins):
            if pt_min <= pb_gamma_pt < pt_max:
                pbpb_jet_counts.Fill(i)
                pp_jet_counts.Fill(i)
        
        processed_pb_events += 1
        processed_pp_events += 1

    print(f"\nCompleted processing. Processed {processed_pb_events} PbPb and {processed_pp_events} PP events.")
    # --- Normalize 2D histograms by number of events and bin area, then subtract hydro ---

    if processed_pb_events > 0:
        # Normalize by number of events
        for key, h_dict in d2_hists.items():
            h_dict['pbpb_spec'].Scale(1.0 / processed_pb_events)
            h_dict['hydro'].Scale(1.0 / processed_pb_events)
            # Do not scale ratio, it's already a ratio

        # Normalize by bin area
        deta_bins, deta_min, deta_max = 24, -6.0, 6.0
        dphi_bins, dphi_min, dphi_max = 20, -np.pi, np.pi
        deta_bin_width = (deta_max - deta_min) / deta_bins
        dphi_bin_width = (dphi_max - dphi_min) / dphi_bins
        bin_area = deta_bin_width * dphi_bin_width

        for key, h_dict in d2_hists.items():
            h_dict['pbpb_spec'].Scale(1.0 / bin_area)
            h_dict['hydro'].Scale(1.0 / bin_area)
            # Do not scale ratio

        # Subtract hydro from pbpb_spec and store
        for key, h_dict in d2_hists.items():
            h_sub = h_dict['pbpb_spec'].Clone(f"pbpb_spec_minus_hydro_{key[0]}_{key[1]}")
            h_sub.SetTitle("PbPb Spec - Hydro;#Delta#eta;#Delta#phi")
            h_sub.Add(h_dict['hydro'], -1.0)
            h_dict['pbpb_spec_minus_hydro'] = h_sub
    
    outfile = ROOT.TFile(outfilename, "RECREATE")

    print("Saving EEC histograms...")
    for prefix in hists:
        path_parts = prefix.split('_')
        top_dir_name_internal = path_parts[0]
        output_top_dir_name = {'pbpb': 'PbPb', 'pp': 'PP'}.get(top_dir_name_internal, top_dir_name_internal)
        top_dir = outfile.Get(output_top_dir_name) or outfile.mkdir(output_top_dir_name)
        final_dir = top_dir
        if len(path_parts) > 1:
            sub_dir_name = "_".join(path_parts[1:])
            sub_dir = top_dir.Get(sub_dir_name) or top_dir.mkdir(sub_dir_name)
            final_dir = sub_dir
        final_dir.cd()
        for hist in hists[prefix].values():
            hist.Write()
    outfile.cd()
    
    # Save 2D histograms
    pbpb_2d_dir = outfile.mkdir("PbPb/spec_2d")
    hydro_2d_dir = outfile.mkdir("Hydro/hydro_2d") 
    ratio_2d_dir = outfile.mkdir("Ratios/ratio_2d")
    sub_2d_dir = outfile.mkdir("PbPb/spec_minus_hydro_2d")
    for key, h_dict in d2_hists.items():
        pbpb_2d_dir.cd()
        h_dict['pbpb_spec'].Write()
        hydro_2d_dir.cd()
        h_dict['hydro'].Write()
        ratio_2d_dir.cd()
        h_dict['ratio'].Write()
        sub_2d_dir.cd()
        h_dict['pbpb_spec_minus_hydro'].Write()

    # Save jet counts
    counts_dir = outfile.mkdir("JetCounts")
    counts_dir.cd()
    pbpb_jet_counts.Write()
    pp_jet_counts.Write()

    outfile.Close()
    print(f"Histograms successfully written to {outfilename}")

if __name__ == '__main__':
    pbpb_datapath  = "/home/CoLBT"
    pp_datapath    = "/home/PP"
    hydro_datapath = "/home/Hydrobackground"
    outfilename    = "/home/Energy_energy_correlators_scripts/Ap_comparison_macros/EEC_embed_augmented.root"
    energy_energy_correlator_pbpbpp(
        pbpb_datapath, pp_datapath, hydro_datapath, outfilename, 
        start_file=0, end_file=999, pp_start=9, pp_end=1999
    )
