"""after generating the minimum bias events, use this to get the augmented EEC"""

import numpy as np
import pandas as pd
import ROOT
from ROOT import TFile, TH1D, TLorentzVector, gROOT
import math
import os
import random
from spec_new import *
from const import *


gROOT.SetBatch(True)

MAX_DR = 0.5
RANDOM_SEED = 22
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED) 



def map_ang_mpitopi(x):
    """Maps an angle to the range [-pi, pi]. Works on scalars and numpy arrays."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def delta_r(eta1, phi1, eta2, phi2):
    """Calculates delta_r between two sets of particles. Works on scalars and numpy arrays."""
    dphi = map_ang_mpitopi(phi1 - phi2)
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)


def load_hadron_particles(filepath, jet_eta=None, jet_phi=None):
    """Loads charged hadrons from hadron.dat, vectorizing all calculations and filtering."""
    try:
        hadron_path = os.path.join(filepath, "hadron.dat")
        if not os.path.exists(hadron_path):
            return pd.DataFrame()

        hadron_df = pd.read_csv(hadron_path, sep='\s+', names=["pid", "px","py","pz","E"], engine='python')
        if hadron_df.empty:
            return pd.DataFrame()

        hadron_df["pt"] = np.sqrt(hadron_df["px"]**2 + hadron_df["py"]**2)
        hadron_df["p"] = np.sqrt(hadron_df["pt"]**2 + hadron_df["pz"]**2)
        hadron_df["phi"] = np.arctan2(hadron_df["py"], hadron_df["px"])

        p_minus_pz = hadron_df["p"] - hadron_df["pz"]
        valid_eta = p_minus_pz > 1e-9
        hadron_df["eta"] = np.nan
        hadron_df.loc[valid_eta, "eta"] = 0.5 * np.log((hadron_df.loc[valid_eta, "p"] + hadron_df.loc[valid_eta, "pz"]) / p_minus_pz[valid_eta])
        hadron_df.dropna(subset=['eta'], inplace=True)

        hadron_df["phi"] = map_ang_mpitopi(hadron_df["phi"])
        print(f"Found {len(hadron_df)} total particles in hadron.dat")

        charged_pids = [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13]
        charged_particles_df = hadron_df[np.abs(hadron_df["pid"]).isin(charged_pids)].copy()
        print(f"Selected {len(charged_particles_df)} charged particles")

        if jet_eta is not None and jet_phi is not None:
            dr_to_jet = delta_r(charged_particles_df["eta"], charged_particles_df["phi"], jet_eta, jet_phi)
            charged_particles_df = charged_particles_df[dr_to_jet < MAX_DR]

        charged_particles_df["source"] = "signal_hadron"
        charged_particles_df["weight"] = 1.0
        charged_particles_df.rename(columns={'E': 'energy'}, inplace=True)
        final_cols = ["pt", "eta", "phi", "pid", "source", "weight", "energy"]

        particles_in_cone = charged_particles_df[[col for col in final_cols if col in charged_particles_df.columns]]
        print(f"Added {len(particles_in_cone)} charged particles from hadron.dat within dR<{MAX_DR}")
        return particles_in_cone

    except Exception as e:
        print(f"Error processing hadron.dat: {e}")
        return pd.DataFrame()
def load_spec_particles(filepath, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=500.0, event_plane=None):
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

        eta_grid, pt_grid, phi_grid = np.meshgrid(Y, PT, PHI, indexing='ij')
        eta_width_grid, pt_width_grid, phi_width_grid = np.meshgrid(eta_width, pt_width, phi_width, indexing='ij')

        particle_counts = spec_data * eta_width_grid * pt_width_grid * phi_width_grid * pt_grid

        eta_flat = eta_grid.flatten()
        pt_flat = pt_grid.flatten()
        phi_flat = map_ang_mpitopi(phi_grid.flatten())
        particle_counts_flat = particle_counts.flatten()
        eta_width_flat = eta_width_grid.flatten()
        phi_width_flat = phi_width_grid.flatten()

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
            "phi_width": phi_width_flat[mask],
            "source": "spec"
        })

        if event_plane is not None:
            particles_df["phi"] = map_ang_mpitopi(particles_df["phi"] - event_plane)

        if jet_eta is not None and jet_phi is not None and cone_radius is not None:
            dr_to_jet = delta_r(particles_df["eta"], particles_df["phi"], jet_eta, jet_phi)
            particles_df = particles_df[dr_to_jet < cone_radius]

        total_particles = particles_df['weight'].sum()
        cone_msg = f"within dR<{cone_radius}" if cone_radius is not None else "(no cone limit)"
        print(f"Added {total_particles:.2f} particles (from {len(particles_df)} bins) from spec data {cone_msg}")
        return particles_df

    except Exception as e:
        print(f"Error processing spec data: {e}")
        return pd.DataFrame()
def load_newconrecom_particles(filepath, jet_eta=None, jet_phi=None, cone_radius=None, event_plane=None):
    """Loads charged particles from NewConRecom.dat, vectorized."""
    try:
        newcon_path = os.path.join(filepath, "NewConRecom.dat")
        if not os.path.exists(newcon_path):
            return pd.DataFrame()

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
        print(f"Found {len(charged_particles_df)} charged particles in NewConRecom.dat")
        
        if event_plane is not None:
            charged_particles_df["phi"] = map_ang_mpitopi(charged_particles_df["phi"] - event_plane)
        
        if jet_eta is not None and jet_phi is not None and cone_radius is not None:
            dr_to_jet = delta_r(charged_particles_df["eta"], charged_particles_df["phi"], jet_eta, jet_phi)
            final_df = charged_particles_df[dr_to_jet < cone_radius].copy()
        else:
            final_df = charged_particles_df.copy() # No cone filter

        final_df["source"] = "newcon"
        final_df["weight"] = 1.0
        final_cols = ["pt", "eta", "phi", "source", "weight"]
        
        cone_msg = f"within dR<{cone_radius}" if cone_radius is not None else "(no cone limit)"
        print(f"Added {len(final_df)} particles from NewConRecom.dat {cone_msg}")
        return final_df[final_cols]

    except Exception as e:
        print(f"Error processing NewConRecom.dat: {e}")
        return pd.DataFrame()
    
def load_colbt_mixed_particles(event_idx, particle_type, jet_eta=None, jet_phi=None, cone_radius=None, event_plane=None, pt_min=None, pt_max=None):
    """Loads particles from CoLBT_Minimum_Bias for m1, m2, or mixed events."""
    try:
        base_path = os.path.join("./CoLBT_Minimum_Bias", f"event{event_idx}", particle_type)
        
        spec_df = load_spec_particles(base_path, jet_eta=None, jet_phi=None, cone_radius=None, 
                                     pt_min=pt_min if pt_min is not None else 0.0, 
                                     pt_max=pt_max if pt_max is not None else 500.0, 
                                     event_plane=None)
        
        newcon_df = load_newconrecom_particles(base_path, jet_eta=None, jet_phi=None, cone_radius=None, event_plane=None)
        
        if newcon_df is not None and not newcon_df.empty:
            if pt_min is not None:
                newcon_df = newcon_df[newcon_df["pt"] >= pt_min]
            if pt_max is not None:
                newcon_df = newcon_df[newcon_df["pt"] <= pt_max]
        
        combined_df = pd.concat([spec_df, newcon_df], ignore_index=True)
        
        if combined_df.empty:
            return pd.DataFrame()
        
        if event_plane is not None:
            combined_df["phi"] = map_ang_mpitopi(combined_df["phi"] - event_plane)
        
        if jet_eta is not None and jet_phi is not None and cone_radius is not None:
            dr_to_jet = delta_r(combined_df["eta"], combined_df["phi"], jet_eta, jet_phi)
            final_df = combined_df[dr_to_jet < cone_radius].copy()
        else:
            final_df = combined_df.copy()
        
        total_particles = final_df['weight'].sum()
        cone_msg = f"within dR<{cone_radius}" if cone_radius is not None else "(no cone limit)"
        print(f"Added {total_particles:.2f} particles (from {len(final_df)} entries) from CoLBT {particle_type} {cone_msg}")
        return final_df

    except Exception as e:
        print(f"Error processing CoLBT {particle_type} for event {event_idx}: {e}")
        return pd.DataFrame()
    
def calc_event_plane(particles_df):
    """Calculate event plane angle from particles."""
    if particles_df.empty:
        return 0.0
    
    phi = particles_df["phi"].to_numpy()
    weight = particles_df["weight"].to_numpy()
    
    q2_x = np.sum(weight * np.cos(2 * phi))
    q2_y = np.sum(weight * np.sin(2 * phi))
    
    return map_ang_mpitopi(np.arctan2(q2_y, q2_x) / 2.0) if (q2_x != 0 or q2_y != 0) else 0.0


def get_histogram_weights_vectorized(hist, deta_array, dphi_array):
    """Get weights from ROOT histogram using truly vectorized operations."""
    if hist is None:
        return np.ones_like(deta_array)
    
    nx = hist.GetNbinsX()
    ny = hist.GetNbinsY()
    xmin = hist.GetXaxis().GetXmin()
    xmax = hist.GetXaxis().GetXmax()
    ymin = hist.GetYaxis().GetXmin()
    ymax = hist.GetYaxis().GetXmax()
    
    ix = ((deta_array - xmin) * nx / (xmax - xmin)).astype(np.int32)
    iy = ((dphi_array - ymin) * ny / (ymax - ymin)).astype(np.int32)
    
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    
    hist_array = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            hist_array[i, j] = hist.GetBinContent(i+1, j+1)
    
    weights = hist_array[ix, iy]
    
    return weights


def get_weights_from_np_hist(hist_info, deta_array, dphi_array):
    if hist_info is None:
        return np.ones_like(deta_array)

    h_np = hist_info["array"]
    nx, ny = hist_info["nx"], hist_info["ny"]
    xmin, xmax = hist_info["xmin"], hist_info["xmax"]
    ymin, ymax = hist_info["ymin"], hist_info["ymax"]
    
    ix = np.clip(((deta_array - xmin) * nx / (xmax - xmin)).astype(np.int32), 0, nx - 1)
    iy = np.clip(((dphi_array - ymin) * ny / (ymax - ymin)).astype(np.int32), 0, ny - 1)
    
    weights = h_np[ix, iy]
    
    return weights
def select_equal_particles(df1, df2):
    if df1.empty or df2.empty:
        return df1.copy(), df2.copy()
    n1, n2 = len(df1), len(df2)
    if n1 == n2:
        return df1.copy(), df2.copy()
    elif n1 > n2:
        return df1.sample(n=n2, random_state=RANDOM_SEED).copy(), df2.copy()
    else: # n2 > n1
        return df1.copy(), df2.sample(n=n1, random_state=RANDOM_SEED).copy()

def rotate_particles_to_match_axes(particles_df, old_axis_eta, old_axis_phi, new_axis_eta, new_axis_phi):
    if particles_df.empty:
        return particles_df.copy()
    deta = new_axis_eta - old_axis_eta
    dphi = map_ang_mpitopi(new_axis_phi - old_axis_phi)
    rotated_df = particles_df.copy()
    rotated_df["eta"] += deta
    rotated_df["phi"] = map_ang_mpitopi(rotated_df["phi"] + dphi)
    return rotated_df

def find_matching_pp_event(pb_gamma_pt, available_pp_events):
    """Finds a matching pp event within a 5% pT tolerance."""
    tolerance = 0.05 * pb_gamma_pt
    for i, (pp_idx, pp_gamma_pt_val) in enumerate(available_pp_events):
        if abs(pp_gamma_pt_val - pb_gamma_pt) < tolerance:
            return i, pp_idx, pp_gamma_pt_val
    return -1, -1, None

def _fill_eec_self_pairs(particles_df, jet_pt, hist_dict, active_pt_bins, p_ch_T_cut, n_exponents, delta_r_max, ratio_hist_info=None):
    """Calculates EEC for pairs within a single particle set and fills histograms."""
    if len(particles_df) < 2 or jet_pt < 1e-6: return

    pt = particles_df["pt"].to_numpy(dtype=np.float64)
    eta = particles_df["eta"].to_numpy(dtype=np.float64)
    phi = particles_df["phi"].to_numpy(dtype=np.float64)
    w = particles_df.get("weight", pd.Series(np.ones_like(pt), index=particles_df.index)).to_numpy(dtype=np.float64)

    n = len(pt)
    i, j = np.triu_indices(n, k=1)

    dr = delta_r(eta[i], phi[i], eta[j], phi[j])
    mask = dr <= delta_r_max
    if not np.any(mask): return

    dr = dr[mask]
    
    pt1, pt2 = pt[i][mask], pt[j][mask]
    w1, w2 = w[i][mask], w[j][mask]
    eta1, eta2 = eta[i][mask], eta[j][mask]
    phi1, phi2 = phi[i][mask], phi[j][mask]

    deta_pairs = eta2 - eta1
    dphi_pairs = map_ang_mpitopi(phi2 - phi1)
    
    hist_weights = get_weights_from_np_hist(ratio_hist_info, deta_pairs, dphi_pairs)

    base_weight = w1 * w2 * hist_weights
    pt_prod_term = (pt1 * pt2) / (jet_pt**2)

    for n_exp in n_exponents:
        pair_weight = base_weight * (pt_prod_term ** n_exp) if n_exp > 0 else base_weight
        for pt_min, pt_max in active_pt_bins:
            key = (n_exp, pt_min, pt_max, p_ch_T_cut)
            hist = hist_dict.get(key)
            if hist:
                hist.FillN(len(dr), dr, pair_weight)
                
def _fill_eec_cross_pairs(df1, df2, jet_pt, hist_dict, active_pt_bins, p_ch_T_cut, n_exponents, delta_r_max, ratio_hist_info=None):
    """Calculates EEC for pairs between two different particle sets and fills histograms."""
    if df1.empty or df2.empty or jet_pt < 1e-6: return

    pt1 = df1["pt"].to_numpy(dtype=np.float64)
    eta1 = df1["eta"].to_numpy(dtype=np.float64)
    phi1 = df1["phi"].to_numpy(dtype=np.float64)
    w1 = df1.get("weight", pd.Series(np.ones_like(pt1), index=df1.index)).to_numpy(dtype=np.float64)

    pt2 = df2["pt"].to_numpy(dtype=np.float64)
    eta2 = df2["eta"].to_numpy(dtype=np.float64)
    phi2 = df2["phi"].to_numpy(dtype=np.float64)
    w2 = df2.get("weight", pd.Series(np.ones_like(pt2), index=df2.index)).to_numpy(dtype=np.float64)

    dr = delta_r(eta1[:, None], phi1[:, None], eta2[None, :], phi2[None, :]).flatten()
    mask = dr <= delta_r_max
    if not np.any(mask): return

    dr = dr[mask]

    pt1_pairs = np.broadcast_to(pt1[:, None], (len(pt1), len(pt2))).flatten()[mask]
    pt2_pairs = np.broadcast_to(pt2[None, :], (len(pt1), len(pt2))).flatten()[mask]
    w1_pairs = np.broadcast_to(w1[:, None], (len(pt1), len(pt2))).flatten()[mask]
    w2_pairs = np.broadcast_to(w2[None, :], (len(pt1), len(pt2))).flatten()[mask]
    
    eta1_pairs = np.broadcast_to(eta1[:, None], (len(pt1), len(pt2))).flatten()[mask]
    eta2_pairs = np.broadcast_to(eta2[None, :], (len(pt1), len(pt2))).flatten()[mask]
    phi1_pairs = np.broadcast_to(phi1[:, None], (len(pt1), len(pt2))).flatten()[mask]
    phi2_pairs = np.broadcast_to(phi2[None, :], (len(pt1), len(pt2))).flatten()[mask]
    
    deta_pairs = eta2_pairs - eta1_pairs
    dphi_pairs = map_ang_mpitopi(phi2_pairs - phi1_pairs)
    
    hist_weights = get_weights_from_np_hist(ratio_hist_info, deta_pairs, dphi_pairs)

    base_weight = w1_pairs * w2_pairs * hist_weights
    pt_prod_term = (pt1_pairs * pt2_pairs) / (jet_pt**2)

    for n_exp in n_exponents:
        pair_weight = base_weight * (pt_prod_term ** n_exp) if n_exp > 0 else base_weight
        for pt_min, pt_max in active_pt_bins:
            key = (n_exp, pt_min, pt_max, p_ch_T_cut)
            hist = hist_dict.get(key)
            if hist:
                hist.FillN(len(dr), dr, pair_weight)

def energy_energy_correlator_pbpbpp(
    pbpb_datapath, pp_datapath, hydro_datapath, outfilename,
    start_file, end_file, pp_start, pp_end
):
    print("Pre-scanning PP events to find available photons...")
    available_pp_events = []
    for pp_idx in range(pp_start, pp_end):
        pp_path = os.path.join(pp_datapath, f"event{pp_idx}")
        gamma_file = os.path.join(pp_path, "gamma.dat")
        if not os.path.exists(gamma_file): continue
        try:
            gamma_df = pd.read_csv(gamma_file, sep='\s+', names=["px","py","pz","E"], engine='python')
            if not gamma_df.empty:
                pp_gamma_pt = np.sqrt(gamma_df["px"][0]**2 + gamma_df["py"][0]**2)
                available_pp_events.append((pp_idx, pp_gamma_pt))
        except Exception:
            continue
    print(f"Found {len(available_pp_events)} valid PP events with photon data.")

    gamma_pt_bins = [(0, 500)]
    p_ch_T_cuts = [0.0, 1.0]
    n_exponents = [0, 1, 2]
    delta_r_min, delta_r_max, n_bins = 0.01, 1.0, 40
    delta_r_bins = np.logspace(np.log10(delta_r_min), np.log10(delta_r_max), n_bins + 1)
    deta_bins, deta_min, deta_max = 30, -4.0, 4.0
    dphi_bins, dphi_min, dphi_max = 30, -np.pi, np.pi
    cone2_radii = np.arange(0.0, 2.0, 0.05)

    hists = {}
    deta_dphi_hists = {} 
    pt_tracking_stats = {} 
    
    prefixes = ["pbpb_signal", "pbpb_sm1", "pbpb_m1m1", "pbpb_m1m2", "pbpb_spec_m1", "pbpb_newcon_cone", "pp_signal", "pp_sm1", "pp_m1m1", "pp_m1m2", "pp_hadron_cone"]
    for prefix in prefixes:
        hists[prefix] = {}
        for n_exp in n_exponents:
            for pt_min, pt_max in gamma_pt_bins:
                for p_cut in p_ch_T_cuts:
                    key = (n_exp, pt_min, pt_max, p_cut)
                    name = f"{prefix}_eec_n{n_exp}_jetpt{pt_min}to{pt_max}_pch{p_cut}"
                    title = f"{prefix.replace('_', ' ')} EEC n={n_exp}, pT {pt_min}-{pt_max}, p_ch>{p_cut};R_L;EEC"
                    hists[prefix][key] = ROOT.TH1D(name, title, n_bins, delta_r_bins)
                    hists[prefix][key].Sumw2()

    for pt_min, pt_max in gamma_pt_bins:
        for p_cut in p_ch_T_cuts:
            key = (pt_min, pt_max, p_cut)
            h_signal = ROOT.TH2D(f"h_deta_dphi_signal_jetpt{pt_min}to{pt_max}_pch{p_cut}", f"Signal #Delta#eta vs #Delta#phi (pT {pt_min}-{pt_max}, p_ch>{p_cut});#Delta#eta;#Delta#phi", deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
            h_signal.Sumw2()
            h_mixed = ROOT.TH2D(f"h_deta_dphi_mixed_jetpt{pt_min}to{pt_max}_pch{p_cut}", f"Mixed #Delta#eta vs #Delta#phi (pT {pt_min}-{pt_max}, p_ch>{p_cut});#Delta#eta;#Delta#phi", deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
            h_mixed.Sumw2()
            deta_dphi_hists[key] = {"signal": h_signal, "mixed": h_mixed}
            pt_tracking_stats[key] = {'n_events': 0, 'jet_pt_sum': 0.0, 'pt_sig_c1_sum': 0.0, 'pt_mix_c1_sum': 0.0, 'pt_sig_c2_sum': np.zeros_like(cone2_radii, dtype=float), 'pt_mix_c2_sum': np.zeros_like(cone2_radii, dtype=float)}

    n_pt_bins = len(gamma_pt_bins)
    pbpb_jet_counts = ROOT.TH1D("pbpb_jet_counts", "PbPb Jet Counts", n_pt_bins, 0, n_pt_bins)
    pp_jet_counts = ROOT.TH1D("pp_jet_counts", "PP Jet Counts", n_pt_bins, 0, n_pt_bins)
    for i, (pt_min, pt_max) in enumerate(gamma_pt_bins):
        pbpb_jet_counts.GetXaxis().SetBinLabel(i+1, f"{pt_min}-{pt_max} GeV")
        pp_jet_counts.GetXaxis().SetBinLabel(i+1, f"{pt_min}-{pt_max} GeV")

    # ---
    # --- PHASE 1: Fill Deta-Dphi and Pt Tracking Stats
    # ---
    print("\n--- Starting Phase 1: Accumulating Correlation Data ---")
    for evt_idx in range(start_file, end_file):
        print(f"\n--- Phase 1: Processing event {evt_idx} ---")
        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        colbt_mixed_path = os.path.join("./CoLBT_Minimum_Bias", f"event{evt_idx}")
        if not os.path.isdir(colbt_mixed_path):
            print(f"Mixed event directory not found for event {evt_idx}. Skipping.")
            continue

        try:
            hadjet_df = pd.read_csv(os.path.join(pbpb_path, "hadjet.dat"), sep='\s+', names=["jet_no","px","py","pz","E","eta"], engine='python')
            gamma_df = pd.read_csv(os.path.join(pbpb_path, "gamma.dat"), sep='\s+', names=["px","py","pz","E"], engine='python')
            if hadjet_df.empty or gamma_df.empty: continue
        except Exception as e:
            print(f"Could not read files for PbPb event {evt_idx}: {e}")
            continue

        all_particles_for_ep = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=100.0, event_plane=None)
        event_plane = calc_event_plane(all_particles_for_ep)

        jet_row = hadjet_df.iloc[0]
        jet_pt = np.sqrt(jet_row["px"]**2+jet_row["py"]**2)
        jet_eta = jet_row["eta"]
        jet_phi_raw = map_ang_mpitopi(np.arctan2(jet_row["py"],jet_row["px"]))
        jet_phi = map_ang_mpitopi(jet_phi_raw - event_plane)  
        pb_gamma_pt = np.sqrt(gamma_df["px"].iloc[0]**2 + gamma_df["py"].iloc[0]**2)
        active_pt_bins = [(pt_min, pt_max) for pt_min, pt_max in gamma_pt_bins if pt_min <= pb_gamma_pt < pt_max]
        if not active_pt_bins: continue

        all_particles_for_ep = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=100.0, event_plane=None)
        event_plane = calc_event_plane(all_particles_for_ep)

        all_particles_for_ep = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=100.0, event_plane=None)
        event_plane = calc_event_plane(all_particles_for_ep)

        pbpb_spec_all_df = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=5.0, event_plane=event_plane)
        pp_mixed_all_particles_df = load_colbt_mixed_particles(evt_idx, "mixed", jet_eta=None, jet_phi=None, cone_radius=None, event_plane=event_plane, pt_min=0, pt_max=5)

        for p_cut in p_ch_T_cuts:
            pbpb_signal_all_f = pbpb_spec_all_df[pbpb_spec_all_df["pt"] >= p_cut] if not pbpb_spec_all_df.empty else pd.DataFrame()
            pp_mixed_all_f = pp_mixed_all_particles_df[pp_mixed_all_particles_df["pt"] >= p_cut] if not pp_mixed_all_particles_df.empty else pd.DataFrame()
            
            for pt_min_bin, pt_max_bin in active_pt_bins:
                key = (pt_min_bin, pt_max_bin, p_cut)
                hists_for_bin = deta_dphi_hists.get(key)
                if not pbpb_signal_all_f.empty:
                    hists_for_bin["signal"].FillN(len(pbpb_signal_all_f), pbpb_signal_all_f["eta"].to_numpy() - jet_eta, map_ang_mpitopi(pbpb_signal_all_f["phi"].to_numpy() - jet_phi), pbpb_signal_all_f["weight"].to_numpy())
                if not pp_mixed_all_f.empty:
                    hists_for_bin["mixed"].FillN(len(pp_mixed_all_f), pp_mixed_all_f["eta"].to_numpy() - jet_eta, map_ang_mpitopi(pp_mixed_all_f["phi"].to_numpy() - jet_phi), pp_mixed_all_f["weight"].to_numpy())
                
                stats = pt_tracking_stats[key]
                if p_cut == p_ch_T_cuts[0]:
                    stats['n_events'] += 1; stats['jet_pt_sum'] += jet_pt
                # Cone 1
                if not pbpb_signal_all_f.empty: stats['pt_sig_c1_sum'] += (pbpb_signal_all_f.loc[delta_r(pbpb_signal_all_f["eta"], pbpb_signal_all_f["phi"], jet_eta, jet_phi) < MAX_DR, 'pt'] * pbpb_signal_all_f.loc[delta_r(pbpb_signal_all_f["eta"], pbpb_signal_all_f["phi"], jet_eta, jet_phi) < MAX_DR, 'weight']).sum()
                if not pp_mixed_all_f.empty: stats['pt_mix_c1_sum'] += (pp_mixed_all_f.loc[delta_r(pp_mixed_all_f["eta"], pp_mixed_all_f["phi"], jet_eta, jet_phi) < MAX_DR, 'pt'] * pp_mixed_all_f.loc[delta_r(pp_mixed_all_f["eta"], pp_mixed_all_f["phi"], jet_eta, jet_phi) < MAX_DR, 'weight']).sum()
                # Cone 2
                cone2_center_eta, cone2_center_phi = jet_eta, map_ang_mpitopi(jet_phi + np.pi)
                if not pbpb_signal_all_f.empty:
                    dr_c2_base = delta_r(pbpb_signal_all_f["eta"], pbpb_signal_all_f["phi"], cone2_center_eta, cone2_center_phi)
                    for i, r in enumerate(cone2_radii): stats['pt_sig_c2_sum'][i] += (pbpb_signal_all_f.loc[dr_c2_base < r, 'pt'] * pbpb_signal_all_f.loc[dr_c2_base < r, 'weight']).sum()
                if not pp_mixed_all_f.empty:
                    dr_c2_base = delta_r(pp_mixed_all_f["eta"], pp_mixed_all_f["phi"], cone2_center_eta, cone2_center_phi)
                    for i, r in enumerate(cone2_radii): stats['pt_mix_c2_sum'][i] += (pp_mixed_all_f.loc[dr_c2_base < r, 'pt'] * pp_mixed_all_f.loc[dr_c2_base < r, 'weight']).sum()

    print("\n--- Processing Phase 1 Data and Performing Cutout Analysis ---")
    outfile = ROOT.TFile(outfilename, "RECREATE")
    
    optimal_radii = {}

    print("\n--- Cone Pt Control Analysis Results & Optimal Radii Determination ---")
    for (pt_min, pt_max, p_cut), stats in pt_tracking_stats.items():
        n_events = stats['n_events']
        if n_events == 0: continue
        print(f"\n>>> Analysis for Jet pT: {pt_min}-{pt_max} GeV, Charged pT > {p_cut} GeV (N_events = {n_events})")
        pt_sig_avg_c1 = stats['pt_sig_c1_sum'] / n_events
        pt_mix_avg_c1 = stats['pt_mix_c1_sum'] / n_events
        pt_ctr = abs(pt_sig_avg_c1 - pt_mix_avg_c1)
        print(f"      - Control Value |pt_sig_avg - pt_mix_avg| = {pt_ctr:.4f}")
        pt_sig_avg_c2 = stats['pt_sig_c2_sum'] / n_events
        pt_mix_avg_c2 = stats['pt_mix_c2_sum'] / n_events
        pt_ctr_c2 = np.abs(pt_sig_avg_c2 - pt_mix_avg_c2)
        best_idx = np.argmin(np.abs(pt_ctr_c2 - pt_ctr))
        best_radius = cone2_radii[best_idx]
        optimal_radii[(pt_min, pt_max, p_cut)] = best_radius
        print(f"    >>> BEST MATCH FOUND: Optimal Away-Side Radius = {best_radius:.2f}")

    print("\nProcessing and saving delta eta/phi histograms...")
    ratio_hist_dir = outfile.mkdir("ratio_histograms")
    sub_diff_dir = ratio_hist_dir.mkdir("signal_minus_mixed")
    sub_ratio_dir = ratio_hist_dir.mkdir("2_minus_signal_over_mixed")
   
    print("\nPerforming away-side cutout, translation, and compression...")
    cutout_dir = outfile.mkdir("AwaySideAnalysis")
    translated_dir = cutout_dir.mkdir("TranslatedCutouts")
    compressed_dir = cutout_dir.mkdir("CompressedCutouts")

    for i, (pt_min, pt_max) in enumerate(gamma_pt_bins):
        key_for_nevents = (pt_min, pt_max, p_ch_T_cuts[0])
        n_events = pt_tracking_stats.get(key_for_nevents, {}).get('n_events', 0)
        if n_events == 0: continue
        
        for p_cut in p_ch_T_cuts:
            key = (pt_min, pt_max, p_cut)
            R_opt = optimal_radii.get(key)
            if R_opt is None: continue

            print(f"  Processing cutouts for pT {pt_min}-{pt_max}, p_ch>{p_cut} with R_opt={R_opt:.2f}")

            hists_for_bin = deta_dphi_hists.get(key)
            h_signal = hists_for_bin["signal"]
            h_mixed = hists_for_bin["mixed"]
            
            h_subtracted = h_signal.Clone(f"h_subtracted_jetpt{pt_min}to{pt_max}_pch{p_cut}")
            h_subtracted.Add(h_mixed, -1.0)
            
            h_ratio = h_signal.Clone(f"h_ratio_jetpt{pt_min}to{pt_max}_pch{p_cut}")
            h_ratio.Divide(h_mixed)
            h_2_minus_ratio = h_signal.Clone(f"h_2_minus_ratio_jetpt{pt_min}to{pt_max}_pch{p_cut}")
            h_2_minus_ratio.Reset()
            for ix in range(1, h_ratio.GetNbinsX() + 1):
                for iy in range(1, h_ratio.GetNbinsY() + 1):
                    if h_mixed.GetBinContent(ix, iy) > 0:
                        h_2_minus_ratio.SetBinContent(ix, iy, 2.0 - h_ratio.GetBinContent(ix, iy))
            
            sub_diff_dir.cd(); h_subtracted.Write()
            sub_ratio_dir.cd(); h_2_minus_ratio.Write()

            source_hists = {
                "subtracted": h_subtracted,
                "2_minus_ratio": h_2_minus_ratio
            }

            for name, h_source in source_hists.items():
                N_phi_bins = h_source.GetNbinsY()
                phi_shift_bins = N_phi_bins // 2

                h_shifted_full = h_source.Clone(f"h_{name}_shifted_full_jetpt{pt_min}to{pt_max}_pch{p_cut}")
                h_shifted_full.Reset() 
                h_shifted_full.SetTitle(f"Full Pi Shift of {name};#Delta#eta;#Delta#phi")
                
                for ix in range(1, h_source.GetNbinsX() + 1):
                    for iy_source in range(1, N_phi_bins + 1):
                        iy_dest = (iy_source - 1 + phi_shift_bins) % N_phi_bins + 1
                        
                        content = h_source.GetBinContent(ix, iy_source)
                        error = h_source.GetBinError(ix, iy_source)
                        
                        h_shifted_full.SetBinContent(ix, iy_dest, content)
                        h_shifted_full.SetBinError(ix, iy_dest, error)

                h_cutout = h_shifted_full.Clone(f"h_{name}_translated_cutout_jetpt{pt_min}to{pt_max}_pch{p_cut}")
                h_cutout.SetTitle(f"Translated Cutout of {name} (R < {R_opt:.2f});#Delta#eta;#Delta#phi")

                for ix in range(1, h_cutout.GetNbinsX() + 1):
                    for iy in range(1, h_cutout.GetNbinsY() + 1):
                        d_eta = h_cutout.GetXaxis().GetBinCenter(ix)
                        d_phi = h_cutout.GetYaxis().GetBinCenter(iy)
                       
                        if np.sqrt(d_eta**2 + d_phi**2) > R_opt:
                            h_cutout.SetBinContent(ix, iy, 0)
                            h_cutout.SetBinError(ix, iy, 0)
                
                translated_dir.cd()
                h_cutout.Write()
                h_cutout.ProjectionX(f"h_{name}_cutout_projX_jetpt{pt_min}to{pt_max}_pch{p_cut}").Write()
                h_cutout.ProjectionY(f"h_{name}_cutout_projY_jetpt{pt_min}to{pt_max}_pch{p_cut}").Write()

                N_eta_bins = h_cutout.GetNbinsX()
                # N_phi_bins is already defined

                h_compressed = ROOT.TH2D(
                    f"h_{name}_compressed_jetpt{pt_min}to{pt_max}_pch{p_cut}",
                    f"Compressed Cutout of {name} (R=0.5);#Delta#eta;#Delta#phi",
                    N_eta_bins, -4, 4,  
                    N_phi_bins, -np.pi, np.pi
                )
                h_compressed.Sumw2()

                compression_factor = 0.5 / R_opt  

                for ix in range(1, N_eta_bins + 1):
                    for iy in range(1, N_phi_bins + 1):
                        
                        eta_low = h_compressed.GetXaxis().GetBinLowEdge(ix)
                        eta_high = h_compressed.GetXaxis().GetBinUpEdge(ix)
                        phi_low = h_compressed.GetYaxis().GetBinLowEdge(iy)
                        phi_high = h_compressed.GetYaxis().GetBinUpEdge(iy)

                        eta_low_orig = eta_low / compression_factor
                        eta_high_orig = eta_high / compression_factor
                        phi_low_orig = phi_low / compression_factor
                        phi_high_orig = phi_high / compression_factor

                        sum_excess = 0.0
                        for jx in range(1, h_cutout.GetNbinsX() + 1):
                            eta_center = h_cutout.GetXaxis().GetBinCenter(jx)
                            if eta_center < eta_low_orig or eta_center >= eta_high_orig:
                                continue
                            for jy in range(1, h_cutout.GetNbinsY() + 1):
                                phi_center = h_cutout.GetYaxis().GetBinCenter(jy)
                                if phi_center < phi_low_orig or phi_center >= phi_high_orig:
                                    continue
                                val = h_cutout.GetBinContent(jx, jy)
                                sum_excess += (val - 1.0)
                        h_compressed.SetBinContent(ix, iy, 1.0 + sum_excess)
                        
                        h_compressed.SetBinError(ix, iy, 0.0)
                
                if name == "2_minus_ratio":
                    print(f"    -> Filling empty (zero) bins with 1.0 in compressed 2-minus-ratio histogram...")
                    
                    for ix in range(1, h_compressed.GetNbinsX() + 1):
                        for iy in range(1, h_compressed.GetNbinsY() + 1):
                            
                            if h_compressed.GetBinContent(ix, iy) == 0.0:
                                
                                h_compressed.SetBinContent(ix, iy, 1.0)
                                
                                h_compressed.SetBinError(ix, iy, 0.0)
            

                
                if name == "2_minus_ratio":  
                    if "compressed_ratio_histograms" not in locals():
                        compressed_ratio_histograms = {}

                    nx = h_compressed.GetNbinsX()
                    ny = h_compressed.GetNbinsY()
                    
                    hist_np = np.array([[h_compressed.GetBinContent(ix, iy) for iy in range(1, ny + 1)] for ix in range(1, nx + 1)], dtype=np.float64)

                    ix, iy = np.indices(hist_np.shape)
                    
                    checkerboard_mask = (ix + iy) % 2 == 0
                    
                    hist_np[checkerboard_mask] = 1.0
                    
                    x_axis = h_compressed.GetXaxis()
                    y_axis = h_compressed.GetYaxis()
                    
                    hist_info = {
                        "array": hist_np,
                        "nx": nx, "ny": ny,
                        "xmin": x_axis.GetXmin(), "xmax": x_axis.GetXmax(),
                        "ymin": y_axis.GetXmin(), "ymax": y_axis.GetXmax()  
                    }
                    
                    compressed_ratio_histograms[key] = hist_info
                   
                    
                compressed_dir.cd()  
                h_compressed.Write()
                h_compressed.ProjectionX(f"h_{name}_compressed_projX_jetpt{pt_min}to{pt_max}_pch{p_cut}").Write()
                h_compressed.ProjectionY(f"h_{name}_compressed_projY_jetpt{pt_min}to{pt_max}_pch{p_cut}").Write()
 

    # ---
    # --- PHASE 2: Get PP events and perform EEC Calculation
    # ---
    print("\n--- Starting Phase 2: EEC Calculation ---")
    processed_pb_events, processed_pp_events = 0, 0
    available_pp_events_for_loop2 = list(available_pp_events)

    for evt_idx in range(start_file, end_file):
        print(f"\n--- Phase 2: Processing event {evt_idx} ---")
        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        colbt_mixed_path = os.path.join("./CoLBT_Minimum_Bias", f"event{evt_idx}")
        if not os.path.isdir(colbt_mixed_path):
            continue

        try:
            hadjet_df = pd.read_csv(os.path.join(pbpb_path, "hadjet.dat"), sep='\s+', names=["jet_no","px","py","pz","E","eta"], engine='python')
            gamma_df = pd.read_csv(os.path.join(pbpb_path, "gamma.dat"), sep='\s+', names=["px","py","pz","E"], engine='python')
            if hadjet_df.empty or gamma_df.empty: continue
        except Exception as e:
            continue

        all_particles_for_ep = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=100.0, event_plane=None)
        event_plane = calc_event_plane(all_particles_for_ep)

        jet_row = hadjet_df.iloc[0]
        jet_pt = np.sqrt(jet_row["px"]**2+jet_row["py"]**2)
        jet_eta = jet_row["eta"]
        jet_phi_raw = map_ang_mpitopi(np.arctan2(jet_row["py"],jet_row["px"]))
        jet_phi = map_ang_mpitopi(jet_phi_raw - event_plane)  # Rotate jet phi by event plane
        pb_gamma_pt = np.sqrt(gamma_df["px"].iloc[0]**2 + gamma_df["py"].iloc[0]**2)
        active_pt_bins = [(pt_min, pt_max) for pt_min, pt_max in gamma_pt_bins if pt_min <= pb_gamma_pt < pt_max]
        if not active_pt_bins: continue

        match_idx, matched_pp_idx, matched_pp_gamma_pt = find_matching_pp_event(pb_gamma_pt, available_pp_events_for_loop2)
        if match_idx == -1: continue
        available_pp_events_for_loop2.pop(match_idx)

        pp_path = os.path.join(pp_datapath, f"event{matched_pp_idx}")
        try:
            pp_hadjet_df = pd.read_csv(os.path.join(pp_path, "hadjet.dat"), sep='\s+', names=["jno","px","py","pz","E","eta"], engine='python')
            pp_jet_row = pp_hadjet_df.iloc[0]
            pp_jet_pt, pp_jet_eta, pp_jet_phi = np.sqrt(pp_jet_row["px"]**2+pp_jet_row["py"]**2), pp_jet_row["eta"], map_ang_mpitopi(np.arctan2(pp_jet_row["py"], pp_jet_row["px"]))
        except Exception as e:
            continue

        all_particles_for_ep = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=100.0, event_plane=None)
        event_plane = calc_event_plane(all_particles_for_ep)

        pbpb_newcon_df = load_newconrecom_particles(pbpb_path, jet_eta, jet_phi, cone_radius=MAX_DR, event_plane=event_plane)
        pp_hadron_df = load_hadron_particles(pp_path, pp_jet_eta, pp_jet_phi)
        pbpb_newcon_df, pp_hadron_df = pbpb_newcon_df, pp_hadron_df
        pbpb_spec_df = load_spec_particles(pbpb_path, jet_eta, jet_phi, cone_radius=MAX_DR, pt_min=0.0, pt_max=100.0, event_plane=event_plane)
        pbpb_signal_df = pd.concat([pbpb_newcon_df, pbpb_spec_df], ignore_index=True) if not pbpb_newcon_df.empty or not pbpb_spec_df.empty else pd.DataFrame()

        hydro_m1_df = load_colbt_mixed_particles(evt_idx, "m1", jet_eta, jet_phi, cone_radius=MAX_DR, event_plane=event_plane, pt_min=None, pt_max=None)
        hydro_m2_df = load_colbt_mixed_particles(evt_idx, "m2", jet_eta, jet_phi, cone_radius=MAX_DR, event_plane=event_plane, pt_min=None, pt_max=None)

        rotated_hadron_df = rotate_particles_to_match_axes(pp_hadron_df, pp_jet_eta, pp_jet_phi, jet_eta, jet_phi)
        pp_mixed_df = load_colbt_mixed_particles(evt_idx, "mixed", jet_eta, jet_phi, cone_radius=MAX_DR, event_plane=event_plane, pt_min=None, pt_max=None)
        pp_signal_df = pd.concat([rotated_hadron_df, pp_mixed_df], ignore_index=True) if not rotated_hadron_df.empty or not pp_mixed_df.empty else pd.DataFrame()
        pp_m1_df = hydro_m1_df.copy()  
        pp_m2_df = hydro_m2_df.copy()  

        for p_cut in p_ch_T_cuts:
            
            pbpb_signal_f = pbpb_signal_df[pbpb_signal_df["pt"] >= p_cut]; pbpb_hydro_m1_f = hydro_m1_df[hydro_m1_df["pt"] >= p_cut]; pbpb_hydro_m2_f = hydro_m2_df[hydro_m2_df["pt"] >= p_cut]; pbpb_newcon_f = pbpb_newcon_df[pbpb_newcon_df["pt"] >= p_cut]; pbpb_spec_f = pbpb_spec_df[pbpb_spec_df["pt"] >= p_cut]
            pp_signal_f = pp_signal_df[pp_signal_df["pt"] >= p_cut]; pp_hadron_cone_f = rotated_hadron_df[rotated_hadron_df["pt"] >= p_cut]; pp_m1_f = pp_m1_df[pp_m1_df["pt"] >= p_cut]; pp_m2_f = pp_m2_df[pp_m2_df["pt"] >= p_cut]
            
            hist_key = (pt_min, pt_max, p_cut) if (pt_min, pt_max) in active_pt_bins else None
            ratio_hist_info = compressed_ratio_histograms.get(hist_key) if hist_key and 'compressed_ratio_histograms' in locals() else None
            
            _fill_eec_self_pairs(pbpb_signal_f, jet_pt, hists["pbpb_signal"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_cross_pairs(pbpb_signal_f, pbpb_hydro_m1_f, jet_pt, hists["pbpb_sm1"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_self_pairs(pbpb_hydro_m1_f, jet_pt, hists["pbpb_m1m1"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_cross_pairs(pbpb_hydro_m1_f, pbpb_hydro_m2_f, jet_pt, hists["pbpb_m1m2"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_cross_pairs(pbpb_spec_f, pbpb_hydro_m1_f, jet_pt, hists["pbpb_spec_m1"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_self_pairs(pbpb_newcon_f, jet_pt, hists["pbpb_newcon_cone"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            
            _fill_eec_self_pairs(pp_signal_f, pp_jet_pt, hists["pp_signal"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_cross_pairs(pp_signal_f, pp_m1_f, pp_jet_pt, hists["pp_sm1"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_self_pairs(pp_m1_f, pp_jet_pt, hists["pp_m1m1"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_cross_pairs(pp_m1_f, pp_m2_f, pp_jet_pt, hists["pp_m1m2"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
            _fill_eec_self_pairs(pp_hadron_cone_f, pp_jet_pt, hists["pp_hadron_cone"], active_pt_bins, p_cut, n_exponents, delta_r_max, ratio_hist_info)
        for i, (pt_min, pt_max) in enumerate(gamma_pt_bins):
            if pt_min <= pb_gamma_pt < pt_max:
                pbpb_jet_counts.Fill(i); pp_jet_counts.Fill(i)
        processed_pb_events += 1; processed_pp_events += 1
        print(f"Progress: {processed_pb_events}/{end_file-start_file} event pairs processed.")

    print(f"\nCompleted processing. Processed {processed_pb_events} PbPb and {processed_pp_events} PP events.")
    
    print("Saving EEC histograms...")
    for prefix in hists:
        path_parts = prefix.split('_'); top_dir_name_internal = path_parts[0]
        output_top_dir_name = {'pbpb': 'PbPb', 'pp': 'PP'}.get(top_dir_name_internal, top_dir_name_internal)
        top_dir = outfile.Get(output_top_dir_name) or outfile.mkdir(output_top_dir_name)
        final_dir = top_dir
        if len(path_parts) > 1:
            sub_dir_name = "_".join(path_parts[1:]); sub_dir = top_dir.Get(sub_dir_name) or top_dir.mkdir(sub_dir_name); final_dir = sub_dir
        final_dir.cd()
        for hist in hists[prefix].values(): hist.Write()
    outfile.cd()

    counts_dir = outfile.mkdir("JetCounts"); counts_dir.cd()
    pbpb_jet_counts.Write(); pp_jet_counts.Write()

    outfile.Close()
    print(f"Histograms successfully written to {outfilename}")

if __name__ == '__main__':
    pbpb_datapath  = "/home/CoLBT"
    pp_datapath    = "/home/PP"
    hydro_datapath = "/home/Hydrobackground"
    outfilename    = "/home/EEC_embed_augmented_expt.root"
    energy_energy_correlator_pbpbpp(
        pbpb_datapath, pp_datapath, hydro_datapath, outfilename,
        start_file=0, end_file=3999, pp_start=0, pp_end=4999
    )