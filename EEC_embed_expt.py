"""
Gets EEC as in experiments without using hydrobackground as m (using random minimum bias events instead), needs minimum bias directory to run
"""

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
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def delta_r(eta1, phi1, eta2, phi2):
    dphi = map_ang_mpitopi(phi1 - phi2)
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

def load_hadron_particles(filepath, jet_eta=None, jet_phi=None):
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
        charged_pids = [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13]
        charged_particles_df = hadron_df[np.abs(hadron_df["pid"]).isin(charged_pids)].copy()
        if jet_eta is not None and jet_phi is not None:
            dr_to_jet = delta_r(charged_particles_df["eta"], charged_particles_df["phi"], jet_eta, jet_phi)
            charged_particles_df = charged_particles_df[dr_to_jet < MAX_DR]
        charged_particles_df["source"] = "signal_hadron"
        charged_particles_df["weight"] = 1.0
        charged_particles_df.rename(columns={'E': 'energy'}, inplace=True)
        final_cols = ["pt", "eta", "phi", "pid", "source", "weight", "energy"]
        particles_in_cone = charged_particles_df[[col for col in final_cols if col in charged_particles_df.columns]]
        return particles_in_cone
    except Exception as e:
        return pd.DataFrame()

def load_spec_particles(filepath, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=0.0, pt_max=500.0, event_plane=None):
    try:
        event_charged_dat = os.path.join(filepath, "dNdEtaPtdPtdPhi_Charged.dat")
        if not os.path.exists(event_charged_dat):
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
        return particles_df
    except Exception as e:
        return pd.DataFrame()

def load_newconrecom_particles(filepath, jet_eta=None, jet_phi=None, cone_radius=None, event_plane=None):
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
        if event_plane is not None:
            charged_particles_df["phi"] = map_ang_mpitopi(charged_particles_df["phi"] - event_plane)
        if jet_eta is not None and jet_phi is not None and cone_radius is not None:
            dr_to_jet = delta_r(charged_particles_df["eta"], charged_particles_df["phi"], jet_eta, jet_phi)
            final_df = charged_particles_df[dr_to_jet < cone_radius].copy()
        else:
            final_df = charged_particles_df.copy()
        final_df["source"] = "newcon"
        final_df["weight"] = 1.0
        final_cols = ["pt", "eta", "phi", "source", "weight"]
        return final_df[final_cols]
    except Exception as e:
        return pd.DataFrame()

def load_colbt_mixed_particles(event_idx, particle_type, jet_eta=None, jet_phi=None, cone_radius=None, event_plane=None, pt_min=None, pt_max=None):
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
        return final_df
    except Exception as e:
        return pd.DataFrame()

def calc_event_plane(particles_df):
    if particles_df.empty:
        return 0.0
    phi = particles_df["phi"].to_numpy()
    weight = particles_df["weight"].to_numpy()
    q2_x = np.sum(weight * np.cos(2 * phi))
    q2_y = np.sum(weight * np.sin(2 * phi))
    return map_ang_mpitopi(np.arctan2(q2_y, q2_x) / 2.0) if (q2_x != 0 or q2_y != 0) else 0.0

def get_weights_from_np_hist(hist_info, deta_array, dphi_array):
    return np.ones_like(deta_array)

def select_equal_particles(df1, df2):
    if df1.empty or df2.empty:
        return df1.copy(), df2.copy()
    n1, n2 = len(df1), len(df2)
    if n1 == n2:
        return df1.copy(), df2.copy()
    elif n1 > n2:
        return df1.sample(n=n2, random_state=RANDOM_SEED).copy(), df2.copy()
    else:
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
    tolerance = 0.05 * pb_gamma_pt
    for i, (pp_idx, pp_gamma_pt_val) in enumerate(available_pp_events):
        if abs(pp_gamma_pt_val - pb_gamma_pt) < tolerance:
            return i, pp_idx, pp_gamma_pt_val
    return -1, -1, None

def _fill_eec_self_pairs(particles_df, jet_pt, hist_dict, active_pt_bins, p_ch_T_cut, n_exponents, delta_r_max, ratio_hist_info=None):
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
    gamma_pt_bins = [(0, 500)]
    p_ch_T_cuts = [0.0, 1.0]
    n_exponents = [0, 1, 2]
    delta_r_min, delta_r_max, n_bins = 0.01, 1.0, 40
    delta_r_bins = np.logspace(np.log10(delta_r_min), np.log10(delta_r_max), n_bins + 1)
    hists = {}
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
    n_pt_bins = len(gamma_pt_bins)
    pbpb_jet_counts = ROOT.TH1D("pbpb_jet_counts", "PbPb Jet Counts", n_pt_bins, 0, n_pt_bins)
    pp_jet_counts = ROOT.TH1D("pp_jet_counts", "PP Jet Counts", n_pt_bins, 0, n_pt_bins)
    for i, (pt_min, pt_max) in enumerate(gamma_pt_bins):
        pbpb_jet_counts.GetXaxis().SetBinLabel(i+1, f"{pt_min}-{pt_max} GeV")
        pp_jet_counts.GetXaxis().SetBinLabel(i+1, f"{pt_min}-{pt_max} GeV")
    processed_pb_events, processed_pp_events = 0, 0
    available_pp_events_for_loop2 = list(available_pp_events)
    for evt_idx in range(start_file, end_file):
        print(f"processing event{evt_idx}")
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
        jet_phi = map_ang_mpitopi(jet_phi_raw - event_plane)
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
        N_MB_EVENTS = end_file - start_file
        while True:
            m1_idx = np.random.randint(start_file, end_file)
            m1_path = os.path.join("./CoLBT_Minimum_Bias", f"event{m1_idx}", "m1")
            if os.path.isdir(m1_path):
                break
        while True:
            m2_idx = np.random.randint(start_file, end_file)
            if m2_idx == m1_idx:
                continue
            m2_path = os.path.join("./CoLBT_Minimum_Bias", f"event{m2_idx}", "m2")
            if os.path.isdir(m2_path):
                break
        hydro_m1_df = load_colbt_mixed_particles(m1_idx, "m1", jet_eta, jet_phi, cone_radius=MAX_DR, event_plane=event_plane, pt_min=None, pt_max=None)
        hydro_m2_df = load_colbt_mixed_particles(m2_idx, "m2", jet_eta, jet_phi, cone_radius=MAX_DR, event_plane=event_plane, pt_min=None, pt_max=None)
        rotated_hadron_df = rotate_particles_to_match_axes(pp_hadron_df, pp_jet_eta, pp_jet_phi, jet_eta, jet_phi)
        pp_mixed_df = load_colbt_mixed_particles(evt_idx, "mixed", jet_eta, jet_phi, cone_radius=MAX_DR, event_plane=event_plane, pt_min=None, pt_max=None)
        pp_signal_df = pd.concat([rotated_hadron_df, pp_mixed_df], ignore_index=True) if not rotated_hadron_df.empty or not pp_mixed_df.empty else pd.DataFrame()
        pp_m1_df = hydro_m1_df.copy()
        pp_m2_df = hydro_m2_df.copy()
        for p_cut in p_ch_T_cuts:
            pbpb_signal_f = pbpb_signal_df[pbpb_signal_df["pt"] >= p_cut]; pbpb_hydro_m1_f = hydro_m1_df[hydro_m1_df["pt"] >= p_cut]; pbpb_hydro_m2_f = hydro_m2_df[hydro_m2_df["pt"] >= p_cut]; pbpb_newcon_f = pbpb_newcon_df[pbpb_newcon_df["pt"] >= p_cut]; pbpb_spec_f = pbpb_spec_df[pbpb_spec_df["pt"] >= p_cut]
            pp_signal_f = pp_signal_df[pp_signal_df["pt"] >= p_cut]; pp_hadron_cone_f = rotated_hadron_df[rotated_hadron_df["pt"] >= p_cut]; pp_m1_f = pp_m1_df[pp_m1_df["pt"] >= p_cut]; pp_m2_f = pp_m2_df[pp_m2_df["pt"] >= p_cut]
            ratio_hist_info = None
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
    outfile = ROOT.TFile(outfilename, "RECREATE")
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

if __name__ == '__main__':
    pbpb_datapath  = "/home/CoLBT"
    pp_datapath    = "/home/PP"
    hydro_datapath = "/home/Hydrobackground"
    outfilename    = "/home/EEC_embed_expt.root"
    energy_energy_correlator_pbpbpp(
        pbpb_datapath, pp_datapath, hydro_datapath, outfilename,
        start_file=0, end_file=1999, pp_start=0, pp_end=3999
    )