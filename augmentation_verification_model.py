"""
This code shows the augmentation of m1 and m2 and how spec - augmented m is closer to 0 than spec - m.
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

def get_multiplicity(particles_df):
    """Calculate multiplicity (sum of weights) for a particle DataFrame."""
    if particles_df.empty:
        return 0
    return particles_df["weight"].sum()

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

        # Load and reshape (from code2)
        spec_data = np.loadtxt(event_charged_dat).reshape(NY, NPT, NPHI) / (HBARC**3.0)

        # Bin widths
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

def find_matching_hydro_events(target_multiplicity, hydro_multiplicities, tolerance=0.1):
    """Find two hydro events with multiplicity within tolerance of target."""
    tol_range = target_multiplicity * tolerance
    min_mult = target_multiplicity - tol_range
    max_mult = target_multiplicity + tol_range
    
    matching_events = []
    for idx, mult in hydro_multiplicities.items():
        if min_mult <= mult <= max_mult:
            matching_events.append(idx)
    
    if len(matching_events) >= 2:
        return random.sample(matching_events, 2)
    elif len(matching_events) == 1:
        # If only one match, use it twice
        return [matching_events[0], matching_events[0]]
    else:
        # If no matches, find the two closest
        sorted_events = sorted(hydro_multiplicities.items(), key=lambda x: abs(x[1] - target_multiplicity))
        return [sorted_events[0][0], sorted_events[1][0]]

def apply_ratio_weights_to_histogram(hist, ratio_hist):
    """Apply ratio weights from ratio_hist to each bin of hist."""
    weighted_hist = hist.Clone(hist.GetName() + "_weighted")
    
    for ix in range(1, hist.GetNbinsX() + 1):
        for iy in range(1, hist.GetNbinsY() + 1):
            original_val = hist.GetBinContent(ix, iy)
            ratio_val = ratio_hist.GetBinContent(ix, iy)
            if ratio_val > 0:
                weighted_hist.SetBinContent(ix, iy, original_val * ratio_val)
            else:
                weighted_hist.SetBinContent(ix, iy, 0.0)
    
    return weighted_hist
def main_analysis(pbpb_datapath, pp_datapath, hydro_datapath, outfilename, start_file, end_file, pp_start, pp_end):
    # --- Pre-calculate hydro multiplicities ---
    print("Pre-calculating hydro event multiplicities...")
    hydro_multiplicities = {}
    for hydro_idx in range(400):
        hydro_path = os.path.join(hydro_datapath, f"event{hydro_idx}")
        hydro_df = load_spec_particles(hydro_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=PT_MIN, pt_max=PT_MAX)
        hydro_multiplicities[hydro_idx] = get_multiplicity(hydro_df)
    print(f"Calculated multiplicities for {len(hydro_multiplicities)} hydro events")

    # --- HISTOGRAM SETUP ---
    n_eta_bins, n_phi_bins = 24, 20
    eta_min, eta_max = -6.0, 6.0
    phi_min, phi_max = -np.pi, np.pi

    # Create one histogram of each kind
    h_signal_spec = ROOT.TH2D("h_signal_spec", "Signal Spec;#Delta#eta;#Delta#phi", n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_hydro      = ROOT.TH2D("h_hydro",      "Hydro;#Delta#eta;#Delta#phi",      n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_m1         = ROOT.TH2D("h_m1",         "M1;#Delta#eta;#Delta#phi",         n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_m2         = ROOT.TH2D("h_m2",         "M2;#Delta#eta;#Delta#phi",         n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_m1_aug     = ROOT.TH2D("h_m1_aug",     "Augmented M1;#Delta#eta;#Delta#phi",n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_m2_aug     = ROOT.TH2D("h_m2_aug",     "Augmented M2;#Delta#eta;#Delta#phi",n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_ratio      = ROOT.TH2D("h_ratio",      "Signal/Hydro;#Delta#eta;#Delta#phi",n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    # New: Difference histograms (unaugmented)
    h_signal_minus_hydro_unaug = ROOT.TH2D("h_signal_minus_hydro_unaug", "Signal Spec - Hydro;#Delta#eta;#Delta#phi", n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_signal_minus_m1 = ROOT.TH2D("h_signal_minus_m1", "Signal Spec - M1;#Delta#eta;#Delta#phi", n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)
    h_signal_minus_m2 = ROOT.TH2D("h_signal_minus_m2", "Signal Spec - M2;#Delta#eta;#Delta#phi", n_eta_bins, eta_min, eta_max, n_phi_bins, phi_min, phi_max)

    # For error propagation
   
    for h in [h_signal_spec, h_hydro, h_m1, h_m2, h_m1_aug, h_m2_aug, h_ratio,
              h_signal_minus_hydro_unaug, h_signal_minus_m1, h_signal_minus_m2]:
        h.Sumw2()

    n_events = 0

    for evt_idx in range(start_file, end_file):
        if evt_idx % 10 == 0:
            print(f"Processing event {evt_idx}")

        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        try:
            hadjet_df = pd.read_csv(os.path.join(pbpb_path, "hadjet.dat"), sep='\s+', 
                                   names=["jet_no","px","py","pz","E","eta"], engine='python')
            if hadjet_df.empty: continue
        except Exception: continue

        jet_row = hadjet_df.iloc[0]
        jet_eta = jet_row["eta"]
        jet_phi = map_ang_mpitopi(np.arctan2(jet_row["py"], jet_row["px"]))

        # Load all needed particle sets
        signal_spec_df = load_spec_particles(pbpb_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=PT_MIN, pt_max=PT_MAX)
        hydro_idx = evt_idx % 400
        hydro_path = os.path.join(hydro_datapath, f"event{hydro_idx}")
        hydro_df = load_spec_particles(hydro_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=PT_MIN, pt_max=PT_MAX)

        hydro_mult = hydro_multiplicities[hydro_idx]
        m1_idx, m2_idx = find_matching_hydro_events(hydro_mult, hydro_multiplicities)
        m1_path = os.path.join(hydro_datapath, f"event{m1_idx}")
        m2_path = os.path.join(hydro_datapath, f"event{m2_idx}")
        m1_df = load_spec_particles(m1_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=PT_MIN, pt_max=PT_MAX)
        m2_df = load_spec_particles(m2_path, jet_eta=None, jet_phi=None, cone_radius=None, pt_min=PT_MIN, pt_max=PT_MAX)

        # Fill histograms for this event
        if not signal_spec_df.empty:
            deta = (signal_spec_df["eta"] - jet_eta).to_numpy()
            dphi = map_ang_mpitopi(signal_spec_df["phi"] - jet_phi).to_numpy()
            weights = signal_spec_df["weight"].to_numpy()
            h_signal_spec.FillN(len(signal_spec_df), deta, dphi, weights)
        if not hydro_df.empty:
            deta = (hydro_df["eta"] - jet_eta).to_numpy()
            dphi = map_ang_mpitopi(hydro_df["phi"] - jet_phi).to_numpy()
            weights = hydro_df["weight"].to_numpy()
            h_hydro.FillN(len(hydro_df), deta, dphi, weights)
        if not m1_df.empty:
            deta = (m1_df["eta"] - jet_eta).to_numpy()
            dphi = map_ang_mpitopi(m1_df["phi"] - jet_phi).to_numpy()
            weights = m1_df["weight"].to_numpy()
            h_m1.FillN(len(m1_df), deta, dphi, weights)
        if not m2_df.empty:
            deta = (m2_df["eta"] - jet_eta).to_numpy()
            dphi = map_ang_mpitopi(m2_df["phi"] - jet_phi).to_numpy()
            weights = m2_df["weight"].to_numpy()
            h_m2.FillN(len(m2_df), deta, dphi, weights)

        n_events += 1

    print(f"Processed {n_events} events. Normalizing histograms...")

    # Normalize by number of events
    if n_events > 0:
        h_signal_spec.Scale(1.0 / n_events)
        h_hydro.Scale(1.0 / n_events)
        h_m1.Scale(1.0 / n_events)
        h_m2.Scale(1.0 / n_events)

    # Ratio and augmented backgrounds
    h_ratio.Divide(h_signal_spec, h_hydro, 1, 1, "B")  # "B" for binomial errors

    # Augment M1 and M2
    for ix in range(1, n_eta_bins+1):
        for iy in range(1, n_phi_bins+1):
            m1val = h_m1.GetBinContent(ix, iy)
            m2val = h_m2.GetBinContent(ix, iy)
            ratio = h_ratio.GetBinContent(ix, iy)
            h_m1_aug.SetBinContent(ix, iy, m1val * ratio)
            h_m2_aug.SetBinContent(ix, iy, m2val * ratio)

    # Differences
    h_signal_minus_hydro = h_signal_spec.Clone("h_signal_minus_hydro")
    h_signal_minus_hydro.Add(h_hydro, -1.0)
    h_signal_minus_m1_aug = h_signal_spec.Clone("h_signal_minus_m1_aug")
    h_signal_minus_m1_aug.Add(h_m1_aug, -1.0)
    h_signal_minus_m2_aug = h_signal_spec.Clone("h_signal_minus_m2_aug")
    h_signal_minus_m2_aug.Add(h_m2_aug, -1.0)
    # Differences with unaugmented backgrounds
    h_signal_minus_hydro_unaug = h_signal_spec.Clone("h_signal_minus_hydro_unaug")
    h_signal_minus_hydro_unaug.Add(h_hydro, -1.0)
    h_signal_minus_m1 = h_signal_spec.Clone("h_signal_minus_m1")
    h_signal_minus_m1.Add(h_m1, -1.0)
    h_signal_minus_m2 = h_signal_spec.Clone("h_signal_minus_m2")
    h_signal_minus_m2.Add(h_m2, -1.0)
    # Projections
    h_signal_spec_projX = h_signal_spec.ProjectionX("h_signal_spec_projX")
    h_signal_minus_m1_aug_projX = h_signal_minus_m1_aug.ProjectionX("h_signal_minus_m1_aug_projX")
    h_signal_minus_m2_aug_projX = h_signal_minus_m2_aug.ProjectionX("h_signal_minus_m2_aug_projX")
    h_signal_spec_projY = h_signal_spec.ProjectionY("h_signal_spec_projY")
    h_signal_minus_m1_aug_projY = h_signal_minus_m1_aug.ProjectionY("h_signal_minus_m1_aug_projY")
    h_signal_minus_m2_aug_projY = h_signal_minus_m2_aug.ProjectionY("h_signal_minus_m2_aug_projY")
    # New: Projections for unaugmented differences
    h_signal_minus_hydro_unaug_projX = h_signal_minus_hydro_unaug.ProjectionX("h_signal_minus_hydro_unaug_projX")
    h_signal_minus_m1_projX = h_signal_minus_m1.ProjectionX("h_signal_minus_m1_projX")
    h_signal_minus_m2_projX = h_signal_minus_m2.ProjectionX("h_signal_minus_m2_projX")
    h_signal_minus_hydro_unaug_projY = h_signal_minus_hydro_unaug.ProjectionY("h_signal_minus_hydro_unaug_projY")
    h_signal_minus_m1_projY = h_signal_minus_m1.ProjectionY("h_signal_minus_m1_projY")
    h_signal_minus_m2_projY = h_signal_minus_m2.ProjectionY("h_signal_minus_m2_projY")
    # Save to file
    outfile = ROOT.TFile(outfilename, "RECREATE")
    h_signal_spec.Write()
    h_hydro.Write()
    h_m1.Write()
    h_m2.Write()
    h_m1_aug.Write()
    h_m2_aug.Write()
    h_ratio.Write()
    h_signal_minus_hydro.Write()
    h_signal_minus_m1_aug.Write()
    h_signal_minus_m2_aug.Write()
    h_signal_spec_projX.Write()
    h_signal_minus_m1_aug_projX.Write()
    h_signal_minus_m2_aug_projX.Write()
    h_signal_spec_projY.Write()
    h_signal_minus_m1_aug_projY.Write()
    h_signal_minus_m2_aug_projY.Write()
    # New: Write unaugmented difference histograms and projections
    h_signal_minus_hydro_unaug.Write()
    h_signal_minus_m1.Write()
    h_signal_minus_m2.Write()
    h_signal_minus_hydro_unaug_projX.Write()
    h_signal_minus_m1_projX.Write()
    h_signal_minus_m2_projX.Write()
    h_signal_minus_hydro_unaug_projY.Write()
    h_signal_minus_m1_projY.Write()
    h_signal_minus_m2_projY.Write()
    # === CANVAS PLOTTING ===
    # Set ROOT style
    ROOT.gStyle.SetOptStat(0)

    # --- Helper for line style ---
    def set_hist_style(hist, color, style=1, width=2):
        hist.SetLineColor(color)
        hist.SetLineStyle(style)
        hist.SetLineWidth(width)

    # --- 1. signal_spec, m1_aug, m1 (ProjectionX) ---
    c_m1 = ROOT.TCanvas("c_m1", "Signal vs M1", 800, 600)
    set_hist_style(h_signal_spec_projX, ROOT.kBlack, 1, 2)
    set_hist_style(h_m1_aug.ProjectionX("h_m1_aug_projX_draw"), ROOT.kRed, 2, 2)
    set_hist_style(h_m1.ProjectionX("h_m1_projX_draw"), ROOT.kBlue, 1, 2)

    h_signal_spec_projX.SetTitle("Signal Spec vs M1;#Delta#eta;Yield")
    h_signal_spec_projX.Draw("HIST")
    h_m1_aug.ProjectionX("h_m1_aug_projX_draw").Draw("HIST SAME")
    h_m1.ProjectionX("h_m1_projX_draw").Draw("HIST SAME")

    leg1 = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg1.AddEntry(h_signal_spec_projX, "Signal Spec", "l")
    leg1.AddEntry(h_m1_aug.ProjectionX("h_m1_aug_projX_draw"), "M1 Augmented", "l")
    leg1.AddEntry(h_m1.ProjectionX("h_m1_projX_draw"), "M1 Unaugmented", "l")
    leg1.Draw()
    c_m1.Write()

    # --- 2. signal_spec, m2_aug, m2 (ProjectionX) ---
    c_m2 = ROOT.TCanvas("c_m2", "Signal vs M2", 800, 600)
    set_hist_style(h_signal_spec_projX, ROOT.kBlack, 1, 2)
    set_hist_style(h_m2_aug.ProjectionX("h_m2_aug_projX_draw"), ROOT.kRed, 2, 2)
    set_hist_style(h_m2.ProjectionX("h_m2_projX_draw"), ROOT.kBlue, 1, 2)

    h_signal_spec_projX.Draw("HIST")
    h_m2_aug.ProjectionX("h_m2_aug_projX_draw").Draw("HIST SAME")
    h_m2.ProjectionX("h_m2_projX_draw").Draw("HIST SAME")

    leg2 = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg2.AddEntry(h_signal_spec_projX, "Signal Spec", "l")
    leg2.AddEntry(h_m2_aug.ProjectionX("h_m2_aug_projX_draw"), "M2 Augmented", "l")
    leg2.AddEntry(h_m2.ProjectionX("h_m2_projX_draw"), "M2 Unaugmented", "l")
    leg2.Draw()
    c_m2.Write()

    # --- 3. signal_spec, hydro, hydro unaugmented (ProjectionX) ---
    c_hydro = ROOT.TCanvas("c_hydro", "Signal vs Hydro", 800, 600)
    set_hist_style(h_signal_spec_projX, ROOT.kBlack, 1, 2)
    set_hist_style(h_hydro.ProjectionX("h_hydro_projX_draw"), ROOT.kBlue, 1, 2)
    set_hist_style(h_hydro.ProjectionX("h_hydro_projX_draw"), ROOT.kRed, 2, 2)  # No augmented hydro, so just plot twice for style

    h_signal_spec_projX.Draw("HIST")
    h_hydro.ProjectionX("h_hydro_projX_draw").Draw("HIST SAME")

    leg3 = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg3.AddEntry(h_signal_spec_projX, "Signal Spec", "l")
    leg3.AddEntry(h_hydro.ProjectionX("h_hydro_projX_draw"), "Hydrobackground", "l")
    leg3.Draw()
    c_hydro.Write()

    # --- 4. Differences: (spec - m1), (spec - m1_aug) ---
    c_diff_m1 = ROOT.TCanvas("c_diff_m1", "Spec - M1", 800, 600)
    set_hist_style(h_signal_minus_m1_projX, ROOT.kBlue, 1, 2)
    set_hist_style(h_signal_minus_m1_aug_projX, ROOT.kRed, 2, 2)
    h_signal_minus_m1_projX.SetTitle("Signal Spec - M1;#Delta#eta;Yield")
    h_signal_minus_m1_projX.Draw("HIST")
    h_signal_minus_m1_aug_projX.Draw("HIST SAME")
    leg4 = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg4.AddEntry(h_signal_minus_m1_projX, "Spec - M1", "l")
    leg4.AddEntry(h_signal_minus_m1_aug_projX, "Spec - M1 Augmented", "l")
    leg4.Draw()
    c_diff_m1.Write()

    # --- 5. Differences: (spec - m2), (spec - m2_aug) ---
    c_diff_m2 = ROOT.TCanvas("c_diff_m2", "Spec - M2", 800, 600)
    set_hist_style(h_signal_minus_m2_projX, ROOT.kBlue, 1, 2)
    set_hist_style(h_signal_minus_m2_aug_projX, ROOT.kRed, 2, 2)
    h_signal_minus_m2_projX.SetTitle("Signal Spec - M2;#Delta#eta;Yield")
    h_signal_minus_m2_projX.Draw("HIST")
    h_signal_minus_m2_aug_projX.Draw("HIST SAME")
    leg5 = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg5.AddEntry(h_signal_minus_m2_projX, "Spec - M2", "l")
    leg5.AddEntry(h_signal_minus_m2_aug_projX, "Spec - M2 Augmented", "l")
    leg5.Draw()
    c_diff_m2.Write()
    # === CANVAS PLOTTING FOR DELTA PHI (Y PROJECTIONS) ===

    # --- 1. signal_spec, m1_aug, m1 (ProjectionY) ---
    c_m1_phi = ROOT.TCanvas("c_m1_phi", "Signal vs M1 (Delta Phi)", 800, 600)
    set_hist_style(h_signal_spec_projY, ROOT.kBlack, 1, 2)
    set_hist_style(h_m1_aug.ProjectionY("h_m1_aug_projY_draw"), ROOT.kRed, 2, 2)
    set_hist_style(h_m1.ProjectionY("h_m1_projY_draw"), ROOT.kBlue, 1, 2)

    h_signal_spec_projY.SetTitle("Signal Spec vs M1;#Delta#phi;Yield")
    h_signal_spec_projY.Draw("HIST")
    h_m1_aug.ProjectionY("h_m1_aug_projY_draw").Draw("HIST SAME")
    h_m1.ProjectionY("h_m1_projY_draw").Draw("HIST SAME")

    leg1_phi = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg1_phi.AddEntry(h_signal_spec_projY, "Signal Spec", "l")
    leg1_phi.AddEntry(h_m1_aug.ProjectionY("h_m1_aug_projY_draw"), "M1 Augmented", "l")
    leg1_phi.AddEntry(h_m1.ProjectionY("h_m1_projY_draw"), "M1 Unaugmented", "l")
    leg1_phi.Draw()
    c_m1_phi.Write()

    # --- 2. signal_spec, m2_aug, m2 (ProjectionY) ---
    c_m2_phi = ROOT.TCanvas("c_m2_phi", "Signal vs M2 (Delta Phi)", 800, 600)
    set_hist_style(h_signal_spec_projY, ROOT.kBlack, 1, 2)
    set_hist_style(h_m2_aug.ProjectionY("h_m2_aug_projY_draw"), ROOT.kRed, 2, 2)
    set_hist_style(h_m2.ProjectionY("h_m2_projY_draw"), ROOT.kBlue, 1, 2)

    h_signal_spec_projY.Draw("HIST")
    h_m2_aug.ProjectionY("h_m2_aug_projY_draw").Draw("HIST SAME")
    h_m2.ProjectionY("h_m2_projY_draw").Draw("HIST SAME")

    leg2_phi = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg2_phi.AddEntry(h_signal_spec_projY, "Signal Spec", "l")
    leg2_phi.AddEntry(h_m2_aug.ProjectionY("h_m2_aug_projY_draw"), "M2 Augmented", "l")
    leg2_phi.AddEntry(h_m2.ProjectionY("h_m2_projY_draw"), "M2 Unaugmented", "l")
    leg2_phi.Draw()
    c_m2_phi.Write()

    # --- 3. signal_spec, hydro, hydro unaugmented (ProjectionY) ---
    c_hydro_phi = ROOT.TCanvas("c_hydro_phi", "Signal vs Hydro (Delta Phi)", 800, 600)
    set_hist_style(h_signal_spec_projY, ROOT.kBlack, 1, 2)
    set_hist_style(h_hydro.ProjectionY("h_hydro_projY_draw"), ROOT.kBlue, 1, 2)
    set_hist_style(h_hydro.ProjectionY("h_hydro_projY_draw"), ROOT.kRed, 2, 2)  # No augmented hydro, so just plot twice for style

    h_signal_spec_projY.Draw("HIST")
    h_hydro.ProjectionY("h_hydro_projY_draw").Draw("HIST SAME")

    leg3_phi = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg3_phi.AddEntry(h_signal_spec_projY, "Signal Spec", "l")
    leg3_phi.AddEntry(h_hydro.ProjectionY("h_hydro_projY_draw"), "Hydrobackground", "l")
    leg3_phi.Draw()
    c_hydro_phi.Write()

    # --- 4. Differences: (spec - m1), (spec - m1_aug) (ProjectionY) ---
    c_diff_m1_phi = ROOT.TCanvas("c_diff_m1_phi", "Spec - M1 (Delta Phi)", 800, 600)
    set_hist_style(h_signal_minus_m1_projY, ROOT.kBlue, 1, 2)
    set_hist_style(h_signal_minus_m1_aug_projY, ROOT.kRed, 2, 2)
    h_signal_minus_m1_projY.SetTitle("Signal Spec - M1;#Delta#phi;Yield")
    h_signal_minus_m1_projY.Draw("HIST")
    h_signal_minus_m1_aug_projY.Draw("HIST SAME")
    leg4_phi = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg4_phi.AddEntry(h_signal_minus_m1_projY, "Spec - M1", "l")
    leg4_phi.AddEntry(h_signal_minus_m1_aug_projY, "U.E. - M1 Augmented", "l")
    leg4_phi.Draw()
    c_diff_m1_phi.Write()

    # --- 5. Differences: (U.E. - m2), (U.E. - m2_aug) (ProjectionY) ---
    c_diff_m2_phi = ROOT.TCanvas("c_diff_m2_phi", "U.E. - M2 (Delta Phi)", 800, 600)
    set_hist_style(h_signal_minus_m2_projY, ROOT.kBlue, 1, 2)
    set_hist_style(h_signal_minus_m2_aug_projY, ROOT.kRed, 2, 2)
    h_signal_minus_m2_projY.SetTitle("Signal U.E. - M2;#Delta#phi;Yield")
    h_signal_minus_m2_projY.Draw("HIST")
    h_signal_minus_m2_aug_projY.Draw("HIST SAME")
    leg5_phi = ROOT.TLegend(0.6,0.7,0.88,0.88)
    leg5_phi.AddEntry(h_signal_minus_m2_projY, "U.E. - M2", "l")
    leg5_phi.AddEntry(h_signal_minus_m2_aug_projY, "U.E. - M2 Augmented", "l")
    leg5_phi.Draw()
    c_diff_m2_phi.Write()

    
    outfile.Close()
    print(f"Analysis complete. Results saved to {outfilename}")
    
if __name__ == '__main__':
    pbpb_datapath  = "/home/CoLBT"
    pp_datapath    = "/home/PP"
    hydro_datapath = "/home/Hydrobackground"
    outfilename    = "/home/eec_embed_augmented_verification.root"
    
    main_analysis(
        pbpb_datapath, pp_datapath, hydro_datapath, outfilename, 
        start_file=0, end_file=4999, pp_start=0, pp_end=6999   
    )