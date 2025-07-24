"""
Run this to ensure data is being read correctly, produces fig, 3(b) of https://arxiv.org/abs/2203.03683, project "fine" histograms with appropriate cuts and binning to get fig 4(a) of the same paper.
"""

import numpy as np
import pandas as pd
import ROOT
from ROOT import TFile, TH2D, gROOT
import os

# Import constants and binning
from spec_new import NY, NPT, NPHI, Y, PT, PHI, gala15w, INVP, gaulew48
from const import HBARC

gROOT.SetBatch(True)

def map_ang_mpitopi(x):
    """Map angle to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def load_spec_particles(filepath, pt_min=0.0, pt_max=2.0):
    """Load spec particles from dNdEtaPtdPtdPhi_Charged.dat with correct weights and pt filter."""
    try:
        event_charged_dat = os.path.join(filepath, "dNdEtaPtdPtdPhi_Charged.dat")
        if not os.path.exists(event_charged_dat):
            print(f"dNdEtaPtdPtdPhi_Charged.dat not found in {filepath}")
            return pd.DataFrame()

        # Load and reshape
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

        # dN = dN/dEta/dPt/dPhi * dEta * dPt * dPhi * pt
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

        return particles_df

    except Exception as e:
        print(f"Error processing spec data: {e}")
        return pd.DataFrame()
    
def load_newconrecom_particles(filepath, pt_min=0.0, pt_max=2.0):
    """Load charged particles from NewConRecom.dat, with pt filter."""
    try:
        newcon_path = os.path.join(filepath, "NewConRecom.dat")
        if not os.path.exists(newcon_path):
            return pd.DataFrame()

        newcon_df = pd.read_csv(newcon_path, sep='\s+', 
                               names=["pid","px","py","pz","E","mass","x","y","z","t"], 
                               engine='python')
        if newcon_df.empty: 
            return pd.DataFrame()

        newcon_df["pt"] = np.sqrt(newcon_df["px"]**2 + newcon_df["py"]**2)
        newcon_df["p"] = np.sqrt(newcon_df["pt"]**2 + newcon_df["pz"]**2)
        newcon_df["phi"] = map_ang_mpitopi(np.arctan2(newcon_df["py"], newcon_df["px"]))

        p_minus_pz = newcon_df["p"] - newcon_df["pz"]
        valid_eta = p_minus_pz > 1e-9
        newcon_df["eta"] = np.nan
        newcon_df.loc[valid_eta, "eta"] = 0.5 * np.log((newcon_df.loc[valid_eta, "p"] + 
                                                         newcon_df.loc[valid_eta, "pz"]) / 
                                                        p_minus_pz[valid_eta])
        newcon_df.dropna(subset=['eta'], inplace=True)

        charged_pids = [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13]
        charged_particles_df = newcon_df[np.abs(newcon_df["pid"]).isin(charged_pids)].copy()
        charged_particles_df = charged_particles_df[
            (charged_particles_df["pt"] >= pt_min) & (charged_particles_df["pt"] <= pt_max)
        ]
        charged_particles_df["weight"] = 1.0
        charged_particles_df["eta_width"] = 1.0  # Uniform for reco particles
        charged_particles_df["phi_width"] = 1.0  # Uniform for reco particles
        final_cols = ["pt", "eta", "phi", "weight", "eta_width", "phi_width"]
        return charged_particles_df[final_cols]

    except Exception as e:
        print(f"Error processing NewConRecom.dat: {e}")
        return pd.DataFrame()

def main_analysis(pbpb_datapath, hydro_datapath, outfilename, start_file, end_file, pt_min=0.0, pt_max=2.0):
    """Main analysis function with normalization by eta and phi bin width."""
    deta_bins, deta_min, deta_max = 24, -6.0, 6.0
    dphi_bins, dphi_min, dphi_max = 20, -np.pi, np.pi

    h_signal = TH2D("h_signal_deta_dphi", 
                    "Signal (spec+newcon) #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                    deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
    h_signal.Sumw2()
    h_hydro = TH2D("h_hydro_deta_dphi", 
                   "Hydro background #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                   deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
    h_hydro.Sumw2()

    n_events = 0

    for evt_idx in range(start_file, end_file):
        if evt_idx % 1 == 0:
            print(f"\nProcessing event {evt_idx}/{end_file-1}")

        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        if not os.path.isdir(pbpb_path):
            print(f"PbPb event directory not found for event {evt_idx}. Skipping.")
            continue

        # Get jet information
        try:
            hadjet_df = pd.read_csv(os.path.join(pbpb_path, "hadjet.dat"), 
                                   sep='\s+', 
                                   names=["jet_no","px","py","pz","E","eta"], 
                                   engine='python')
            if hadjet_df.empty:
                continue
        except Exception as e:
            print(f"Could not read hadjet.dat for event {evt_idx}: {e}")
            continue

        jet_row = hadjet_df.iloc[0]
        jet_eta = jet_row["eta"]
        jet_phi = map_ang_mpitopi(np.arctan2(jet_row["py"], jet_row["px"]))

        # Load signal particles (spec + newcon)
        spec_df = load_spec_particles(pbpb_path, pt_min=pt_min, pt_max=pt_max)
        newcon_df = load_newconrecom_particles(pbpb_path, pt_min=pt_min, pt_max=pt_max)
        signal_df = pd.concat([spec_df, newcon_df], ignore_index=True)

        if not signal_df.empty:
            deta_signal = signal_df["eta"].to_numpy() - jet_eta
            dphi_signal = map_ang_mpitopi(signal_df["phi"].to_numpy() - jet_phi)
            weights_signal = signal_df["weight"].to_numpy()
            for deta, dphi, w in zip(deta_signal, dphi_signal, weights_signal):
                h_signal.Fill(deta, dphi, w)

        # Load hydro background (corresponding event is evt_idx % 400)
        hydro_idx = evt_idx % 400
        hydro_path = os.path.join(hydro_datapath, f"event{hydro_idx}")
        hydro_df = load_spec_particles(hydro_path, pt_min=pt_min, pt_max=pt_max)

        if not hydro_df.empty:
            deta_hydro = hydro_df["eta"].to_numpy() - jet_eta
            dphi_hydro = map_ang_mpitopi(hydro_df["phi"].to_numpy() - jet_phi)
            weights_hydro = hydro_df["weight"].to_numpy()
            for deta, dphi, w in zip(deta_hydro, dphi_hydro, weights_hydro):
                h_hydro.Fill(deta, dphi, w)

        n_events += 1

    print(f"\nProcessed {n_events} events total")

    if n_events > 0:
        h_signal.Scale(1.0 / n_events)
        h_hydro.Scale(1.0 / n_events)

        # Optional: Normalize by bin area for density
        deta_bin_width = (deta_max - deta_min) / deta_bins
        dphi_bin_width = (dphi_max - dphi_min) / dphi_bins
        bin_area = deta_bin_width * dphi_bin_width
        h_signal.Scale(1.0 / bin_area)
        h_hydro.Scale(1.0 / bin_area)

    h_subtracted = h_signal.Clone("h_signal_minus_hydro")
    h_subtracted.SetTitle("Signal - Hydro background;#Delta#eta;#Delta#phi")
    h_subtracted.Add(h_hydro, -1.0)

    outfile = TFile(outfilename, "RECREATE")
    h_signal.Write()
    h_hydro.Write()
    h_subtracted.Write()
    h_signal.ProjectionX("h_signal_projX").Write()
    h_signal.ProjectionY("h_signal_projY").Write()
    h_hydro.ProjectionX("h_hydro_projX").Write()
    h_hydro.ProjectionY("h_hydro_projY").Write()
    h_subtracted.ProjectionX("h_subtracted_projX").Write()
    h_subtracted.ProjectionY("h_subtracted_projY").Write()
    outfile.Close()
    print(f"\nHistograms saved to {outfilename}")

if __name__ == '__main__':
    pbpb_datapath = "/home/CoLBT"
    hydro_datapath = "/home/Hydrobackground"
    outfilename = "./wake_check.root"
    pt_min = 0.0
    pt_max = 2.0  # Set your desired pt range here

    main_analysis(
        pbpb_datapath,
        hydro_datapath,
        outfilename,
        start_file=0,
        end_file=5999,
        pt_min=pt_min,
        pt_max=pt_max
    )
