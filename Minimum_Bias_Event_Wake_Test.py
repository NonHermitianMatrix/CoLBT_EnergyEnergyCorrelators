"""
Uses pre-saved minimum bias events made by Minimum Bias Creation Macro from CoLBT_Minimum_Bias directory structure to see wakes

mb1: Uses m1 directory (matched multiplicity only)
mb2: Uses m2 directory (matched multiplicity only, but different event)
mb3: Uses mixed directory (matched multiplicity only, third different event)

"""

import numpy as np
import pandas as pd
import ROOT
from ROOT import TFile, TH2D, gROOT
import os

from spec_new import NY, NPT, NPHI, Y, PT, PHI, gala15w, INVP, gaulew48
from const import HBARC

gROOT.SetBatch(True)

def map_ang_mpitopi(x):
    """Map angle to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def load_spec_particles(filepath, pt_min=0.0, pt_max=500.0):
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

        # dN = dN/dEta/PtdPt/dPhi * dEta * dPt * dPhi * pt
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

def calc_multiplicity(particles_df):
    """Calculate total multiplicity from particles DataFrame."""
    if particles_df.empty:
        return 0.0
    return particles_df["weight"].sum()

def calc_vn(particles_df, n):
    """Calculate flow coefficient vn."""
    if particles_df.empty:
        return 0.0
    
    phi = particles_df["phi"].to_numpy()
    weight = particles_df["weight"].to_numpy()
    
    qn_x = np.sum(weight * np.cos(n * phi))
    qn_y = np.sum(weight * np.sin(n * phi))
    total_weight = np.sum(weight)
    
    return np.sqrt(qn_x**2 + qn_y**2) / total_weight if total_weight > 0 else 0.0

def calc_v2(particles_df):
    """Calculate v2 flow coefficient."""
    return calc_vn(particles_df, 2)

def calc_event_plane(particles_df):
    """Calculate event plane angle from particles."""
    if particles_df.empty:
        return 0.0
    
    phi = particles_df["phi"].to_numpy()
    weight = particles_df["weight"].to_numpy()
    
    q2_x = np.sum(weight * np.cos(2 * phi))
    q2_y = np.sum(weight * np.sin(2 * phi))
    
    return map_ang_mpitopi(np.arctan2(q2_y, q2_x) / 2.0) if (q2_x != 0 or q2_y != 0) else 0.0

def load_event_properties(filepath):
    """Load event properties including multiplicity, v2, and event plane."""
    try:
        spec_df = load_spec_particles(filepath)
        if spec_df.empty:
            return None
        
        return {
            "multiplicity": calc_multiplicity(spec_df),
            "v2": calc_v2(spec_df),
            "event_plane": calc_event_plane(spec_df)
        }
    except Exception as e:
        print(f"Error loading event properties: {e}")
        return None

def main_analysis(pbpb_datapath, mb_datapath, hydro_datapath, outfilename, start_file, end_file, 
                  pt_min=0.0, pt_max=2.0):
    """Main analysis function using pre-saved MB events."""
    
    # Define histogram parameters
    deta_bins, deta_min, deta_max = 29, -8.0, 8.0
    dphi_bins, dphi_min, dphi_max = 28, -np.pi, np.pi
    
    # Define MB types and their corresponding directories
    mb_types = ["mb1", "mb2", "mb3"]
    mb_dirs = {"mb1": "m1", "mb2": "m2", "mb3": "mixed"}
    
    # Create all histograms
    hist_dict = {}
    
    # Signal and hydro histograms
    hist_dict["h_signal"] = TH2D("h_signal_deta_dphi", 
                                "Signal (spec+newcon) #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                                deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
    hist_dict["h_hydro"] = TH2D("h_hydro_deta_dphi", 
                               "Hydro background #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                               deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
    
    # MB histograms
    for mb_type in mb_types:
        hist_dict[f"h_{mb_type}"] = TH2D(f"h_{mb_type}_deta_dphi", 
                                         f"{mb_type} #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                                         deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
        hist_dict[f"h_signal_minus_{mb_type}"] = TH2D(f"h_signal_minus_{mb_type}_deta_dphi", 
                                                      f"Signal - {mb_type} #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                                                      deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
        hist_dict[f"h_{mb_type}_minus_hydro"] = TH2D(f"h_{mb_type}_minus_hydro_deta_dphi", 
                                                     f"{mb_type} - Hydro #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                                                     deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
    
    # Signal minus hydro
    hist_dict["h_signal_minus_hydro"] = TH2D("h_signal_minus_hydro_deta_dphi", 
                                             "Signal - Hydro #Delta#eta vs #Delta#phi;#Delta#eta;#Delta#phi", 
                                             deta_bins, deta_min, deta_max, dphi_bins, dphi_min, dphi_max)
    
    # Enable Sumw2 for all histograms
    for hist in hist_dict.values():
        hist.Sumw2()
    
    n_events = 0
    n_mb_events = {mb_type: 0 for mb_type in mb_types}

    # Process jet events
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
            if hadjet_df.empty or len(hadjet_df) != 1:
                continue
        except Exception as e:
            print(f"Could not read hadjet.dat for event {evt_idx}: {e}")
            continue

        jet_row = hadjet_df.iloc[0]
        jet_eta = jet_row["eta"]
        jet_phi = map_ang_mpitopi(np.arctan2(jet_row["py"], jet_row["px"]))

        # Check if MB events exist for this signal event
        mb_base_path = os.path.join(mb_datapath, f"event{evt_idx}")
        mb_exists = True
        for mb_type, mb_dir in mb_dirs.items():
            mb_path = os.path.join(mb_base_path, mb_dir)
            if not os.path.exists(mb_path):
                print(f"MB directory {mb_path} not found for event {evt_idx}")
                mb_exists = False
                break
        
        if not mb_exists:
            continue

        # Get signal event properties
        signal_props = load_event_properties(pbpb_path)
        if not signal_props:
            continue
        
        signal_centrality = signal_props["multiplicity"]
        signal_v2 = signal_props["v2"]
        signal_ep = signal_props["event_plane"]

        # Load signal particles (spec + newcon) with event plane rotation
        spec_df = load_spec_particles(pbpb_path, pt_min=pt_min, pt_max=pt_max)
        newcon_df = load_newconrecom_particles(pbpb_path, pt_min=pt_min, pt_max=pt_max)
        signal_df = pd.concat([spec_df, newcon_df], ignore_index=True)

        if not signal_df.empty:
            # Apply event plane rotation
            signal_df["phi_rotated"] = map_ang_mpitopi(signal_df["phi"] - signal_ep)
            jet_phi_rotated = map_ang_mpitopi(jet_phi - signal_ep)
            
            deta_signal = signal_df["eta"].to_numpy() - jet_eta
            dphi_signal = map_ang_mpitopi(signal_df["phi_rotated"].to_numpy() - jet_phi_rotated)
            weights_signal = signal_df["weight"].to_numpy()
            for deta, dphi, w in zip(deta_signal, dphi_signal, weights_signal):
                hist_dict["h_signal"].Fill(deta, dphi, w)

        # Process each MB type
        for mb_type, mb_dir in mb_dirs.items():
            mb_path = os.path.join(mb_base_path, mb_dir)
            
            # Load MB event properties
            mb_props = load_event_properties(mb_path)
            if not mb_props:
                print(f"Could not load properties for {mb_type} at {mb_path}")
                continue
            
            mb_ep = mb_props["event_plane"]
            
            # Load MB particles
            mb_spec_df = load_spec_particles(mb_path, pt_min=pt_min, pt_max=pt_max)
            mb_newcon_df = load_newconrecom_particles(mb_path, pt_min=pt_min, pt_max=pt_max)
            mb_df = pd.concat([mb_spec_df, mb_newcon_df], ignore_index=True)
            
            if not mb_df.empty:
                # Apply event plane rotation
                mb_df["phi_rotated"] = map_ang_mpitopi(mb_df["phi"] - mb_ep)
                mb_jet_phi_rotated = map_ang_mpitopi(jet_phi - mb_ep)
                
                deta_mb = mb_df["eta"].to_numpy() - jet_eta
                dphi_mb = map_ang_mpitopi(mb_df["phi_rotated"].to_numpy() - mb_jet_phi_rotated)
                weights_mb = mb_df["weight"].to_numpy()
                
                for deta, dphi, w in zip(deta_mb, dphi_mb, weights_mb):
                    hist_dict[f"h_{mb_type}"].Fill(deta, dphi, w)
                
                n_mb_events[mb_type] += 1
                print(f"Signal event idx: {evt_idx}, {mb_type} from directory: {mb_path}")

        # Load hydro background (corresponding event is evt_idx % 400)
        hydro_idx = evt_idx % 400
        hydro_path = os.path.join(hydro_datapath, f"event{hydro_idx}")
        hydro_props = load_event_properties(hydro_path)
        hydro_ep = hydro_props["event_plane"] if hydro_props else 0.0
        
        hydro_df = load_spec_particles(hydro_path, pt_min=pt_min, pt_max=pt_max)
        
        if not hydro_df.empty:
            # Apply event plane rotation
            hydro_df["phi_rotated"] = map_ang_mpitopi(hydro_df["phi"] - hydro_ep)
            hydro_jet_phi_rotated = map_ang_mpitopi(jet_phi - hydro_ep)
            
            deta_hydro = hydro_df["eta"].to_numpy() - jet_eta
            dphi_hydro = map_ang_mpitopi(hydro_df["phi_rotated"].to_numpy() - hydro_jet_phi_rotated)
            weights_hydro = hydro_df["weight"].to_numpy()
            
            for deta, dphi, w in zip(deta_hydro, dphi_hydro, weights_hydro):
                hist_dict["h_hydro"].Fill(deta, dphi, w)

        n_events += 1

    print(f"\nProcessed {n_events} events total")
    for mb_type in mb_types:
        print(f"{mb_type} events processed: {n_mb_events[mb_type]}")

    if n_events > 0:
        # Normalize by number of events
        hist_dict["h_signal"].Scale(1.0 / n_events)
        hist_dict["h_hydro"].Scale(1.0 / n_events)
        
        for mb_type in mb_types:
            if n_mb_events[mb_type] > 0:
                hist_dict[f"h_{mb_type}"].Scale(1.0 / n_mb_events[mb_type])

        # Normalize by bin area for density
        deta_bin_width = (deta_max - deta_min) / deta_bins
        dphi_bin_width = (dphi_max - dphi_min) / dphi_bins
        bin_area = deta_bin_width * dphi_bin_width
        
        for hist_name in hist_dict:
            hist_dict[hist_name].Scale(1.0 / bin_area)

    # Create subtracted histograms
    hist_dict["h_signal_minus_hydro"] = hist_dict["h_signal"].Clone("h_signal_minus_hydro")
    hist_dict["h_signal_minus_hydro"].SetTitle("Signal - Hydro background;#Delta#eta;#Delta#phi")
    hist_dict["h_signal_minus_hydro"].Add(hist_dict["h_hydro"], -1.0)
    
    for mb_type in mb_types:
        # Signal minus MB
        hist_dict[f"h_signal_minus_{mb_type}"] = hist_dict["h_signal"].Clone(f"h_signal_minus_{mb_type}")
        hist_dict[f"h_signal_minus_{mb_type}"].SetTitle(f"Signal - {mb_type};#Delta#eta;#Delta#phi")
        hist_dict[f"h_signal_minus_{mb_type}"].Add(hist_dict[f"h_{mb_type}"], -1.0)
        
        # MB minus hydro
        hist_dict[f"h_{mb_type}_minus_hydro"] = hist_dict[f"h_{mb_type}"].Clone(f"h_{mb_type}_minus_hydro")
        hist_dict[f"h_{mb_type}_minus_hydro"].SetTitle(f"{mb_type} - Hydro;#Delta#eta;#Delta#phi")
        hist_dict[f"h_{mb_type}_minus_hydro"].Add(hist_dict["h_hydro"], -1.0)

    # Save all histograms
    outfile = TFile(outfilename, "RECREATE")
    
    # Write 2D histograms
    for hist_name, hist in hist_dict.items():
        hist.Write()
    
    # Create and write projections
    # Signal projections
    hist_dict["h_signal"].ProjectionX("h_signal_projX").Write()
    hist_dict["h_signal"].ProjectionY("h_signal_projY").Write()
    
    # Hydro projections
    hist_dict["h_hydro"].ProjectionX("h_hydro_projX").Write()
    hist_dict["h_hydro"].ProjectionY("h_hydro_projY").Write()
    
    # MB projections
    for mb_type in mb_types:
        hist_dict[f"h_{mb_type}"].ProjectionX(f"h_{mb_type}_projX").Write()
        hist_dict[f"h_{mb_type}"].ProjectionY(f"h_{mb_type}_projY").Write()
        
        hist_dict[f"h_signal_minus_{mb_type}"].ProjectionX(f"h_signal_minus_{mb_type}_projX").Write()
        hist_dict[f"h_signal_minus_{mb_type}"].ProjectionY(f"h_signal_minus_{mb_type}_projY").Write()
        
        hist_dict[f"h_{mb_type}_minus_hydro"].ProjectionX(f"h_{mb_type}_minus_hydro_projX").Write()
        hist_dict[f"h_{mb_type}_minus_hydro"].ProjectionY(f"h_{mb_type}_minus_hydro_projY").Write()
    
    # Subtracted projections
    hist_dict["h_signal_minus_hydro"].ProjectionX("h_signal_minus_hydro_projX").Write()
    hist_dict["h_signal_minus_hydro"].ProjectionY("h_signal_minus_hydro_projY").Write()
    
    outfile.Close()
    print(f"\nHistograms saved to {outfilename}")

if __name__ == '__main__':
    pbpb_datapath = "/home/CoLBT"
    mb_datapath = "./CoLBT_Minimum_Bias"  # Path to pre-saved MB events
    hydro_datapath = "/home/Hydrobackground"
    outfilename = "./min_bias_creation_test.root"
    pt_min = 0.0
    pt_max = 500  # Set your desired pt range here

    main_analysis(
        pbpb_datapath,
        mb_datapath,
        hydro_datapath,
        outfilename,
        start_file=0,
        end_file=3999,
        pt_min=pt_min,
        pt_max=pt_max
    )