# Event classes:
#   - signal:      Jet events (CoLBT)
#   - hydro:       Hydrodynamic background events
#   - mb:          All minimum bias events (no jets)
#   - mb_m:        Minimum bias events matched in multiplicity to signal
# This code produces the following types of plots for each event class (signal, hydro, mb, mb_m):
#
# 1. v_n distributions (n=1..4):         # TH1D, event-by-event v_n values
# 2. <cos(n(phi-Psi_n))> distributions:  # TH1D, event-by-event average cosines
# 3. <cos(n(phi-Psi_n))> vs (eta, pt):   # TProfile2D, average cosine in (eta, pt) bins
#
# Additionally, for each event class:
#   - "spec_only" plots: using only spec particles
#   - "newcon_only" plots: using only newcon particles
#

import numpy as np
import pandas as pd
import ROOT
import os
from spec_new import *  # Must contain NY, NPT, NPHI, Y, PT, PHI, gala15w, gaulew48
from const import *     # Must contain HBARC, INVP


def map_ang_mpitopi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def load_spec_particles(filepath):
    particles = []
    event_charged_dat = os.path.join(filepath, "dNdEtaPtdPtdPhi_Charged.dat")
    if not os.path.exists(event_charged_dat):
        return particles
    try:
        spec_data = np.loadtxt(event_charged_dat).reshape(NY, NPT, NPHI) / (HBARC**3.0)
        eta, pt, phi = Y, PT, map_ang_mpitopi(PHI)
        eta_width = np.zeros(NY)
        eta_width[0] = (Y[1] - Y[0]) / 2.0
        eta_width[-1] = (Y[-1] - Y[-2]) / 2.0
        eta_width[1:-1] = (Y[2:] - Y[:-2]) / 2.0
        pt_width = gala15w * INVP
        phi_width = np.pi * np.concatenate((gaulew48, gaulew48[::-1]))
        
        particle_count = (spec_data * 
                         eta_width[:, np.newaxis, np.newaxis] * 
                         pt_width[np.newaxis, :, np.newaxis] * 
                         phi_width[np.newaxis, np.newaxis, :] * 
                         pt[np.newaxis, :, np.newaxis])  
        
        mask = (particle_count > 0)
        idxs = np.where(mask)
        for i, j, k in zip(*idxs):
            particles.append({"pt": pt[j], "eta": eta[i], "phi": phi[k], "weight": particle_count[i, j, k], "source": "spec"})
    except Exception as e:
        print(f"Error processing dNdEtaPtdPtdPhi_Charged.dat in {filepath}: {e}")
    return particles

def load_newconrecom_particles(filepath):
    particles = []
    newcon_path = os.path.join(filepath, "NewConRecom.dat")
    if not os.path.exists(newcon_path): return particles
    try:
        df = pd.read_csv(newcon_path, sep='\s+', names=["pid","px","py","pz","E","mass","x","y","z","t"], engine='python')
        if df.empty: return particles
        px, py, pz, pid = df["px"].values, df["py"].values, df["pz"].values, df["pid"].values
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        phi = map_ang_mpitopi(np.arctan2(py, px))
        eta = np.zeros_like(p)
        p_plus_pz = p + pz
        p_minus_pz = p - pz
        valid_mask = (p_plus_pz > 1e-9) & (p_minus_pz > 1e-9)
        eta[valid_mask] = 0.5 * np.log(p_plus_pz[valid_mask] / p_minus_pz[valid_mask])
        abs_pid = np.abs(pid)
        charged_mask = np.isin(abs_pid, [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13])
        eta_c, phi_c, pt_c = eta[charged_mask], phi[charged_mask], pt[charged_mask]
        for e, p_, f in zip(eta_c, pt_c, phi_c):
            particles.append({"pt": p_, "eta": e, "phi": f, "source": "newcon", "weight": 1.0})
    except Exception as e:
        print(f"Error processing NewConRecom.dat in {filepath}: {e}")
    return particles

def load_all_particles(filepath):
    return load_spec_particles(filepath) + load_newconrecom_particles(filepath)

# ==============================================================================
# 2. EVENT PROPERTY CALCULATION FUNCTIONS (Modified)
# ==============================================================================

def calc_multiplicity(particles):
    return sum(p.get("weight", 1.0) for p in particles)

def calc_vn(particles, n):
    spec_particles = [p for p in particles if p.get("source") == "spec"]
    if not spec_particles: return 0.0
    phi = np.array([p["phi"] for p in spec_particles])
    weight = np.array([p.get("weight", 1.0) for p in spec_particles])
    qn_x = np.sum(weight * np.cos(n * phi))
    qn_y = np.sum(weight * np.sin(n * phi))
    total_weight = np.sum(weight)
    return np.sqrt(qn_x**2 + qn_y**2) / total_weight if total_weight > 0 else 0.0

def calc_event_plane_n(particles, n):
    spec_particles = [p for p in particles if p.get("source") == "spec"]
    if not spec_particles: return 0.0
    phi = np.array([p["phi"] for p in spec_particles])
    weight = np.array([p.get("weight", 1.0) for p in spec_particles])
    qn_x = np.sum(weight * np.cos(n * phi))
    qn_y = np.sum(weight * np.sin(n * phi))
    return map_ang_mpitopi(np.arctan2(qn_y, qn_x) / n) if (qn_x != 0 or qn_y != 0) else 0.0
def calc_vn_filtered(particles, n, source_filter):
    """Calculate vn using only particles from specified source."""
    filtered_particles = [p for p in particles if p.get("source") == source_filter]
    if not filtered_particles: return 0.0
    phi = np.array([p["phi"] for p in filtered_particles])
    weight = np.array([p.get("weight", 1.0) for p in filtered_particles])
    qn_x = np.sum(weight * np.cos(n * phi))
    qn_y = np.sum(weight * np.sin(n * phi))
    total_weight = np.sum(weight)
    return np.sqrt(qn_x**2 + qn_y**2) / total_weight if total_weight > 0 else 0.0

def calc_event_plane_n_filtered(particles, n, source_filter):
    """Calculate event plane using only particles from specified source."""
    filtered_particles = [p for p in particles if p.get("source") == source_filter]
    if not filtered_particles: return 0.0
    phi = np.array([p["phi"] for p in filtered_particles])
    weight = np.array([p.get("weight", 1.0) for p in filtered_particles])
    qn_x = np.sum(weight * np.cos(n * phi))
    qn_y = np.sum(weight * np.sin(n * phi))
    return map_ang_mpitopi(np.arctan2(qn_y, qn_x) / n) if (qn_x != 0 or qn_y != 0) else 0.0

def fill_filtered_histograms(hist_dict, event_type, all_particles, source_filter):
    """Fill histograms using only particles from specified source."""
    if not all_particles: return
    
    filtered_particles = [p for p in all_particles if p.get("source") == source_filter]
    if not filtered_particles: return
    
    psi_n_list = [calc_event_plane_n_filtered(all_particles, n, source_filter) for n in range(1, 5)]
    
    for n in range(1, 5):
        vn = calc_vn_filtered(all_particles, n, source_filter)
        hist_dict[f"{event_type}_{source_filter}_v{n}"].Fill(vn)
        
        cosn_val = np.average(
            np.cos(n * map_ang_mpitopi(np.array([p["phi"] for p in filtered_particles]) - psi_n_list[n-1])), 
            weights=np.array([p.get("weight", 1.0) for p in filtered_particles])
        )
        hist_dict[f"{event_type}_{source_filter}_cosn{n}"].Fill(cosn_val)
        
        prof = hist_dict[f"{event_type}_{source_filter}_cosn{n}_etapt"]
        psi_n = psi_n_list[n-1]
        for p in filtered_particles:
            val_to_fill = np.cos(n * map_ang_mpitopi(p["phi"] - psi_n))
            prof.Fill(p["eta"], p["pt"], val_to_fill, p.get("weight", 1.0))
def load_and_calc_event_properties(filepath):
    """Loads all particles and calculates a dictionary of properties including multiplicity."""
    try:
        all_particles = load_all_particles(filepath)
        if all_particles:
            return {
                "multiplicity": calc_multiplicity(all_particles),
                "v1": calc_vn(all_particles, 1),
                "v2": calc_vn(all_particles, 2),
                "v3": calc_vn(all_particles, 3),
                "v4": calc_vn(all_particles, 4),
                "all_particles": all_particles
            }
    except Exception as e:
        print(f"Error processing event properties for {filepath}: {e}")
    return None

# ==============================================================================
# 3. HISTOGRAM AND PROFILE MANAGEMENT (Modified for mb_m)
# ==============================================================================
def make_histograms_and_profiles():
    """Creates and returns a dictionary of all required ROOT objects, including for mb_m."""
    hist_dict = {}
    # Added "mb_m" for matched minimum bias
    event_types = ["signal", "hydro", "mb", "mb_m"]
    eta_bins, eta_min, eta_max = 50, -3, 3
    pt_bins, pt_min, pt_max = 300, 0, 150

    for event_type in event_types:
        for n in range(1, 5):
            hist_dict[f"{event_type}_v{n}"] = ROOT.TH1D(f"{event_type}_v{n}", f"{event_type.upper()} Event v_{n};v_{n};Number of Events", 100, 0, 0.2)
            hist_dict[f"{event_type}_cosn{n}"] = ROOT.TH1D(f"{event_type}_cosn{n}", f"{event_type.upper()} <cos({n}(#phi-#Psi_{n}))>;<cos({n}(#phi-#Psi_{n}))>;Number of Events", 5000, -1, 1)
            hist_dict[f"{event_type}_cosn{n}_etapt"] = ROOT.TProfile2D(
                f"{event_type}_cosn{n}_etapt",
                f"{event_type.upper()} #LTcos({n}(#phi-#Psi_{n}))#GT vs #eta, p_{{T}};#eta;p_{{T}} [GeV];#LTcos({n}(#phi-#Psi_{n}))#GT",
                eta_bins, eta_min, eta_max, pt_bins, pt_min, pt_max
            )
    
    # Add newcon-only histograms for signal and mb
    newcon_types = ["signal_newcon", "mb_newcon"]
    for event_type in newcon_types:
        for n in range(1, 5):
            hist_dict[f"{event_type}_v{n}"] = ROOT.TH1D(f"{event_type}_v{n}", f"{event_type.upper()} Event v_{n};v_{n};Number of Events", 100, 0, 0.2)
            hist_dict[f"{event_type}_cosn{n}"] = ROOT.TH1D(f"{event_type}_cosn{n}", f"{event_type.upper()} <cos({n}(#phi-#Psi_{n}))>;<cos({n}(#phi-#Psi_{n}))>;Number of Events", 5000, -1, 1)
            hist_dict[f"{event_type}_cosn{n}_etapt"] = ROOT.TProfile2D(
                f"{event_type}_cosn{n}_etapt",
                f"{event_type.upper()} #LTcos({n}(#phi-#Psi_{n}))#GT vs #eta, p_{{T}};#eta;p_{{T}} [GeV];#LTcos({n}(#phi-#Psi_{n}))#GT",
                eta_bins, eta_min, eta_max, pt_bins, pt_min, pt_max
            )
    
    # Add spec-only histograms for signal, mb, and hydro
    spec_types = ["signal_spec", "mb_spec", "hydro_spec"]
    for event_type in spec_types:
        for n in range(1, 5):
            hist_dict[f"{event_type}_v{n}"] = ROOT.TH1D(f"{event_type}_v{n}", f"{event_type.upper()} Event v_{n};v_{n};Number of Events", 100, 0, 0.2)
            hist_dict[f"{event_type}_cosn{n}"] = ROOT.TH1D(f"{event_type}_cosn{n}", f"{event_type.upper()} <cos({n}(#phi-#Psi_{n}))>;<cos({n}(#phi-#Psi_{n}))>;Number of Events", 5000, -1, 1)
            hist_dict[f"{event_type}_cosn{n}_etapt"] = ROOT.TProfile2D(
                f"{event_type}_cosn{n}_etapt",
                f"{event_type.upper()} #LTcos({n}(#phi-#Psi_{n}))#GT vs #eta, p_{{T}};#eta;p_{{T}} [GeV];#LTcos({n}(#phi-#Psi_{n}))#GT",
                eta_bins, eta_min, eta_max, pt_bins, pt_min, pt_max
            )
    
    return hist_dict

def fill_event_histograms(hist_dict, event_type, event_props):
    """Fills all histograms and profiles for a single event."""
    all_particles = event_props.get("all_particles")
    if not all_particles: return

    psi_n_list = [calc_event_plane_n(all_particles, n) for n in range(1, 5)]

    for n in range(1, 5):
        hist_dict[f"{event_type}_v{n}"].Fill(event_props[f"v{n}"])
        cosn_val = np.average(np.cos(n * map_ang_mpitopi(np.array([p["phi"] for p in all_particles]) - psi_n_list[n-1])), weights=np.array([p.get("weight", 1.0) for p in all_particles]))
        hist_dict[f"{event_type}_cosn{n}"].Fill(cosn_val)
        
        prof = hist_dict[f"{event_type}_cosn{n}_etapt"]
        psi_n = psi_n_list[n-1]
        for p in all_particles:
            val_to_fill = np.cos(n * map_ang_mpitopi(p["phi"] - psi_n))
            prof.Fill(p["eta"], p["pt"], val_to_fill, p.get("weight", 1.0))

# ==============================================================================
# 4. MAIN PROCESSING LOGIC (Modified for Matching)
# ==============================================================================
def process_events_with_mb_matching(pbpb_datapath, hydro_datapath, outfilename, start_file, end_file, mb_start_file, mb_end_file):
    """Main function with a two-stage process for MB and Signal events."""
    print("Initializing histograms and profiles...")
    hist_dict = make_histograms_and_profiles()
    event_counts = {"signal": 0, "mb": 0, "hydro": 0, "mb_m": 0}
    mb_pool = []

    # Create subdirectories in the output file
    outfile = ROOT.TFile(outfilename, "RECREATE")
    main_dir = outfile.mkdir("main")
    newcon_dir = outfile.mkdir("newcon_only")
    spec_dir = outfile.mkdir("spec_only")

    # --- Stage 1: Pre-load and Process ALL Minimum Bias Events ---
    print(f"\n--- Stage 1: Scanning for ALL Minimum Bias events from index {mb_start_file} to {mb_end_file} ---")
    for evt_idx in range(mb_start_file, mb_end_file):
        if evt_idx % 1 == 0: print(f"  Scanning MB at index {evt_idx}...")
        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        hadjet_file = os.path.join(pbpb_path, "hadjet.dat")
        
        num_jets = 0
        if os.path.exists(hadjet_file) and os.path.getsize(hadjet_file) > 0:
            try:
                num_jets = len(pd.read_csv(hadjet_file, sep='\s+', engine='python', header=None))
            except (pd.errors.EmptyDataError, StopIteration): pass
        
        if num_jets == 0:
            mb_props = load_and_calc_event_properties(pbpb_path)
            if mb_props:
                # Process for the general "mb" category
                fill_event_histograms(hist_dict, "mb", mb_props)
                event_counts["mb"] += 1
                
                # Fill newcon-only histograms for MB
                fill_filtered_histograms(hist_dict, "mb", mb_props.get("all_particles", []), "newcon")
                
                # Fill spec-only histograms for MB
                fill_filtered_histograms(hist_dict, "mb", mb_props.get("all_particles", []), "spec")
                
                # Add to the pool for potential matching
                mb_pool.append(mb_props)
    
    print(f"--- Stage 1 Complete. Found {event_counts['mb']} total MB events and created a pool for matching. ---")

    # --- Stage 2: Process Signal Events and Find Matches ---
    print(f"\n--- Stage 2: Processing Signal events from index {start_file} to {end_file} and matching to MB pool ---")
    for evt_idx in range(start_file, end_file):
        if evt_idx % 1 == 0: print(f"  Processing Signal at index {evt_idx}...")
        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        if not os.path.exists(os.path.join(pbpb_path, "hadjet.dat")) or os.path.getsize(os.path.join(pbpb_path, "hadjet.dat")) == 0:
            continue

        signal_props = load_and_calc_event_properties(pbpb_path)
        if not signal_props: continue

        # A. Process the signal event
        fill_event_histograms(hist_dict, "signal", signal_props)
        event_counts["signal"] += 1
        
        # Fill newcon-only histograms for signal
        fill_filtered_histograms(hist_dict, "signal", signal_props.get("all_particles", []), "newcon")
        
        # Fill spec-only histograms for signal
        fill_filtered_histograms(hist_dict, "signal", signal_props.get("all_particles", []), "spec")
        
        # B. Process its corresponding hydro event
        hydro_path = os.path.join(hydro_datapath, f"event{evt_idx % 400}")
        if os.path.isdir(hydro_path):
            hydro_props = load_and_calc_event_properties(hydro_path)
            if hydro_props:
                fill_event_histograms(hist_dict, "hydro", hydro_props)
                event_counts["hydro"] += 1
                
                # Fill spec-only histograms for hydro
                fill_filtered_histograms(hist_dict, "hydro", hydro_props.get("all_particles", []), "spec")

        # C. Find a matching MB event from the pool (with reuse capability)
        signal_multiplicity = signal_props["multiplicity"]
        match_found = False
        
        # First, try to find an unused MB event
        for i, mb_candidate in enumerate(mb_pool):
            if mb_candidate.get("used", False):
                continue  # Skip already used events in first pass
            mb_multiplicity = mb_candidate["multiplicity"]
            # Check if multiplicity is within 2% of the signal event
            if abs(mb_multiplicity - signal_multiplicity) / (signal_multiplicity + 1e-9) <= 0.02:
                # Found a match!
                fill_event_histograms(hist_dict, "mb_m", mb_candidate)
                event_counts["mb_m"] += 1
                # Mark as used but keep in pool
                mb_candidate["used"] = True
                match_found = True
                break # Stop searching once a match is found
        
        # If no unused match found, search through used events
        if not match_found:
            for i, mb_candidate in enumerate(mb_pool):
                mb_multiplicity = mb_candidate["multiplicity"]
                # Check if multiplicity is within 2% of the signal event
                if abs(mb_multiplicity - signal_multiplicity) / (signal_multiplicity + 1e-9) <= 0.02:
                    # Found a match from used events!
                    fill_event_histograms(hist_dict, "mb_m", mb_candidate)
                    event_counts["mb_m"] += 1
                    match_found = True
                    break # Stop searching once a match is found
        
        if not match_found:
            print(f"  - Warning: No MB match found for signal event {evt_idx} (mult: {signal_multiplicity:.0f})")
    print(f"--- Stage 2 Complete. ---")

    print("\n--- Final Event Summary ---")
    print(f"Total Signal (CoLBT) events: {event_counts['signal']}")
    print(f"Total Hydrodynamic events:   {event_counts['hydro']}")
    print(f"Total Minimum Bias events:   {event_counts['mb']}")
    print(f"Matched Minimum Bias events: {event_counts['mb_m']}")
    print("---------------------------\n")

    print(f"Writing all objects to {outfilename}...")
    
    # Write histograms to appropriate directories
    main_dir.cd()
    for name, hist in hist_dict.items():
        
        if "newcon" in name:
            newcon_dir.cd()
            hist.Write()
        elif "spec" in name:
            spec_dir.cd()
            hist.Write()
        else:
            main_dir.cd()
            hist.Write()
            
    outfile.Close()
    print("Done.")

if __name__ == "__main__":
    pbpb_datapath  = "/home/CoLBT"
    hydro_datapath = "/home/Hydrobackground"
    outfilename    = "/home/vn_analysis.root"
    
    start_file = 0
    end_file = 599 

    mb_start_file = 0
    mb_end_file = 599

    process_events_with_mb_matching(
        pbpb_datapath=pbpb_datapath,
        hydro_datapath=hydro_datapath,
        outfilename=outfilename,
        start_file=start_file,
        end_file=end_file,
        mb_start_file=mb_start_file,
        mb_end_file=mb_end_file
    )