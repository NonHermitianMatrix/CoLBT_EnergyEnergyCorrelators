"""
Finds CoLBT events with 3 or more matching minimum bias events and creates mixed event directories.

For each CoLBT event that has >=3 matching MB events (using mb1 criteria - multiplicity only),
it creates:
./CoLBT_Minimum_Bias/event{i}/m1
./CoLBT_Minimum_Bias/event{i}/m2 
./CoLBT_Minimum_Bias/event{i}/mixed

Each file contains copies of NewConRecom.dat and dNdEtaPtdPtdPhi_Charged.dat from one of the matching MB events.
minimum bias -> multiplicity within 0.1%
"""

import numpy as np
import pandas as pd
import os
import shutil
import random

# Import constants and binning
from spec_new import NY, NPT, NPHI, Y, PT, PHI, gala15w, INVP, gaulew48
from const import HBARC

def map_ang_mpitopi(x):
    """Map angle to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def load_spec_particles(filepath, pt_min=0.0, pt_max=500.0):
    """Load spec particles from dNdEtaPtdPtdPhi_Charged.dat with correct weights and pt filter."""
    try:
        event_charged_dat = os.path.join(filepath, "dNdEtaPtdPtdPhi_Charged.dat")
        if not os.path.exists(event_charged_dat):
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

        # Mask for nonzero and pt range
        mask = (particle_counts_flat > 0) & (pt_flat >= pt_min) & (pt_flat <= pt_max)
        if not np.any(mask):
            return pd.DataFrame()

        particles_df = pd.DataFrame({
            "pt": pt_flat[mask],
            "eta": eta_flat[mask],
            "phi": phi_flat[mask],
            "weight": particle_counts_flat[mask]
        })

        return particles_df

    except Exception as e:
        print(f"Error processing spec data: {e}")
        return pd.DataFrame()

def calc_multiplicity(particles_df):
    """Calculate total multiplicity from particles DataFrame."""
    if particles_df.empty:
        return 0.0
    return particles_df["weight"].sum()

def load_event_properties(filepath):
    """Load event properties including multiplicity."""
    try:
        spec_df = load_spec_particles(filepath)
        if spec_df.empty:
            return None
        
        return {
            "multiplicity": calc_multiplicity(spec_df)
        }
    except Exception as e:
        print(f"Error loading event properties: {e}")
        return None

def find_all_mb_events_centrality_only(mb_events, target_centrality, signal_event_idx, tol_cent=0.001):
    """Find ALL MB events matching centrality only (mb1 criteria)."""
    epsilon = 1e-9
    matching_events = []
    
    for mb_event in mb_events:
        if mb_event["event_idx"] == signal_event_idx:
            continue
        if mb_event.get("num_jets", 0) != 0:
            continue
        centrality_match = abs(mb_event["multiplicity"] - target_centrality) / (target_centrality + epsilon) <= tol_cent
        if centrality_match:
            matching_events.append(mb_event)
    
    return matching_events

def copy_event_files(source_path, dest_path, files_to_copy):
    """Copy specified files from source to destination."""
    os.makedirs(dest_path, exist_ok=True)
    
    for filename in files_to_copy:
        source_file = os.path.join(source_path, filename)
        dest_file = os.path.join(dest_path, filename)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
        else:
            print(f"Warning: {source_file} not found")

def main_analysis(pbpb_datapath, output_base_path, start_file, end_file, 
                  mb_start_file, mb_end_file):
    """Main analysis function to find CoLBT events with 3+ MB matches."""
    
    # First, load MB events
    mb_events = []
    print(f"Loading minimum bias events from {mb_start_file} to {mb_end_file}")
    for evt_idx in range(mb_start_file, mb_end_file):
        if evt_idx % 1 == 0:
            print(f"Loading MB event {evt_idx}")
        
        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        hadjet_file = os.path.join(pbpb_path, "hadjet.dat")
        
        # Check if event has jets
        num_jets = 0
        if os.path.exists(hadjet_file) and os.path.getsize(hadjet_file) > 0:
            try:
                hadjet_df = pd.read_csv(hadjet_file, sep='\s+', engine='python')
                num_jets = len(hadjet_df)
            except:
                pass
        
        # Only use events with no jets as MB
        if num_jets == 0:
            event_props = load_event_properties(pbpb_path)
            if event_props:
                event_props.update({"event_idx": evt_idx, "num_jets": num_jets})
                mb_events.append(event_props)
    
    print(f"Found {len(mb_events)} usable minimum bias events")
    
    # Files to copy for each MB event
    files_to_copy = ["NewConRecom.dat", "dNdEtaPtdPtdPhi_Charged.dat"]
    
    # Process CoLBT events
    colbt_events_with_matches = 0
    
    for evt_idx in range(start_file, end_file):
        if evt_idx % 1 == 0:
            print(f"\nProcessing CoLBT event {evt_idx}/{end_file-1}")

        pbpb_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        if not os.path.isdir(pbpb_path):
            continue

        # Check if this is a CoLBT event (has jets)
        try:
            hadjet_df = pd.read_csv(os.path.join(pbpb_path, "hadjet.dat"), 
                                   sep='\s+', 
                                   names=["jet_no","px","py","pz","E","eta"], 
                                   engine='python')
            if hadjet_df.empty or len(hadjet_df) == 0:
                continue
        except:
            continue

        # Get CoLBT event properties
        colbt_props = load_event_properties(pbpb_path)
        if not colbt_props:
            continue
        
        colbt_centrality = colbt_props["multiplicity"]

        # Find all matching MB events using mb1 criteria (multiplicity only)
        matching_mb_events = find_all_mb_events_centrality_only(
            mb_events, colbt_centrality, evt_idx
        )
        
        # If we have 3 or more matches, create the mixed event directories
        if len(matching_mb_events) >= 3:
            print(f"CoLBT event {evt_idx} has {len(matching_mb_events)} matching MB events")
            colbt_events_with_matches += 1
            
            # If more than 3 matches, randomly select 3
            if len(matching_mb_events) > 3:
                selected_mb_events = random.sample(matching_mb_events, 3)
            else:
                selected_mb_events = matching_mb_events
            
            # Create output directories and copy files
            output_event_path = os.path.join(output_base_path, f"event{evt_idx}")
            
            # Create m1.dat directory with files from first MB event
            m1_path = os.path.join(output_event_path, "m1")
            mb1_source = os.path.join(pbpb_datapath, f"event{selected_mb_events[0]['event_idx']}")
            copy_event_files(mb1_source, m1_path, files_to_copy)
            
            # Create m2.dat directory with files from second MB event
            m2_path = os.path.join(output_event_path, "m2")
            mb2_source = os.path.join(pbpb_datapath, f"event{selected_mb_events[1]['event_idx']}")
            copy_event_files(mb2_source, m2_path, files_to_copy)
            
            # Create mixed.dat directory with files from third MB event
            mixed_path = os.path.join(output_event_path, "mixed")
            mb3_source = os.path.join(pbpb_datapath, f"event{selected_mb_events[2]['event_idx']}")
            copy_event_files(mb3_source, mixed_path, files_to_copy)
            
            # Save info about which MB events were used
            info_file = os.path.join(output_event_path, "mb_event_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"CoLBT event: {evt_idx}\n")
                f.write(f"CoLBT multiplicity: {colbt_centrality:.2f}\n")
                f.write(f"Total matching MB events found: {len(matching_mb_events)}\n")
                f.write(f"\nSelected MB events:\n")
                f.write(f"m1.dat: event{selected_mb_events[0]['event_idx']} (multiplicity: {selected_mb_events[0]['multiplicity']:.2f})\n")
                f.write(f"m2.dat: event{selected_mb_events[1]['event_idx']} (multiplicity: {selected_mb_events[1]['multiplicity']:.2f})\n")
                f.write(f"mixed.dat: event{selected_mb_events[2]['event_idx']} (multiplicity: {selected_mb_events[2]['multiplicity']:.2f})\n")
                
                if len(matching_mb_events) > 3:
                    f.write(f"\nOther matching MB events not used:\n")
                    for mb_event in matching_mb_events:
                        if mb_event not in selected_mb_events:
                            f.write(f"event{mb_event['event_idx']} (multiplicity: {mb_event['multiplicity']:.2f})\n")

    print(f"\n\nAnalysis complete!")
    print(f"Found {colbt_events_with_matches} CoLBT events with 3+ matching MB events")
    print(f"Output saved to {output_base_path}")

if __name__ == '__main__':
    pbpb_datapath = "/home/CoLBT"
    output_base_path = "./CoLBT_Minimum_Bias"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)

    main_analysis(
        pbpb_datapath,
        output_base_path,
        start_file=0,
        end_file=7999,
        mb_start_file=0,
        mb_end_file=7999
    )