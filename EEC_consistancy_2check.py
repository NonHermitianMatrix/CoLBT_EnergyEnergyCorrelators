"""
This macro can be used to get plots consistant with Fig 4 Pb+Pb, k=0.2 plot and Fig 5 Pb+Pb and P+P plots for k = 0.2 in the paper: PHYSICAL REVIEW LETTERS 132, 011901 (2024)
"""

import numpy as np
import pandas as pd
import ROOT
from ROOT import TFile, TH1D, TCanvas, TLegend, gROOT, gPad, gStyle, TNamed
import os

gROOT.SetBatch(True)
gStyle.SetOptStat(0)

JET_CONE_RADIUS = 0.5
MIN_THETA = 0.001 

def map_ang_mpitopi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def theta_between(p1, p2):

    dot = np.sum(p1 * p2, axis=-1)
    norm1 = np.linalg.norm(p1, axis=-1)
    norm2 = np.linalg.norm(p2, axis=-1)
    cos_theta = dot / (norm1 * norm2)
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
def load_gamma_particles(filepath):
    try:
        gamma_path = os.path.join(filepath, "gamma.dat")
        if not os.path.exists(gamma_path): 
            return pd.DataFrame()
       
        df = pd.read_csv(gamma_path, sep='\s+', names=["px", "py", "pz", "E"], engine='python')
        if df.empty: 
            return df
       
        df["pt"] = np.sqrt(df["px"]**2 + df["py"]**2)
        df["p"] = np.sqrt(df["pt"]**2 + df["pz"]**2)
        df["phi"] = map_ang_mpitopi(np.arctan2(df["py"], df["px"]))
        p_minus_pz = df["p"] - df["pz"]
        valid_eta = p_minus_pz > 1e-9
        df["eta"] = np.nan
        df.loc[valid_eta, "eta"] = 0.5 * np.log((df.loc[valid_eta, "p"] + df.loc[valid_eta, "pz"]) / p_minus_pz[valid_eta])
        df.dropna(subset=['eta'], inplace=True)
        return df
    except Exception as e:
        print(f"Error in load_gamma_particles: {e}")
        return pd.DataFrame()
    
def load_newcon_particles(filepath):
    try:
        newcon_path = os.path.join(filepath, "NewConRecom.dat")
        if not os.path.exists(newcon_path): return pd.DataFrame()
        df = pd.read_csv(newcon_path, sep='\s+', names=["pid","px","py","pz","E","mass","x","y","z","t"], engine='python')
        if df.empty: return df
        df["pt"] = np.sqrt(df["px"]**2 + df["py"]**2)
        df["p"] = np.sqrt(df["pt"]**2 + df["pz"]**2)
        df["phi"] = map_ang_mpitopi(np.arctan2(df["py"], df["px"]))
        p_minus_pz = df["p"] - df["pz"]
        valid_eta = p_minus_pz > 1e-9
        df["eta"] = np.nan
        df.loc[valid_eta, "eta"] = 0.5 * np.log((df.loc[valid_eta, "p"] + df.loc[valid_eta, "pz"]) / p_minus_pz[valid_eta])
        df.dropna(subset=['eta'], inplace=True)
        charged_pids = [211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13]
        return df[np.abs(df["pid"]).isin(charged_pids)].copy()
    except Exception as e:
        print(f"Error in load_newcon_particles: {e}")
        return pd.DataFrame()

def load_hadron_particles(filepath):
    try:
        hadron_path = os.path.join(filepath, "hadron.dat")
        if not os.path.exists(hadron_path): return pd.DataFrame()
        df = pd.read_csv(hadron_path, sep='\s+', names=["pid", "px", "py", "pz", "E"], engine='python')
        if df.empty: return df
        df["pt"] = np.sqrt(df["px"]**2 + df["py"]**2)
        df["p"] = np.sqrt(df["pt"]**2 + df["pz"]**2)
        df["phi"] = map_ang_mpitopi(np.arctan2(df["py"], df["px"]))
        p_minus_pz = df["p"] - df["pz"]
        valid_eta = p_minus_pz > 1e-9
        df["eta"] = np.nan
        df.loc[valid_eta, "eta"] = 0.5 * np.log((df.loc[valid_eta, "p"] + df.loc[valid_eta, "pz"]) / p_minus_pz[valid_eta])
        df.dropna(subset=['eta'], inplace=True)
        charged_pids = [211, -211, 321, -321, 411, -411, 421, -421, 431, -431, 441, -441, 521, -521, 531, -531, 541, -541, 213, -213, 323, -323, 413, -413, 423, -423, 433, -433, 443, -443, 523, -523, 533, -533, 543, -543, 100211, -100211, 100321, -100321, 10211, -10211, 10321, -10321, 20213, -20213, 20323, -20323, 2212, -2212, 2112, -2112, 3122, -3122, 3112, -3112, 3222, -3222, 3312, -3312, 3322, -3322, 3334, -3334, 4122, -4122, 4112, -4112, 4212, -4212, 4222, -4222, 4232, -4232, 4132, -4132, 4332, -4332, 5122, -5122, 5112, -5112, 5212, -5212, 5222, -5222, 5232, -5232, 5132, -5132, 5332, -5332]
        df_charged = df[df["pid"].abs().isin(charged_pids)].copy()
        return df_charged
    except Exception as e:
        print(f"Error in load_hadron_particles: {e}")
        return pd.DataFrame()

def load_parton_particles(filepath):
    try:
        parton_path = os.path.join(filepath, "tc_inf.dat")
        if not os.path.exists(parton_path): return pd.DataFrame()
        df = pd.read_csv(parton_path, sep='\s+', names=["pid", "px", "py", "pz", "E", "x", "y", "z", "t", "cat"], engine='python')
        if df.empty: return df
        df["pt"] = np.sqrt(df["px"]**2 + df["py"]**2)
        df["p"] = np.sqrt(df["pt"]**2 + df["pz"]**2)
        df["phi"] = map_ang_mpitopi(np.arctan2(df["py"], df["px"]))
        p_minus_pz = df["p"] - df["pz"]
        valid_eta = p_minus_pz > 1e-9
        df["eta"] = np.nan
        df.loc[valid_eta, "eta"] = 0.5 * np.log((df.loc[valid_eta, "p"] + df.loc[valid_eta, "pz"]) / p_minus_pz[valid_eta])
        df.dropna(subset=['eta'], inplace=True)
        return df
    except Exception as e:
        print(f"Error in load_parton_particles: {e}")
        return pd.DataFrame()

def fill_eec_histogram(histogram, particles_in_cone_df, jet_pt, pt_cut=0.0):
    filtered_df = particles_in_cone_df[particles_in_cone_df["pt"] > pt_cut]
    if len(filtered_df) < 2: return
    px = filtered_df["px"].to_numpy()
    py = filtered_df["py"].to_numpy()
    pz = filtered_df["pz"].to_numpy()
    pt = filtered_df["pt"].to_numpy()
    n = len(px)
    idx_i, idx_j = np.triu_indices(n, k=1)
    pvecs_i = np.stack([px[idx_i], py[idx_i], pz[idx_i]], axis=1)
    pvecs_j = np.stack([px[idx_j], py[idx_j], pz[idx_j]], axis=1)
    thetas = theta_between(pvecs_i, pvecs_j)
    weights = (pt[idx_i] * pt[idx_j]) / (jet_pt**2)
    valid_pairs = thetas >= MIN_THETA
    theta_to_fill = thetas[valid_pairs]
    weights_to_fill = weights[valid_pairs]
    if len(theta_to_fill) > 0:
        histogram.FillN(len(theta_to_fill), theta_to_fill, weights_to_fill)

def create_eec_histograms(pbpb_datapath, pp_datapath, outfilename, start_event, end_event):
    outfile = TFile(outfilename, "RECREATE")
    pt_cuts = [0.0, 1.0, 2.0]
    
    n_bins = 50
    log_min = np.log10(MIN_THETA)
    log_max = 0
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_array = np.array(bin_edges, dtype=np.float64)
    
    histograms = {}
    
    dirs = { "NewCon": outfile.mkdir("NewCon"), "Hadron": outfile.mkdir("Hadron"), "Parton": outfile.mkdir("Parton") }
    parton_cat_dirs = { "all": dirs["Parton"].mkdir("All"), "shower": dirs["Parton"].mkdir("Shower"), "recoil": dirs["Parton"].mkdir("Recoil"), "radiated": dirs["Parton"].mkdir("Radiated") }
    
    y_axis_title_dr_mult = "EEC as given in the paper"

    for pt_cut in pt_cuts:
        pt_str = f"pt{str(pt_cut).replace('.', 'p')}"
        histograms[f"newcon_{pt_cut}"] = TH1D(f"eec_newcon_{pt_str}", f"EEC NewCon (pT > {pt_cut} GeV);#theta;{y_axis_title_dr_mult}", n_bins, bin_array)
        histograms[f"hadron_{pt_cut}"] = TH1D(f"eec_hadron_{pt_str}", f"EEC Hadron (pT > {pt_cut} GeV);#theta;{y_axis_title_dr_mult}", n_bins, bin_array)
        histograms[f"parton_all_{pt_cut}"] = TH1D(f"eec_parton_all_{pt_str}", f"EEC All Partons (pT > {pt_cut} GeV);#theta;{y_axis_title_dr_mult}", n_bins, bin_array)
        for cat in ["shower", "recoil", "radiated"]:
            histograms[f"parton_{cat}_{pt_cut}"] = TH1D(f"eec_parton_{cat}_{pt_str}", f"EEC {cat.capitalize()} Partons (pT > {pt_cut} GeV);#theta;{y_axis_title_dr_mult}", n_bins, bin_array)

    for hist in histograms.values():
        hist.Sumw2()
   
    n_with_pbpb_jet = 0
    n_with_pp_jet = 0
    n_pbpb_gamma_jet_events = 0
    n_pp_gamma_jet_events = 0
    
    for evt_idx in range(start_event, end_event):
        if evt_idx % 1 == 0: 
            print(f"Processing event {evt_idx}...")
        
        pbpb_event_path = os.path.join(pbpb_datapath, f"event{evt_idx}")
        pp_event_path = os.path.join(pp_datapath, f"event{evt_idx}")
        
        pbpb_hadjet_file = os.path.join(pbpb_event_path, "hadjet.dat")
        if os.path.exists(pbpb_hadjet_file):
            try:
                
                pbpb_hadjet_df = pd.read_csv(pbpb_hadjet_file, sep='\s+', names=["jno","px","py","pz","E","eta"], engine='python')
                pbpb_gamma_df = load_gamma_particles(pbpb_event_path)
                
                if len(pbpb_hadjet_df) == 1 and len(pbpb_gamma_df) == 1:
                    jet_row = pbpb_hadjet_df.iloc[0]
                    gamma_row = pbpb_gamma_df.iloc[0]
                    
                    jet_pt = np.sqrt(jet_row["px"]**2 + jet_row["py"]**2)
                    jet_eta, jet_phi = jet_row["eta"], map_ang_mpitopi(np.arctan2(jet_row["py"], jet_row["px"]))
                    gamma_pt = gamma_row["pt"]
                    gamma_phi = gamma_row["phi"]
                    
                    if gamma_pt >= 100.0 and jet_pt >= 50.0:
                    
                        delta_phi = abs(map_ang_mpitopi(jet_phi - gamma_phi))
                        if delta_phi > 2.0: 
                            n_pbpb_gamma_jet_events += 1
                            n_with_pbpb_jet += 1
                            
                            newcon_all, partons_all = load_newcon_particles(pbpb_event_path), load_parton_particles(pbpb_event_path)
                            
                    
                    newcon_all, partons_all = load_newcon_particles(pbpb_event_path), load_parton_particles(pbpb_event_path)
                    
                    if not newcon_all.empty:
                        dr_to_jet = np.sqrt((newcon_all['eta'] - jet_eta)**2 + map_ang_mpitopi(newcon_all['phi'] - jet_phi)**2)
                        newcon_in_cone = newcon_all[dr_to_jet < JET_CONE_RADIUS]
                        for pt_cut in pt_cuts: 
                            fill_eec_histogram(histograms[f"newcon_{pt_cut}"], newcon_in_cone, jet_pt, pt_cut)
                    
                    if not partons_all.empty:
                        dr_to_jet = np.sqrt((partons_all['eta'] - jet_eta)**2 + map_ang_mpitopi(partons_all['phi'] - jet_phi)**2)
                        partons_in_cone = partons_all[dr_to_jet < JET_CONE_RADIUS]
                        for pt_cut in pt_cuts:
                            fill_eec_histogram(histograms[f"parton_all_{pt_cut}"], partons_in_cone, jet_pt, pt_cut)
                            for cat_name, cat_val in {"shower": 0, "recoil": 2, "radiated": 4}.items():
                                fill_eec_histogram(histograms[f"parton_{cat_name}_{pt_cut}"], partons_in_cone[partons_in_cone["cat"] == cat_val], jet_pt, pt_cut)
                
            except Exception as e: 
                print(f"Error processing PbPb event {evt_idx}: {e}")
        pp_hadjet_file = os.path.join(pp_event_path, "hadjet.dat")
        if os.path.exists(pp_hadjet_file):
            try:
              
                pp_hadjet_df = pd.read_csv(pp_hadjet_file, sep='\s+', names=["jno","px","py","pz","E","eta"], engine='python')
                pp_gamma_df = load_gamma_particles(pp_event_path)
                
                if len(pp_hadjet_df) == 1 and len(pp_gamma_df) == 1:
                    jet_row = pp_hadjet_df.iloc[0]
                    gamma_row = pp_gamma_df.iloc[0]
                    
                    jet_pt = np.sqrt(jet_row["px"]**2 + jet_row["py"]**2)
                    jet_eta, jet_phi = jet_row["eta"], map_ang_mpitopi(np.arctan2(jet_row["py"], jet_row["px"]))
                    gamma_pt = gamma_row["pt"]
                    gamma_phi = gamma_row["phi"]
                    
                    if gamma_pt >= 100.0 and jet_pt >= 50.0:
                      
                        delta_phi = abs(map_ang_mpitopi(jet_phi - gamma_phi))
                        if delta_phi > 2.0: 
                            n_pp_gamma_jet_events += 1
                            n_with_pp_jet += 1
                            
                            hadrons_all = load_hadron_particles(pp_event_path)
                            if not hadrons_all.empty:
                                dr_to_jet = np.sqrt((hadrons_all['eta'] - jet_eta)**2 + map_ang_mpitopi(hadrons_all['phi'] - jet_phi)**2)
                                hadrons_in_cone = hadrons_all[dr_to_jet < JET_CONE_RADIUS]
                                for pt_cut in pt_cuts: 
                                    fill_eec_histogram(histograms[f"hadron_{pt_cut}"], hadrons_in_cone, jet_pt, pt_cut)
                                    
            except Exception as e: 
                print(f"Error processing PP event {evt_idx}: {e}")


    print("\n--- Normalizing Histograms ---")
    print(f"Total PbPb gamma+jet events: {n_pbpb_gamma_jet_events}")
    print(f"Total PP gamma+jet events: {n_pp_gamma_jet_events}")
    print(f"Total PbPb jets processed: {n_with_pbpb_jet}")
    print(f"Total PP jets processed: {n_with_pp_jet}")


    summary_text = TNamed("Summary", f"PbPb: {pbpb_datapath}\nPP: {pp_datapath}\nPbPb Gamma+Jet Events: {n_pbpb_gamma_jet_events}, PP Gamma+Jet Events: {n_pp_gamma_jet_events}\nPbPb Jets: {n_with_pbpb_jet}, PP Jets: {n_with_pp_jet}\nCone: {JET_CONE_RADIUS}")

    print("\n--- Normalizing Histograms ---")
    print(f"Total PbPb jets processed: {n_with_pbpb_jet}")
    print(f"Total PP jets processed: {n_with_pp_jet}")
    
    for hist_name, hist in histograms.items():
        n_jets = n_with_pp_jet if "hadron" in hist_name else n_with_pbpb_jet
        if n_jets > 0:
            for i_bin in range(1, hist.GetNbinsX() + 1):
                bin_content = hist.GetBinContent(i_bin)
                bin_error = hist.GetBinError(i_bin)
                bin_width = hist.GetBinWidth(i_bin)
                if bin_width > 0:
                    norm_content = bin_content / (n_jets * bin_width)
                    norm_error = bin_error / (n_jets * bin_width)
                    
                    if "parton" not in hist_name:
                        bin_center = hist.GetBinCenter(i_bin)
                        final_content = norm_content * bin_center
                        final_error = norm_error * bin_center
                    else:
                       
                        final_content = norm_content
                        final_error = norm_error
                        
                    hist.SetBinContent(i_bin, final_content)
                    hist.SetBinError(i_bin, final_error)
    print("\n--- Writing to ROOT file ---")
    for pt_cut in pt_cuts:
        dirs["NewCon"].cd(); histograms[f"newcon_{pt_cut}"].Write()
        dirs["Hadron"].cd(); histograms[f"hadron_{pt_cut}"].Write()
        parton_cat_dirs["all"].cd(); histograms[f"parton_all_{pt_cut}"].Write()
        for cat in ["shower", "recoil", "radiated"]:
            parton_cat_dirs[cat].cd(); histograms[f"parton_{cat}_{pt_cut}"].Write()

    comp_dir = outfile.mkdir("Comparisons")
    comp_dir.cd()
    for pt_cut in pt_cuts:
        canvas = TCanvas(f"c_comp_pt{pt_cut}", f"EEC Comparison (pT > {pt_cut} GeV)", 800, 600)
        canvas.SetLogx(); canvas.SetLogy()
        h_newcon = histograms[f"newcon_{pt_cut}"]
        h_hadron = histograms[f"hadron_{pt_cut}"]
        h_parton = histograms[f"parton_all_{pt_cut}"]
        h_newcon.SetLineColor(ROOT.kRed); h_newcon.SetMarkerColor(ROOT.kRed)
        h_hadron.SetLineColor(ROOT.kBlue); h_hadron.SetMarkerColor(ROOT.kBlue)
        h_parton.SetLineColor(ROOT.kGreen+2); h_parton.SetMarkerColor(ROOT.kGreen+2)
        h_newcon.GetYaxis().SetTitle("Normalized EEC Observable")
        h_hadron.GetYaxis().SetTitle("Normalized EEC Observable")
        h_parton.GetYaxis().SetTitle("Normalized EEC Observable")
        h_newcon.Draw("E1")
        h_hadron.Draw("E1 SAME")
        h_parton.Draw("E1 SAME")
        legend = TLegend(0.55, 0.7, 0.88, 0.88)
        legend.AddEntry(h_newcon, "NewCon #times #theta (PbPb)", "le")
        legend.AddEntry(h_hadron, "Hadron #times  (PP)", "le")
        legend.AddEntry(h_parton, "All Partons #times #theta (PbPb)", "le")
        legend.SetBorderSize(0); legend.SetFillStyle(0)
        legend.Draw()
        canvas.Write()

    outfile.cd()
    summary_text = TNamed("Summary", f"PbPb: {pbpb_datapath}\nPP: {pp_datapath}\nPbPb Jets: {n_with_pbpb_jet}, PP Jets: {n_with_pp_jet}\nCone: {JET_CONE_RADIUS}")
    summary_text.Write()
    outfile.Close()
    print(f"\nEEC histograms saved to {outfilename}")

if __name__ == '__main__':
    pbpb_datapath = "/home/CoLBT"
    pp_datapath = "/home/PP"
    outfilename = "/home/EEC_consistancy_2check.root"
    start_event = 0
    end_event = 3999
    create_eec_histograms(pbpb_datapath, pp_datapath, outfilename, start_event, end_event)