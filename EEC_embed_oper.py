"""
Run this after getting EECs, it will extract the jet eec from that data and present actual vs calculated eec comparisons
"""
import ROOT

def get_bin_width(hist, bin_num):
    return hist.GetBinWidth(bin_num)

def normalize_by_jets(hist, n_jets):
    if n_jets > 0:
        hist.Scale(1.0 / n_jets)

def normalize_by_rel_bin_width(hist):
    first_bin_width = get_bin_width(hist, 1)
    for i in range(1, hist.GetNbinsX() + 1):
        current_bin_width = get_bin_width(hist, i)
        ratio = current_bin_width / first_bin_width
        content = hist.GetBinContent(i)
        error = hist.GetBinError(i)
        hist.SetBinContent(i, content / ratio)
        hist.SetBinError(i, error / ratio)

def subtract_histograms(s, sm1, m1m2, m1m1, name):
    result = s.Clone(name)
    result.Add(sm1, -1)
    result.Add(m1m2, 1)
    result.Add(m1m1, -1) 
    return result

def plot_histograms(hists, canvas_name, title, directory_name, output_file):
    c = ROOT.TCanvas(canvas_name, title, 800, 600)
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    
    colors = [1, 2, 3, 4, 6]
    
    for i, hist in enumerate(hists):
        if hist:
            hist.SetLineColor(colors[i % 5])
            hist.SetMarkerColor(colors[i % 5])
            hist.GetXaxis().SetTitle("|r_{i}-r_{j}|")
            if i == 0:
                hist.Draw()
            else:
                hist.Draw("same")
            leg.AddEntry(hist, hist.GetName(), "l")
    
    leg.Draw()
    output_file.cd(directory_name)
    c.Write()

def extract_histogram_parameters(hist_name):
    parts = hist_name.split('_')
    try:
        for i, part in enumerate(parts):
            if part.startswith('n'):
                n_exp = part[1:]
            elif part.startswith('jetpt'):
                jetpt_parts = part[5:].split('to')
                jetpt_min = int(jetpt_parts[0])
                jetpt_max = int(jetpt_parts[1])
            elif part.startswith('pch'):
                pch_cut = part[3:]
        
        return {
            'n_exp': n_exp,
            'jetpt_min': jetpt_min,
            'jetpt_max': jetpt_max,
            'pch_cut': pch_cut
        }
    except:
        return None
def plot_added_histograms(added_hists, output_file, directory_name):
    if not added_hists:
        return
    
    plot_dir = output_file.GetDirectory(directory_name)
    if not plot_dir:
        plot_dir = output_file.mkdir(directory_name)
    plot_dir.cd()
    
    for name, hist in added_hists.items():
        canvas_name = f"canvas_{name}"
        canvas = ROOT.TCanvas(canvas_name, name, 800, 600)
        canvas.SetLogx()
        
        hist.SetLineColor(ROOT.kBlue)
        hist.SetLineWidth(2)
        hist.SetMarkerColor(ROOT.kBlue)
        hist.SetMarkerStyle(20)
        hist.GetXaxis().SetTitle("#Delta r")
        hist.GetYaxis().SetTitle("EEC Distribution")
        hist.SetTitle(name)
        
        hist.Draw("E1")
        canvas.Write()
def create_added_jetpt_comparison_plots(pbpb_hists, pp_hists, pbpb_newcon_hists, pp_hadron_hists, output_file, jetpt_ranges, pch_values):
    output_file.cd("added_jetpt_comparison_plots")
    
    n_values = ["n0", "n1", "n2"]
    plot_count = 0
    
    for n in n_values:
        for pch in pch_values:
            key = f"{n}_{pch}"
            
            # Check if all histograms exist
            if key not in pbpb_hists or key not in pp_hists:
                print(f"Warning: Missing histograms for key {key}")
                continue
                
            # Clone histograms and detach from directory
            pbpb_hist = pbpb_hists[key].Clone(f"pbpb_temp_{key}")
            pbpb_hist.SetDirectory(0)
            
            pp_hist = pp_hists[key].Clone(f"pp_temp_{key}")
            pp_hist.SetDirectory(0)
            
            pbpb_newcon_hist = None
            if key in pbpb_newcon_hists:
                pbpb_newcon_hist = pbpb_newcon_hists[key].Clone(f"pbpb_newcon_temp_{key}")
                pbpb_newcon_hist.SetDirectory(0)
                
            pp_hadron_hist = None
            if key in pp_hadron_hists:
                pp_hadron_hist = pp_hadron_hists[key].Clone(f"pp_hadron_temp_{key}")
                pp_hadron_hist.SetDirectory(0)
            
            # Create canvas
            canvas_name = f"added_jetpt_comparison_{n}_{pch}"
            c = ROOT.TCanvas(canvas_name, f"Added Jet pT Comparison {n} {pch}", 800, 600)
            c.SetGridx()
            c.SetGridy()
            c.SetLogx()
            
            # Set styles and draw
            pbpb_hist.SetLineColor(2)
            pbpb_hist.SetMarkerColor(2)
            pbpb_hist.SetMarkerStyle(20)
            pbpb_hist.SetMarkerSize(0.8)
            pbpb_hist.GetXaxis().SetTitle("#Delta r")
            pbpb_hist.GetYaxis().SetTitle("EEC (summed over jet p_{T})")
            pbpb_hist.SetTitle(f"Added Jet p_t Comparison: {n}, p_{{ch}} > {pch.replace('pch', '')} GeV")
            pbpb_hist.Draw("E1")
            
            pp_hist.SetLineColor(4)
            pp_hist.SetMarkerColor(4)
            pp_hist.SetMarkerStyle(21)
            pp_hist.SetMarkerSize(0.8)
            pp_hist.Draw("E1 SAME")
            
            if pbpb_newcon_hist:
                pbpb_newcon_hist.SetLineColor(ROOT.kGreen+2)
                pbpb_newcon_hist.SetMarkerColor(ROOT.kGreen+2)
                pbpb_newcon_hist.SetMarkerStyle(22)
                pbpb_newcon_hist.SetMarkerSize(0.8)
                pbpb_newcon_hist.SetLineStyle(2)
                pbpb_newcon_hist.Draw("E1 SAME")
            
            if pp_hadron_hist:
                pp_hadron_hist.SetLineColor(ROOT.kMagenta)
                pp_hadron_hist.SetMarkerColor(ROOT.kMagenta)
                pp_hadron_hist.SetMarkerStyle(23)
                pp_hadron_hist.SetMarkerSize(0.8)
                pp_hadron_hist.SetLineStyle(2)
                pp_hadron_hist.Draw("E1 SAME")
            
            # Create legend
            legend = ROOT.TLegend(0.65, 0.65, 0.88, 0.88)
            legend.SetFillStyle(1001)
            legend.SetFillColor(ROOT.kWhite)
            legend.SetBorderSize(1)
            legend.AddEntry(pbpb_hist, "PbPb (subtracted)", "lp")
            legend.AddEntry(pp_hist, "PP embedded (subtracted)", "lp")
            if pbpb_newcon_hist:
                legend.AddEntry(pbpb_newcon_hist, "PbPb Jet (actual)", "lp")
            if pp_hadron_hist:
                legend.AddEntry(pp_hadron_hist, "PP Jet (actual)", "lp")
            legend.Draw()
            
            # Write canvas
            output_file.cd("added_jetpt_comparison_plots")
            c.Write()
            
            # Clean up - delete temporary objects
            del pbpb_hist
            del pp_hist
            if pbpb_newcon_hist:
                del pbpb_newcon_hist
            if pp_hadron_hist:
                del pp_hadron_hist
            del legend
            del c
            
            plot_count += 1
    
    print(f"Created {plot_count} added jetpt comparison plots")
                
def create_combined_comparison_plots(pbpb_results, pp_results, pbpb_newcon_cone_processed, pp_hadron_cone_processed, output_file):
    comparison_dir = output_file.GetDirectory("combined_comparison_plots")
    if not comparison_dir:
        comparison_dir = output_file.mkdir("combined_comparison_plots")
    comparison_dir.cd()
    
    pp_results_map = {}
    for key, hist in pp_results.items():
        pp_results_map[key] = hist
    
    plots_created = 0
    grid_data = {}
    
    for pbpb_key, pbpb_hist in pbpb_results.items():
        params = extract_histogram_parameters(f"pbpb_{pbpb_key}")
        if not params:
            print(f"Warning: Could not extract parameters from PbPb result key: {pbpb_key}")
            continue
        
        n_exp = params['n_exp']
        jetpt_min = params['jetpt_min']
        jetpt_max = params['jetpt_max']
        pch_cut = params['pch_cut']
        
        pp_hist = pp_results_map.get(pbpb_key)
        if not pp_hist:
            print(f"Warning: No matching PP result found for {pbpb_key}")
            continue
        
        pbpb_newcon_key = f"pbpb_newcon_cone_eec_n{n_exp}_jetpt{jetpt_min}to{jetpt_max}_pch{pch_cut}"
        pbpb_newcon_hist = pbpb_newcon_cone_processed.get(pbpb_newcon_key)
        if not pbpb_newcon_hist:
            print(f"Warning: No matching PbPb newcon cone histogram found for {pbpb_key}")
            continue
        
        pp_hadron_key = f"pp_hadron_cone_eec_n{n_exp}_jetpt{jetpt_min}to{jetpt_max}_pch{pch_cut}"
        pp_hadron_hist = pp_hadron_cone_processed.get(pp_hadron_key)
        if not pp_hadron_hist:
            print(f"Warning: No matching PP hadron cone histogram found for {pbpb_key}")
            continue
        
        canvas_name = f"combined_{pbpb_key}"
        title_info = f"n={n_exp}, Jet pT={jetpt_min}-{jetpt_max} GeV, pT_ch>{pch_cut} GeV"
        canvas = ROOT.TCanvas(canvas_name, f"Combined Comparison: {title_info}", 1000, 700)
        canvas.SetLogx()
        canvas.SetLeftMargin(0.12)
        canvas.SetRightMargin(0.05)
        canvas.SetTopMargin(0.08)
        canvas.SetBottomMargin(0.12)
        
        pbpb_hist.SetLineColor(ROOT.kBlue)
        pbpb_hist.SetLineWidth(3)
        pbpb_hist.SetMarkerColor(ROOT.kBlue)
        pbpb_hist.SetMarkerStyle(20)
        pbpb_hist.SetMarkerSize(0.8)
        pbpb_hist.SetTitle(f"Combined Comparison: {title_info}")
        pbpb_hist.GetXaxis().SetTitle("#Delta r")
        pbpb_hist.GetYaxis().SetTitle("EEC Distribution")
        pbpb_hist.GetXaxis().SetTitleSize(0.045)
        pbpb_hist.GetYaxis().SetTitleSize(0.045)
        pbpb_hist.GetXaxis().SetLabelSize(0.04)
        pbpb_hist.GetYaxis().SetLabelSize(0.04)
        pbpb_hist.GetXaxis().SetTitleOffset(1.1)
        pbpb_hist.GetYaxis().SetTitleOffset(1.3)
        
        pp_hist.SetLineColor(ROOT.kRed)
        pp_hist.SetLineWidth(3)
        pp_hist.SetMarkerColor(ROOT.kRed)
        pp_hist.SetMarkerStyle(21)
        pp_hist.SetMarkerSize(0.8)
        
        pbpb_newcon_hist.SetLineColor(ROOT.kGreen+2)
        pbpb_newcon_hist.SetLineWidth(2)
        pbpb_newcon_hist.SetLineStyle(2)
        pbpb_newcon_hist.SetMarkerColor(ROOT.kGreen+2)
        pbpb_newcon_hist.SetMarkerStyle(22)
        pbpb_newcon_hist.SetMarkerSize(0.7)
        
        pp_hadron_hist.SetLineColor(ROOT.kMagenta)
        pp_hadron_hist.SetLineWidth(2)
        pp_hadron_hist.SetLineStyle(2)
        pp_hadron_hist.SetMarkerColor(ROOT.kMagenta)
        pp_hadron_hist.SetMarkerStyle(23)
        pp_hadron_hist.SetMarkerSize(0.7)
        
        all_hists = [pbpb_hist, pp_hist, pbpb_newcon_hist, pp_hadron_hist]
        max_value = max([h.GetMaximum() for h in all_hists]) * 1.3
        min_value = min([h.GetMinimum() for h in all_hists])
        
        if min_value >= 0:
            min_value = min_value * 0.8 if min_value > 0 else -max_value * 0.1
        else:
            min_value = min_value * 1.2
        
        pbpb_hist.SetMaximum(max_value)
        pbpb_hist.SetMinimum(min_value)
        
        pbpb_hist.Draw("E1")
        pp_hist.Draw("E1 SAME")
        pbpb_newcon_hist.Draw("E1 SAME")
        pp_hadron_hist.Draw("E1 SAME")
        
        legend = ROOT.TLegend(0.15, 0.75, 0.45, 0.92)
        legend.SetFillStyle(1001)
        legend.SetFillColor(ROOT.kWhite)
        legend.SetBorderSize(1)
        legend.SetTextSize(0.035)
        legend.SetTextFont(42)
        legend.AddEntry(pbpb_hist, "PbPb Result", "lp")
        legend.AddEntry(pp_hist, "PP Result", "lp")
        legend.AddEntry(pbpb_newcon_hist, "PbPb NewCon Cone", "lp")
        legend.AddEntry(pp_hadron_hist, "PP Hadron Cone", "lp")
        legend.Draw()
        
        
        if pbpb_hist or pp_hist:
            reference_hist = pbpb_hist if pbpb_hist else pp_hist
            if reference_hist and reference_hist.InheritsFrom("TH1"):
                zero_line = ROOT.TLine(reference_hist.GetXaxis().GetXmin(), 
                                            0, 
                                            reference_hist.GetXaxis().GetXmax(), 
                                            0)
                zero_line.SetLineStyle(3)
                zero_line.SetLineColor(ROOT.kGray+1)
                zero_line.SetLineWidth(1)
                zero_line.Draw("same")
        
        canvas.Write()
        plots_created += 1
        
        if n_exp not in grid_data:
            grid_data[n_exp] = {}
        
        jetpt_key = f"{jetpt_min}to{jetpt_max}"
        pch_key = f"{pch_cut}"
        
        if jetpt_key not in grid_data[n_exp]:
            grid_data[n_exp][jetpt_key] = {}
        
        grid_data[n_exp][jetpt_key][pch_key] = {
            'pbpb_hist': pbpb_hist.Clone(),
            'pp_hist': pp_hist.Clone(),
            'pbpb_newcon_hist': pbpb_newcon_hist.Clone(),
            'pp_hadron_hist': pp_hadron_hist.Clone(),
            'jetpt_min': jetpt_min,
            'jetpt_max': jetpt_max,
            'pch_cut': pch_cut,
            'n_exp': n_exp
        }
    
    for n_exp in sorted(grid_data.keys()):
        jetpt_keys = sorted(grid_data[n_exp].keys(), key=lambda k: int(k.split('to')[0]))
        all_pch_keys = {pch for jet_data in grid_data[n_exp].values() for pch in jet_data}
        pch_keys = sorted(all_pch_keys, key=float)
        
        n_cols, n_rows = len(jetpt_keys), len(pch_keys)
        if n_cols == 0 or n_rows == 0:
            continue

        grid_canvas = ROOT.TCanvas(f"grid_combined_n{n_exp}", f"Grid n={n_exp}", 5000 * n_cols, 5000 * n_rows)
        pads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        
        margins = [0.15, 0.05, 0.15, 0.15]
        pad_width = (1. - margins[0] - margins[1]) / n_cols
        pad_height = (1. - margins[2] - margins[3]) / n_rows

        for r in range(n_rows):
            for c in range(n_cols):
                pad_name = f"pad_n{n_exp}_r{r}_c{c}"
                
                x1 = margins[0] + c * pad_width
                y1 = margins[3] + (n_rows - 1 - r) * pad_height
                x2 = x1 + pad_width
                y2 = y1 + pad_height
                
                pad = ROOT.TPad(pad_name, pad_name, x1, y1, x2, y2)
                pad.SetLogx()
                pad.SetGridx()
                pad.SetGridy()
                pad.Draw()
                pads[r][c] = pad
        
        grid_canvas.cd() 
        
        for c, jetpt_key in enumerate(jetpt_keys):
            x_pos = margins[0] + (c + 0.5) * pad_width
            y_pos = 1. - margins[2] / 2
            label = ROOT.TLatex(x_pos, y_pos, f"Jet pT: {jetpt_key.replace('to', '-')} GeV")
            label.SetNDC()
            label.SetTextAlign(22)
            label.SetTextSize(0.03)
            label.Draw()

        for r, pch_key in enumerate(pch_keys):
            x_pos = margins[0] / 2
            y_pos = margins[3] + (n_rows - 1 - r + 0.5) * pad_height
            label = ROOT.TLatex(x_pos, y_pos, f"p_{{T,ch}} > {pch_key} GeV")
            label.SetNDC()
            label.SetTextAlign(22)
            label.SetTextAngle(90)
            label.SetTextSize(0.03)
            label.Draw()

        for r, pch_key in enumerate(pch_keys):
        
            row_max, row_min = -1e9, 1e9
            for c, jetpt_key in enumerate(jetpt_keys):
                if pch_key in grid_data[n_exp][jetpt_key]:
                    h_data = grid_data[n_exp][jetpt_key][pch_key]
                    for h in h_data.values():
                        
                        if hasattr(h, 'GetMaximum'):
                            row_max = max(row_max, h.GetMaximum())
                            row_min = min(row_min, h.GetMinimum())
            
            y_range = [row_min * 1.2 if row_min < 0 else row_min * 0.8, row_max * 1.2]

            for c, jetpt_key in enumerate(jetpt_keys):
                if pch_key not in grid_data[n_exp][jetpt_key]: continue
                
                pad = pads[r][c]
                pad.cd() 
                
                hists = grid_data[n_exp][jetpt_key][pch_key]
                pbpb, pp, pbpb_nc, pp_hc = hists['pbpb_hist'], hists['pp_hist'], hists['pbpb_newcon_hist'], hists['pp_hadron_hist']
                
                pbpb.SetStats(0)
                pbpb.SetTitle("")
                pbpb.GetYaxis().SetRangeUser(y_range[0], y_range[1])
                pbpb.GetXaxis().SetTitle("#Delta r")
                pbpb.GetYaxis().SetTitle("EEC")
                pbpb.GetXaxis().SetTitleSize(0.1)
                pbpb.GetYaxis().SetTitleSize(0.1)
                pbpb.GetXaxis().SetLabelSize(0.05)
                pbpb.GetYaxis().SetLabelSize(0.05)
                pbpb.GetYaxis().SetTitleOffset(1.2)

                pbpb.SetLineColor(ROOT.kBlue)
                pbpb.SetMarkerColor(ROOT.kBlue)
                pbpb.SetMarkerStyle(20)
                pp.SetLineColor(ROOT.kRed)
                pp.SetMarkerColor(ROOT.kRed)
                pp.SetMarkerStyle(21)
                pbpb_nc.SetLineColor(ROOT.kGreen+2)
                pbpb_nc.SetMarkerColor(ROOT.kGreen+2)
                pbpb_nc.SetMarkerStyle(22)
                pbpb_nc.SetLineStyle(2)
                pp_hc.SetLineColor(ROOT.kMagenta)
                pp_hc.SetMarkerColor(ROOT.kMagenta)
                pp_hc.SetMarkerStyle(23)
                pp_hc.SetLineStyle(2)

                pbpb.Draw("E1")
                pp.Draw("E1 SAME")
                pbpb_nc.Draw("E1 SAME")
                pp_hc.Draw("E1 SAME")

                xmin, xmax = pbpb.GetXaxis().GetXmin(), pbpb.GetXaxis().GetXmax()
                zero_line = ROOT.TLine(xmin, 0, xmax, 0)
                zero_line.SetLineStyle(3)
                zero_line.SetLineColor(ROOT.kGray+1)
                zero_line.Draw("same")

                if r == 0 and c == n_cols - 1:
                    legend = ROOT.TLegend(0.2, 0.65, 0.95, 0.9)
                    legend.SetFillStyle(0)
                    legend.SetBorderSize(0)
                    legend.SetTextSize(0.05)
                    legend.AddEntry(pbpb, "PbPb Result", "lp")
                    legend.AddEntry(pp, "PP Result", "lp")
                    legend.AddEntry(pbpb_nc, "PbPb New Cone", "lp")
                    legend.AddEntry(pp_hc, "PP Hadron Cone", "lp")
                    legend.Draw()

        comparison_dir.cd()
        grid_canvas.Write()
        grid_canvas.Close()
        
    return len(pbpb_results)

def copy_and_rename_histogram(hist, output_file, directory_name):
    if '.' in hist.GetName() or '0' in hist.GetName():
        new_name = hist.GetName().replace('.', 'p').replace('0', 'o')
        new_hist = hist.Clone(new_name)
        output_file.cd(directory_name)
        new_hist.Write()
        
def process_data(rebin_factor=2):
    input_file = ROOT.TFile.Open("/home/Energy_energy_correlators_scripts/Ap_comparison_macros/EEC_embed.root", "READ")
    output_file = ROOT.TFile.Open("/home/Energy_energy_correlators_scripts/Ap_comparison_macros/EEC_embed_oper.root", "RECREATE")
    
    print("=== DEBUG: Available histograms in input file ===")
    input_file.ls()
    print("=== END DEBUG ===")

    #for canvas by n
    CANVAS_WIDTH = 7200          # Canvas width
    CANVAS_HEIGHT = 4800         # Canvas height  
    HISTOGRAM_TITLE_SIZE = 0.2   # Title size for histograms
    AXIS_TITLE_SIZE = 0.2        # X and Y axis title size
    MARKER_SIZE = 0.5            # Marker size for all histograms
    LEGEND_TEXT_SIZE = 0.005     # Legend text size
    
    pbpb_jet_counts = None
    pp_jet_counts = None

    pbpb_jet_counts = input_file.Get("JetCounts/pbpb_jet_counts")
    pp_jet_counts = input_file.Get("JetCounts/pp_jet_counts")

    print(f"First attempt - PbPb jet counts: {pbpb_jet_counts}")
    print(f"First attempt - PP jet counts: {pp_jet_counts}")

    if not pbpb_jet_counts or not pp_jet_counts:
        pbpb_jet_counts = input_file.Get("JetCountsandFallBack/pbpb_jet_counts")
        pp_jet_counts = input_file.Get("JetCountsandFallBack/pp_jet_counts")
        print(f"Fallback attempt - PbPb jet counts: {pbpb_jet_counts}")
        print(f"Fallback attempt - PP jet counts: {pp_jet_counts}")

    if not pbpb_jet_counts or not pp_jet_counts:
        print("Error: Could not find jet counts in either location")
        input_file.ls()
        return

    pbpb_n_jets = pbpb_jet_counts.GetBinContent(1)
    pp_n_jets = pp_jet_counts.GetBinContent(1)

    print(f"PbPb number of jets: {pbpb_n_jets}")
    print(f"PP number of jets: {pp_n_jets}")
    pbpb_jet_counts_pt = {}
    pp_jet_counts_pt = {}

    jetpt_ranges_for_counts = ["jetpt0to500" ]

    jetpt_to_bin = {
        "jetpt0to500": 1,
        
        
    }

    for jetpt in jetpt_ranges_for_counts:
        bin_number = jetpt_to_bin[jetpt]
        pbpb_jet_counts_pt[jetpt] = pbpb_jet_counts.GetBinContent(bin_number)
        pp_jet_counts_pt[jetpt] = pp_jet_counts.GetBinContent(bin_number)
        print(f"PbPb {jetpt} jet count (bin {bin_number}): {pbpb_jet_counts_pt[jetpt]}")
        print(f"PP {jetpt} jet count (bin {bin_number}): {pp_jet_counts_pt[jetpt]}")
    collision_types = ["pbpb", "pp"]
    directories = ["signal", "sm1", "m1m1", "m1m2", "spec_m1"]
    n_values = ["n0", "n1", "n2"]
    jetpt_ranges = ["jetpt0to500"]
    pch_values = ["pch0.0", "pch1.0", "pch2.0", "pch3.0"]
    
    pbpb_dirs = ["signal", "sm1", "m1m1", "m1m2", "spec_m1", "newcon_cone"]
    pp_dirs = ["signal", "sm1", "m1m1", "m1m2", "hadron_cone"] 
    
    output_file.mkdir("raw_plots")
    output_file.mkdir("subtracted_raw")
    output_file.mkdir("subtracted_jet_normalized")
    output_file.mkdir("subtracted_jet_normalized_rel_binwidth")
    output_file.mkdir("canvases_by_n")
    output_file.mkdir("cone_raw_plots")
    output_file.mkdir("combined_comparison_plots")
    output_file.mkdir("raw_plots_jet_normalized")
    output_file.mkdir("added_by_jetpt")
    output_file.mkdir("renamed_histograms")
    output_file.mkdir("added_jetpt_comparison_plots")
    
    pbpb_results = {}
    pp_results = {}
    pbpb_newcon_cone_processed = {}
    pp_hadron_cone_processed = {}
    added_by_jetpt = {}
    added_by_pch = {}
    
    for n in n_values:
        for jetpt in jetpt_ranges:
            for pch in pch_values:
                
                pbpb_hists = []
                pp_hists = []
                
                for dir_name in pbpb_dirs:
                    if dir_name == "newcon_cone":
                        hist_name = f"pbpb_{dir_name}_eec_{n}_{jetpt}_{pch}"
                    else:
                        hist_name = f"pbpb_{dir_name}_eec_{n}_{jetpt}_{pch}"
                    path = f"PbPb/{dir_name}/{hist_name}"
                    hist = input_file.Get(path)
                    if hist:
                        cloned_hist = hist.Clone(f"{hist_name}_raw")
                        
                        cloned_hist.SetDirectory(0)
                        if rebin_factor > 1:
                            cloned_hist.Rebin(rebin_factor)
                        cloned_hist.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                        pbpb_hists.append(cloned_hist)
                
                for i, dir_name in enumerate(pp_dirs):
                    if dir_name == "hadron_cone":
                        hist_name = f"pp_{dir_name}_eec_{n}_{jetpt}_{pch}"
                    else:
                        hist_name = f"pp_{dir_name}_eec_{n}_{jetpt}_{pch}"
                    path = f"PP/{dir_name}/{hist_name}"
                    hist = input_file.Get(path)
                    if hist:
                        cloned_hist = hist.Clone(f"{hist_name}_raw")
                        cloned_hist.SetDirectory(0)
                        if rebin_factor > 1:
                            cloned_hist.Rebin(rebin_factor)
                        cloned_hist.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                        pp_hists.append(cloned_hist)
                
                pbpb_cone_hists = []
                pp_cone_hists = []
                
                if len(pbpb_hists) > 5:  
                    pbpb_cone_hists.append(pbpb_hists[5])
                    
                    pbpb_cone_normalized = pbpb_hists[5].Clone()
                    normalize_by_jets(pbpb_cone_normalized, pbpb_jet_counts_pt[jetpt])
                    normalize_by_rel_bin_width(pbpb_cone_normalized)
                    pbpb_newcon_cone_processed[f"pbpb_newcon_cone_eec_{n}_{jetpt}_{pch}"] = pbpb_cone_normalized
                if len(pp_hists) > 4:  
                    pp_cone_hists.append(pp_hists[4])
                    
                    pp_cone_normalized = pp_hists[4].Clone()
                    normalize_by_jets(pp_cone_normalized, pp_jet_counts_pt[jetpt])
                    normalize_by_rel_bin_width(pp_cone_normalized)
                    pp_hadron_cone_processed[f"pp_hadron_cone_eec_{n}_{jetpt}_{pch}"] = pp_cone_normalized
                if pbpb_hists:
                    plot_histograms(pbpb_hists, f"pbpb_raw_{n}_{jetpt}_{pch}", f"PbPb Raw {n} {jetpt} {pch}", "raw_plots", output_file)
                    
                    pbpb_hists_jet_norm = []
                    for hist in pbpb_hists:
                        if hist:
                            normalized_hist = hist.Clone(f"{hist.GetName()}_jet_norm")
                            normalize_by_jets(normalized_hist, pbpb_jet_counts_pt[jetpt])
                            pbpb_hists_jet_norm.append(normalized_hist)
                    
                    if pbpb_hists_jet_norm:
                        plot_histograms(pbpb_hists_jet_norm, f"pbpb_raw_jet_norm_{n}_{jetpt}_{pch}", 
                                    f"PbPb Raw Jet Normalized {n} {jetpt} {pch}", "raw_plots_jet_normalized", output_file)

                if pp_hists:
                    plot_histograms(pp_hists, f"pp_raw_{n}_{jetpt}_{pch}", f"PP Raw {n} {jetpt} {pch}", "raw_plots", output_file)
                    
                    pp_hists_jet_norm = []
                    for hist in pp_hists:
                        if hist:
                            normalized_hist = hist.Clone(f"{hist.GetName()}_jet_norm")
                            normalize_by_jets(normalized_hist, pp_jet_counts_pt[jetpt])
                            pp_hists_jet_norm.append(normalized_hist)
                    
                    if pp_hists_jet_norm:
                        plot_histograms(pp_hists_jet_norm, f"pp_raw_jet_norm_{n}_{jetpt}_{pch}", 
                                    f"PP Raw Jet Normalized {n} {jetpt} {pch}", "raw_plots_jet_normalized", output_file)
                if pbpb_cone_hists:
                    plot_histograms(pbpb_cone_hists, f"pbpb_cone_raw_{n}_{jetpt}_{pch}", f"PbPb Cone Raw {n} {jetpt} {pch}", "cone_raw_plots", output_file)
                if pp_cone_hists:
                    plot_histograms(pp_cone_hists, f"pp_cone_raw_{n}_{jetpt}_{pch}", f"PP Cone Raw {n} {jetpt} {pch}", "cone_raw_plots", output_file)
                
                if len(pbpb_hists) >= 4:
                    s = pbpb_hists[0]
                    sm1 = pbpb_hists[1]
                    m1m1 = pbpb_hists[2]
                    m1m2 = pbpb_hists[3]
                    
                    subtracted_raw = subtract_histograms(s, sm1, m1m2, m1m1, f"pbpb_subtracted_raw_{n}_{jetpt}_{pch}")
                    subtracted_raw.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                    
                    output_file.cd("subtracted_raw")
                    
                    subtracted_raw.Write()
                    
                    subtracted_jet_norm = subtracted_raw.Clone(f"pbpb_subtracted_jet_norm_{n}_{jetpt}_{pch}")
                    normalize_by_jets(subtracted_jet_norm, pbpb_jet_counts_pt[jetpt])
                    subtracted_jet_norm.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                    
                    output_file.cd("subtracted_jet_normalized")
                    subtracted_jet_norm.Write()
                    
                    subtracted_jet_norm_rel_binwidth = subtracted_jet_norm.Clone(f"pbpb_subtracted_jet_norm_rel_binwidth_{n}_{jetpt}_{pch}")
                    normalize_by_rel_bin_width(subtracted_jet_norm_rel_binwidth)
                    subtracted_jet_norm_rel_binwidth.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                    
                    output_file.cd("subtracted_jet_normalized_rel_binwidth")
                    subtracted_jet_norm_rel_binwidth.Write()
                    
                    clone = subtracted_jet_norm_rel_binwidth.Clone()
                    clone.SetDirectory(0)
                    pbpb_results[f"subtracted_jet_norm_rel_binwidth_{n}_{jetpt}_{pch}"] = clone
                    copy_and_rename_histogram(subtracted_jet_norm_rel_binwidth, output_file, "renamed_histograms")
                    
                if len(pp_hists) >= 4:
                    s = pp_hists[0]
                    sm1 = pp_hists[1]
                    m1m1 = pp_hists[2]
                    m1m2 = pp_hists[3]
                    
                    subtracted_raw = subtract_histograms(s, sm1, m1m2, m1m1, f"pp_subtracted_raw_{n}_{jetpt}_{pch}")
                    subtracted_raw.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                    
                    output_file.cd("subtracted_raw")
                    
                    subtracted_raw.Write()
                    
                    subtracted_jet_norm = subtracted_raw.Clone(f"pp_subtracted_jet_norm_{n}_{jetpt}_{pch}")
                    normalize_by_jets(subtracted_jet_norm, pp_jet_counts_pt[jetpt])
                    subtracted_jet_norm.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                    
                    output_file.cd("subtracted_jet_normalized")
                    subtracted_jet_norm.Write()
                    
                    subtracted_jet_norm_rel_binwidth = subtracted_jet_norm.Clone(f"pp_subtracted_jet_norm_rel_binwidth_{n}_{jetpt}_{pch}")
                    normalize_by_rel_bin_width(subtracted_jet_norm_rel_binwidth)
                    subtracted_jet_norm_rel_binwidth.GetXaxis().SetTitle("|r_{i}-r_{j}|")
                    
                    output_file.cd("subtracted_jet_normalized_rel_binwidth")
                    subtracted_jet_norm_rel_binwidth.Write()
                    
                    pp_results[f"subtracted_jet_norm_rel_binwidth_{n}_{jetpt}_{pch}"] = subtracted_jet_norm_rel_binwidth.Clone()
                    copy_and_rename_histogram(subtracted_jet_norm_rel_binwidth, output_file, "renamed_histograms")

    for n in n_values:
        output_file.cd("canvases_by_n")
        canvas_name = f"canvas_{n}"
        # Use control variables for canvas size
        c = ROOT.TCanvas(canvas_name, f"Subtracted Jet Normalized First BinWidth {n}", CANVAS_WIDTH, CANVAS_HEIGHT)
        c.Divide(3, 3, 0.001, 0.001)  # 3 columns (jetpt ranges) x 3 rows (pch values)
        
        # First pass: collect all histograms and find y-ranges for each pch value
        pch_y_ranges = {}
        all_hists = {}
        
        for pch in pch_values:
            max_val = -1e10
            min_val = 1e10
            
            for jetpt in jetpt_ranges:
                pbpb_hist_name = f"pbpb_subtracted_jet_norm_rel_binwidth_{n}_{jetpt}_{pch}"
                pp_hist_name = f"pp_subtracted_jet_norm_rel_binwidth_{n}_{jetpt}_{pch}"
                
                pbpb_hist = output_file.Get(f"subtracted_jet_normalized_rel_binwidth/{pbpb_hist_name}")
                pp_hist = output_file.Get(f"subtracted_jet_normalized_rel_binwidth/{pp_hist_name}")
                
                # Get cone histograms from the processed dictionaries
                pbpb_newcon_key = f"pbpb_newcon_cone_eec_{n}_{jetpt}_{pch}"
                pp_hadron_key = f"pp_hadron_cone_eec_{n}_{jetpt}_{pch}"
                
                pbpb_newcon_hist = pbpb_newcon_cone_processed.get(pbpb_newcon_key)
                pp_hadron_hist = pp_hadron_cone_processed.get(pp_hadron_key)
                
                # Check if objects exist and are histograms
                if pbpb_hist and not pbpb_hist.InheritsFrom("TH1"):
                    print(f"Warning: {pbpb_hist_name} is not a histogram")
                    pbpb_hist = None
                    
                if pp_hist and not pp_hist.InheritsFrom("TH1"):
                    print(f"Warning: {pp_hist_name} is not a histogram")
                    pp_hist = None
                
                # Store histograms (now including cone histograms)
                all_hists[(pch, jetpt)] = (pbpb_hist, pp_hist, pbpb_newcon_hist, pp_hadron_hist)
                
                # Find min/max for this pch value (including cone histograms)
                if pbpb_hist:
                    max_val = max(max_val, pbpb_hist.GetMaximum())
                    min_val = min(min_val, pbpb_hist.GetMinimum())
                if pp_hist:
                    max_val = max(max_val, pp_hist.GetMaximum())
                    min_val = min(min_val, pp_hist.GetMinimum())
                if pbpb_newcon_hist:
                    max_val = max(max_val, pbpb_newcon_hist.GetMaximum())
                    min_val = min(min_val, pbpb_newcon_hist.GetMinimum())
                if pp_hadron_hist:
                    max_val = max(max_val, pp_hadron_hist.GetMaximum())
                    min_val = min(min_val, pp_hadron_hist.GetMinimum())
            
            # Store y-range for this pch value (with some padding)
            if max_val > -1e10 and min_val < 1e10:
                pch_y_ranges[pch] = (min_val * 1.1 if min_val < 0 else min_val * 0.9, max_val * 1.1)
        
        # Second pass: create plots with consistent y-ranges per pch row
        pad_counter = 1
        
        for pch_idx, pch in enumerate(pch_values):  # Rows (pch values)
            for jetpt_idx, jetpt in enumerate(jetpt_ranges):  # Columns (jetpt ranges)
                c.cd(pad_counter)
                ROOT.gPad.SetGridx()
                ROOT.gPad.SetGridy()
                ROOT.gPad.SetLogx()
                ROOT.gPad.SetLeftMargin(0.12)   
                ROOT.gPad.SetRightMargin(0.02)  
                ROOT.gPad.SetTopMargin(0.08)    
                ROOT.gPad.SetBottomMargin(0.10)
                
                # Unpack all four histograms
                pbpb_hist, pp_hist, pbpb_newcon_hist, pp_hadron_hist = all_hists[(pch, jetpt)]
                
                jetpt_display = jetpt.replace("jetpt", "").replace("to", "-")
                pch_display = pch.replace("pch", "")
                pad_title = f"Jet pT: {jetpt_display} GeV, p_{{ch}} > {pch_display} GeV"
                
                # Set consistent y-range for this pch row
                if pch in pch_y_ranges:
                    min_val, max_val = pch_y_ranges[pch]
                    if pbpb_hist:
                        pbpb_hist.SetMaximum(max_val)
                        pbpb_hist.SetMinimum(min_val)
                    if pp_hist:
                        pp_hist.SetMaximum(max_val)
                        pp_hist.SetMinimum(min_val)
                    if pbpb_newcon_hist:
                        pbpb_newcon_hist.SetMaximum(max_val)
                        pbpb_newcon_hist.SetMinimum(min_val)
                    if pp_hadron_hist:
                        pp_hadron_hist.SetMaximum(max_val)
                        pp_hadron_hist.SetMinimum(min_val)
                
                # Plot PbPb result - USE CONTROL VARIABLES
                if pbpb_hist:
                    pbpb_hist.SetLineColor(2)  # Red
                    pbpb_hist.SetMarkerColor(2)
                    pbpb_hist.SetLineWidth(2)
                    pbpb_hist.SetMarkerStyle(20)
                    pbpb_hist.SetMarkerSize(MARKER_SIZE)  # CONTROLLED MARKER SIZE
                    pbpb_hist.GetXaxis().SetTitle("#Delta r")
                    pbpb_hist.GetYaxis().SetTitle("EEC")
                    pbpb_hist.SetTitle(pad_title)
                    pbpb_hist.SetTitleSize(HISTOGRAM_TITLE_SIZE) 
                    pbpb_hist.GetXaxis().SetTitleSize(AXIS_TITLE_SIZE)    # CONTROLLED AXIS TITLE SIZE
                    pbpb_hist.GetYaxis().SetTitleSize(AXIS_TITLE_SIZE)    # CONTROLLED AXIS TITLE SIZE  
                    pbpb_hist.GetXaxis().SetLabelSize(0.05)
                    pbpb_hist.GetYaxis().SetLabelSize(0.05)
                    pbpb_hist.Draw("E1")
                
                # Plot PP result - USE CONTROL VARIABLES
                if pp_hist:
                    pp_hist.SetLineColor(4)  # Blue
                    pp_hist.SetMarkerColor(4)
                    pp_hist.SetLineWidth(2)
                    pp_hist.SetMarkerStyle(21)
                    pp_hist.SetMarkerSize(MARKER_SIZE)  # CONTROLLED MARKER SIZE
                    pp_hist.GetXaxis().SetTitle("#Delta r")
                    pp_hist.GetYaxis().SetTitle("EEC")
                    if pbpb_hist:
                        pp_hist.Draw("E1 SAME")
                    else:
                        pp_hist.SetTitle(pad_title)
                        pp_hist.SetTitleSize(HISTOGRAM_TITLE_SIZE)
                        pp_hist.GetXaxis().SetTitleSize(AXIS_TITLE_SIZE)    # CONTROLLED AXIS TITLE SIZE
                        pp_hist.GetYaxis().SetTitleSize(AXIS_TITLE_SIZE)    # CONTROLLED AXIS TITLE SIZE
                        pp_hist.GetXaxis().SetLabelSize(0.04)
                        pp_hist.GetYaxis().SetLabelSize(0.04)
                        pp_hist.Draw("E1")
                
                # Plot PbPb NewCon cone - USE CONTROL VARIABLES
                if pbpb_newcon_hist:
                    pbpb_newcon_hist.SetLineColor(ROOT.kGreen+2)  # Green
                    pbpb_newcon_hist.SetMarkerColor(ROOT.kGreen+2)
                    pbpb_newcon_hist.SetLineWidth(2)
                    pbpb_newcon_hist.SetLineStyle(2)  # Dashed line
                    pbpb_newcon_hist.SetMarkerStyle(22)
                    pbpb_newcon_hist.SetMarkerSize(MARKER_SIZE * 0.8)  # CONTROLLED MARKER SIZE (slightly smaller)
                    if pbpb_hist or pp_hist:
                        pbpb_newcon_hist.Draw("E1 SAME")
                    else:
                        pbpb_newcon_hist.SetTitle(pad_title)
                        pbpb_newcon_hist.GetXaxis().SetTitle("#Delta r")
                        pbpb_newcon_hist.GetYaxis().SetTitle("EEC")
                        pbpb_newcon_hist.GetXaxis().SetTitleSize(AXIS_TITLE_SIZE)  # CONTROLLED AXIS TITLE SIZE
                        pbpb_newcon_hist.GetYaxis().SetTitleSize(AXIS_TITLE_SIZE)  # CONTROLLED AXIS TITLE SIZE
                        pbpb_newcon_hist.GetXaxis().SetLabelSize(0.04)
                        pbpb_newcon_hist.GetYaxis().SetLabelSize(0.04)
                        pbpb_newcon_hist.Draw("E1")
                
                # Plot PP Hadron cone - USE CONTROL VARIABLES
                if pp_hadron_hist:
                    pp_hadron_hist.SetLineColor(ROOT.kMagenta)  # Magenta
                    pp_hadron_hist.SetMarkerColor(ROOT.kMagenta)
                    pp_hadron_hist.SetLineWidth(2)
                    pp_hadron_hist.SetLineStyle(2)  # Dashed line
                    pp_hadron_hist.SetMarkerStyle(23)
                    pp_hadron_hist.SetMarkerSize(MARKER_SIZE * 0.8)  # CONTROLLED MARKER SIZE (slightly smaller)
                    if pbpb_hist or pp_hist or pbpb_newcon_hist:
                        pp_hadron_hist.Draw("E1 SAME")
                    else:
                        pp_hadron_hist.SetTitle(pad_title)
                        pp_hadron_hist.GetXaxis().SetTitle("#Delta r")
                        pp_hadron_hist.GetYaxis().SetTitle("EEC")
                        pp_hadron_hist.GetXaxis().SetTitleSize(AXIS_TITLE_SIZE)  # CONTROLLED AXIS TITLE SIZE
                        pp_hadron_hist.GetYaxis().SetTitleSize(AXIS_TITLE_SIZE)  # CONTROLLED AXIS TITLE SIZE
                        pp_hadron_hist.GetXaxis().SetLabelSize(0.04)
                        pp_hadron_hist.GetYaxis().SetLabelSize(0.04)
                        pp_hadron_hist.Draw("E1")
                
                # REMOVED INDIVIDUAL LEGEND CODE FROM HERE
                
                # Draw zero line
                reference_hist = None
                for hist in [pbpb_hist, pp_hist, pbpb_newcon_hist, pp_hadron_hist]:
                    if hist and hist.InheritsFrom("TH1"):
                        reference_hist = hist
                        break
                
                if reference_hist:
                    zero_line = ROOT.TLine(reference_hist.GetXaxis().GetXmin(), 
                                        0, 
                                        reference_hist.GetXaxis().GetXmax(), 
                                        0)
                    zero_line.SetLineStyle(3)
                    zero_line.SetLineColor(ROOT.kGray+1)
                    zero_line.SetLineWidth(1)
                    zero_line.Draw("same")

                pad_counter += 1
        
        # CREATE SINGLE LEGEND ON THE MAIN CANVAS (OUTSIDE THE PADS)
        c.cd()  # Go back to main canvas
        
        # Create a single legend positioned in the top-right area of the canvas
        # Adjust these coordinates based on your canvas layout
        single_legend = ROOT.TLegend(0.75, 0.85, 0.98, 0.98)
        single_legend.SetFillStyle(1001)
        single_legend.SetFillColor(ROOT.kWhite)
        single_legend.SetBorderSize(1)
        single_legend.SetTextSize(0.015)  # Adjust text size as needed
        single_legend.SetTextFont(42)
        
        # Add entries with your requested labels
        # Create dummy histograms for legend entries with correct styling
        dummy_pbpb = ROOT.TH1F(f"dummy_pbpb_{n}", "", 1, 0, 1)
        dummy_pbpb.SetLineColor(2)
        dummy_pbpb.SetMarkerColor(2)
        dummy_pbpb.SetMarkerStyle(20)
        dummy_pbpb.SetLineWidth(2)
        
        dummy_pp = ROOT.TH1F(f"dummy_pp_{n}", "", 1, 0, 1)
        dummy_pp.SetLineColor(4)
        dummy_pp.SetMarkerColor(4)
        dummy_pp.SetMarkerStyle(21)
        dummy_pp.SetLineWidth(2)
        
        dummy_newcon = ROOT.TH1F(f"dummy_newcon_{n}", "", 1, 0, 1)
        dummy_newcon.SetLineColor(ROOT.kGreen+2)
        dummy_newcon.SetMarkerColor(ROOT.kGreen+2)
        dummy_newcon.SetMarkerStyle(22)
        dummy_newcon.SetLineWidth(2)
        dummy_newcon.SetLineStyle(2)
        
        dummy_hadron = ROOT.TH1F(f"dummy_hadron_{n}", "", 1, 0, 1)
        dummy_hadron.SetLineColor(ROOT.kMagenta)
        dummy_hadron.SetMarkerColor(ROOT.kMagenta)
        dummy_hadron.SetMarkerStyle(23)
        dummy_hadron.SetLineWidth(2)
        dummy_hadron.SetLineStyle(2)
        
        # Add entries with your requested labels
        single_legend.AddEntry(dummy_pbpb, "PbPb (subtracted)", "lp")
        single_legend.AddEntry(dummy_pp, "PP embedded (subtracted)", "lp")
        single_legend.AddEntry(dummy_newcon, "PbPb Jet (actual)", "lp")
        single_legend.AddEntry(dummy_hadron, "PP Jet (actual)", "lp")
        
        single_legend.Draw()
        
        c.Write()
    create_combined_comparison_plots(pbpb_results, pp_results, pbpb_newcon_cone_processed, pp_hadron_cone_processed, output_file)

    added_by_jetpt_pbpb = {}
    added_by_jetpt_pp = {}
    added_by_jetpt_pbpb_newcon = {}
    added_by_jetpt_pp_hadron = {}

    for collision in ["pbpb", "pp"]:
        results = pbpb_results if collision == "pbpb" else pp_results
        cone_results = pbpb_newcon_cone_processed if collision == "pbpb" else pp_hadron_cone_processed
        
        for n in n_values:
            for pch in pch_values:
                jetpt_key = f"{collision}_added_jetpt_{n}_{pch}"
                sum_hist = None
                for jetpt in jetpt_ranges:  
                    hist_key = f"subtracted_jet_norm_rel_binwidth_{n}_{jetpt}_{pch}"
                    if hist_key in results:
                        if sum_hist is None:
                            sum_hist = results[hist_key].Clone(jetpt_key)
                        else:
                            sum_hist.Add(results[hist_key])
                if sum_hist:
                    output_file.cd("added_by_jetpt")
                    sum_hist.Write()
                    added_by_jetpt[jetpt_key] = sum_hist.Clone()
                    
                    if collision == "pbpb":
                        added_by_jetpt_pbpb[f"{n}_{pch}"] = sum_hist.Clone()
                    else:
                        added_by_jetpt_pp[f"{n}_{pch}"] = sum_hist.Clone()
                
                cone_sum_hist = None
                for jetpt in jetpt_ranges:
                    if collision == "pbpb":
                        cone_key = f"pbpb_newcon_cone_eec_{n}_{jetpt}_{pch}"
                    else:
                        cone_key = f"pp_hadron_cone_eec_{n}_{jetpt}_{pch}"
                    
                    if cone_key in cone_results:
                        if cone_sum_hist is None:
                            cone_sum_hist = cone_results[cone_key].Clone(f"{collision}_cone_added_jetpt_{n}_{pch}")
                        else:
                            cone_sum_hist.Add(cone_results[cone_key])
                
                if cone_sum_hist:
                    if collision == "pbpb":
                        added_by_jetpt_pbpb_newcon[f"{n}_{pch}"] = cone_sum_hist.Clone()
                    else:
                        added_by_jetpt_pp_hadron[f"{n}_{pch}"] = cone_sum_hist.Clone()
        
        
    plot_added_histograms(added_by_jetpt, output_file, "added_jetpt_plots")
    create_added_jetpt_comparison_plots(added_by_jetpt_pbpb, added_by_jetpt_pp, 
                                   added_by_jetpt_pbpb_newcon, added_by_jetpt_pp_hadron, 
                                   output_file, jetpt_ranges, pch_values)
    
    output_file.Close()
    input_file.Close()

if __name__ == "__main__":
    rebin_factor = 2
    process_data(rebin_factor)