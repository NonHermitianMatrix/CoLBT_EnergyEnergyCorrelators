"""
Uses the output root file of the EEC_embed_oper macro and gives fig. 10 pbpb/pp for q = 0.5 of https://arxiv.org/abs/2503.19993
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

input_file = ROOT.TFile.Open("/home/Energy_energy_correlators_scripts/Ap_comparison_macros/EEC_embed_augmented.root", "READ")
output_file = ROOT.TFile.Open("/home/Energy_energy_correlators_scripts/Ap_comparison_macros/EEC_consistancy_1check.root", "RECREATE")

if not output_file.GetDirectory("consistancy_1check"):
    output_file.mkdir("consistancy_1check")
output_file.cd("consistancy_1check")

n_values = ["n0", "n1", "n2"]
jetpt_ranges = ["jetpt0to500"]
pch_values = ["pch0.0", "pch1.0", "pch2.0", "pch3.0"]

pbpb_jet_counts = input_file.Get("JetCounts/pbpb_jet_counts")
pp_jet_counts = input_file.Get("JetCounts/pp_jet_counts")
pbpb_n_jets = pbpb_jet_counts.GetBinContent(1)
pp_n_jets = pp_jet_counts.GetBinContent(1)

for n in n_values:
    for jetpt in jetpt_ranges:
        for pch in pch_values:
            # PbPb subtracted
            s    = input_file.Get(f"PbPb/signal/pbpb_signal_eec_{n}_{jetpt}_{pch}")
            sm1  = input_file.Get(f"PbPb/sm1/pbpb_sm1_eec_{n}_{jetpt}_{pch}")
            m1m1 = input_file.Get(f"PbPb/m1m1/pbpb_m1m1_eec_{n}_{jetpt}_{pch}")
            m1m2 = input_file.Get(f"PbPb/m1m2/pbpb_m1m2_eec_{n}_{jetpt}_{pch}")
            if not (s and sm1 and m1m1 and m1m2):
                continue
            subtracted = subtract_histograms(s, sm1, m1m2, m1m1, f"pbpb_subtracted_{n}_{jetpt}_{pch}")
            normalize_by_jets(subtracted, pbpb_n_jets)
            normalize_by_rel_bin_width(subtracted)
            # PP hadron cone
            pp_hadron = input_file.Get(f"PP/hadron_cone/pp_hadron_cone_eec_{n}_{jetpt}_{pch}")
            if not pp_hadron:
                continue
            pp_hadron = pp_hadron.Clone(f"pp_hadron_cone_{n}_{jetpt}_{pch}_clone")
            normalize_by_jets(pp_hadron, pp_n_jets)
            normalize_by_rel_bin_width(pp_hadron)
            # Ratio
            ratio = subtracted.Clone(f"pbpb_subtracted_over_pp_hadron_cone_{n}_{jetpt}_{pch}")
            ratio.Divide(pp_hadron)
            ratio.SetTitle(f"PbPb(subtracted)/PP(hadron cone): {n} {jetpt} {pch}")
            ratio.GetXaxis().SetTitle("#Delta r")
            ratio.GetYaxis().SetTitle("PbPb(subtracted)/PP(hadron cone)")
            ratio.Write()
            # Difference
            diff = subtracted.Clone(f"pbpb_subtracted_minus_pp_hadron_cone_{n}_{jetpt}_{pch}")
            diff.Add(pp_hadron, -1)
            diff.SetTitle(f"PbPb(subtracted) - PP(hadron cone): {n} {jetpt} {pch}")
            diff.GetXaxis().SetTitle("#Delta r")
            diff.GetYaxis().SetTitle("PbPb(subtracted) - PP(hadron cone)")
            diff.Write()

output_file.Close()
input_file.Close()