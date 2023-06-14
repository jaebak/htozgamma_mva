#!/usr/bin/env python3
import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)
import rootutils

if __name__ == '__main__':
  ROOT.EnableImplicitMT()

  print('Make weights that converts input distribution into target distribution')
  #target_filename = 'ntuples/train_decorr_sig.root'
  #target_filename = 'ntuples/train_decorr_sig_run2.root'
  target_filename = 'ntuples/train_decorr_sig_run2_lumi.root'
  target_tree = 'tree'
  target_branch = 'llg_mass'
  target_weight = 'w_lumiXyear'
  target_cut = '1'

  #input_filename = 'ntuples/train_decorr_bak.root'
  #input_filename = 'ntuples/train_decorr_bak_run2.root'
  input_filename = 'ntuples/train_decorr_bak_run2_lumi.root'
  input_tree = 'tree'
  input_branch = 'llg_mass'
  input_weight = 'w_lumiXyear'
  input_cut = '1'

  nbins = 160
  output_weight_filename = 'shape_wgt_run2_lumi.root'
  #output_filename = 'ntuples/train_decorr_bak_shapewgt.root'
  #output_filename = 'ntuples/train_decorr_bak_shapewgt_run2.root'
  output_filename = 'ntuples/train_decorr_bak_shapewgt_run2_lumi.root'
  output_weight = 'w_lumiXyearXshape'

  # Open files
  target_chain = ROOT.TChain(target_tree)
  target_chain.Add(target_filename)

  input_chain = ROOT.TChain(input_tree)
  input_chain.Add(input_filename)

  # Find min max
  min_value = min(target_chain.GetMinimum(target_branch), input_chain.GetMinimum(input_branch))
  max_value = max(target_chain.GetMaximum(target_branch), input_chain.GetMaximum(input_branch))

  # Make histograms
  input_hist = ROOT.TH1F('input_hist', 'input_hist', nbins, min_value, max_value)
  input_chain.Draw(f'{input_branch} >> input_hist', f'({input_cut})*{input_weight}', 'goff')
  target_hist = ROOT.TH1F('target_hist', 'target_hist', nbins, min_value, max_value)
  target_chain.Draw(f'{target_branch} >> target_hist', f'({target_cut})*{target_weight}', 'goff')

  # Normalize hists
  rootutils.normalize_hist(input_hist)
  rootutils.normalize_hist(target_hist)

  # Make weights
  weight_hist = target_hist.Clone('weight_hist')
  weight_hist.Divide(input_hist)
  #print(weight_hist.GetBinContent(weight_hist.FindBin(120)))

  # Save weights to file
  weight_file = ROOT.TFile(output_weight_filename, 'recreate')
  weight_hist.Write()
  weight_file.Close()

  # Draw histograms
  c1 = rootutils.new_canvas()
  input_hist.Draw()
  c1.SaveAs(f'plots/input_shape.pdf')
  c2 = rootutils.new_canvas()
  target_hist.Draw()
  c2.SaveAs(f'plots/target_shape.pdf')
  c3 = rootutils.new_canvas()
  weight_hist.Draw()
  c3.SaveAs(f'plots/weight_shape.pdf')

  # Apply weights to input_file
  defines = [('w_llg_mass','get_shape_weight(llg_mass)'),
             (output_weight, f'{input_weight}*w_llg_mass')]
  cuts = ['1',]
  branches = ()

  ROOT.gInterpreter.Declare("""
  template <class C>
  using RVec = ROOT::VecOps::RVec<C>;
  
  TFile * weight_file = new TFile("shape_wgt.root");
  TH1F * weight_hist = (TH1F*)weight_file->Get("weight_hist");
  
  float get_shape_weight(const Float_t & llg_mass) {
    //std::cout<<llg_mass<<std::endl;
    //std::cout<<weight_hist->GetBinContent(weight_hist->FindBin(120))<<std::endl;
    float shape_weight = weight_hist->GetBinContent(weight_hist->FindBin(llg_mass));
    return shape_weight;
  }
  
  """)

  df = ROOT.RDataFrame(input_tree, input_filename)
  for define in defines:
    df = df.Define(define[0],define[1])
  for cut in cuts:
    df = df.Filter(cut)
  if (branches == ()):
    df.Snapshot(input_tree,output_filename)
  else:
    df.Snapshot(input_tree, output_filename, branches)
  print('Wrote '+output_filename)

  # Compare target and output hist
  output_chain = ROOT.TChain(input_tree)
  output_chain.Add(output_filename)
  output_hist = ROOT.TH1F('output_hist', 'output_hist', nbins, min_value, max_value)
  output_chain.Draw(f'{input_branch} >> output_hist', f'({input_cut})*{output_weight}', 'goff')
  rootutils.normalize_hist(output_hist)
  ROOT.gStyle.SetOptStat(0)
  c4 = rootutils.new_canvas()
  target_hist.SetLineColor(ROOT.kRed)
  target_hist.Draw()
  input_hist.SetLineColor(ROOT.kBlue)
  input_hist.Draw('same')
  output_hist.SetLineColor(ROOT.kBlue)
  output_hist.Draw('same')
  legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
  legend.AddEntry(target_hist)
  legend.AddEntry(input_hist)
  legend.AddEntry(output_hist)
  legend.Draw()
  c4.SaveAs('plots/input_mod_shape.pdf')

