#!/usr/bin/env python3
import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)

if __name__ == '__main__':
  ROOT.EnableImplicitMT()

  #input_filename = 'ntuples/train_decorr_sig.root'
  #input_filename = 'ntuples/train_decorr_sig_run2.root'
  input_filename = 'ntuples/train_decorr_sig_run2_lumi.root'
  input_tree = 'tree'
  #output_filename = 'ntuples/train_decorr_sig_shapewgt.root'
  #output_filename = 'ntuples/train_decorr_sig_shapewgt_run2.root'
  output_filename = 'ntuples/train_decorr_sig_shapewgt_run2_lumi.root'

  defines = [('w_llg_mass','1.'),
             #('weight', 'w_lumi*w_llg_mass')]
             ('w_lumiXyearXshape', 'w_lumiXyear*w_llg_mass')]
  cuts = ['1',]
  branches = ()

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
