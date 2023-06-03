#!/usr/bin/env python3
from ROOT import TMVA, TFile, TTree, TCut, TChain, TH1F, TString, TCanvas
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
import math
import numpy as np
import array

def train_bdt(folder, train_filename, test_filename, tmva_filename):
  output = TFile.Open(tmva_filename, 'RECREATE')
  factory = TMVA.Factory('TMVAClassification', output,
                         '!V:ROC:!Correlations:!Silent:Color:'
                         '!DrawProgressBar:AnalysisType=Classification')
  dataloader = TMVA.DataLoader(folder)
  dataloader.AddVariable("photon_mva",'F')
  dataloader.AddVariable("llg_ptt",'F')
  dataloader.AddVariable("llg_eta",'F')
  dataloader.AddVariable("llg_phi",'F')
  dataloader.AddVariable("z_eta",'F')
  dataloader.AddVariable("z_phi",'F')
  dataloader.AddVariable("l1_rapidity",'F')
  dataloader.AddVariable("l1_phi",'F')
  dataloader.AddVariable("l2_rapidity",'F')
  dataloader.AddVariable("l2_phi",'F')
  dataloader.AddVariable("gamma_eta",'F')
  dataloader.AddVariable("gamma_phi",'F')
  dataloader.SetBackgroundWeightExpression("w_lumiXyear")
  dataloader.SetSignalWeightExpression("w_lumiXyear")
  dataloader.AddSpectator("llg_mass", 'F')
  dataloader.AddSpectator("w_lumiXyear", 'F')

  # Add data.
  train_chain = TChain('train_tree')
  train_chain.Add(train_filename)
  test_chain = TChain('test_tree')
  test_chain.Add(test_filename)
  dataloader.AddTree(train_chain, 'Background', 1., 'classID==0', TMVA.Types.kTraining)
  dataloader.AddTree(train_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTraining)
  dataloader.AddTree(test_chain, 'Background', 1., 'classID==0', TMVA.Types.kTesting)
  dataloader.AddTree(test_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTesting)
  cut_s = TCut('llg_mass>120 && llg_mass < 130');
  cut_b = TCut('llg_mass>120 && llg_mass < 130');
  dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=NumEvents:ScaleWithPreselEff:!V");

  factory.BookMethod(dataloader,TMVA.Types.kBDT,"BDT","!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20");
  factory.TrainAllMethods();
  factory.TestAllMethods();
  factory.EvaluateAllMethods();
  output.Close()

if __name__ == "__main__":
  # Make options for kfold
  
  train_bdt(folder="raw_bdt", 
            train_filename="train_sample_run2_lumi.root", 
            test_filename="test_sample_run2_lumi.root", 
            tmva_filename="ntuples_mva/raw_bdt.root")

  #TMVA::TMVAGui("ntuples_mva/TMVA_bdt.root")
