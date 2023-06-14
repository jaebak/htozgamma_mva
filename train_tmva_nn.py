#!/usr/bin/env python3
from ROOT import TMVA, TFile, TTree, TCut, TChain, TH1F, TString, TCanvas
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
import math
import numpy as np
import array
import time

def train_nn(folder, train_filename, test_filename, tmva_filename):
  output = TFile.Open(tmva_filename, 'RECREATE')
  factory = TMVA.Factory('TMVAClassification', output,
                         '!V:ROC:!Correlations:!Silent:Color:'
                         '!DrawProgressBar:AnalysisType=Classification')
  dataloader = TMVA.DataLoader(folder)
  dataloader.AddVariable("min_dR",'F')
  dataloader.AddVariable("max_dR",'F')
  dataloader.AddVariable("pt_mass",'F')
  dataloader.AddVariable("cosTheta",'F')
  dataloader.AddVariable("costheta",'F')
  dataloader.AddVariable("phi",'F')
  dataloader.AddVariable("photon_rapidity",'F')
  dataloader.AddVariable("l1_rapidity",'F')
  dataloader.AddVariable("l2_rapidity",'F')
  dataloader.AddVariable("llg_flavor",'F')
  dataloader.AddVariable("llg_ptt",'F')
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


  layoutString = TString("Layout=TANH|40,LINEAR")
  #trainingString = TString("LearningRate=1e-6,ConvergenceSteps=100,BatchSize=1,TestRepetitions=10,MaxEpochs=10")
  #trainingString = TString("LearningRate=1e-6,ConvergenceSteps=100,BatchSize=1,TestRepetitions=10,MaxEpochs=3000")
  trainingString = TString("LearningRate=1e-6,ConvergenceSteps=100,BatchSize=1,TestRepetitions=10,MaxEpochs=80000")
  #trainingString = TString("LearningRate=1e-6,ConvergenceSteps=100,BatchSize=1,TestRepetitions=10,MaxEpochs=100")
  trainingStrategyString = TString("TrainingStrategy=")
  trainingStrategyString += trainingString;
  dnnOptionString = TString("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM")
  dnnOptionString.Append (":"); dnnOptionString.Append (layoutString);
  dnnOptionString.Append (":"); dnnOptionString.Append (trainingStrategyString);
  cpuOptionString = dnnOptionString + TString(":Architecture=CPU")
  #cpuOptionString = dnnOptionString + TString(":Architecture=GPU")

  factory.BookMethod(dataloader,TMVA.Types.kDL,"DNN",cpuOptionString);
  factory.TrainAllMethods();
  factory.TestAllMethods();
  factory.EvaluateAllMethods();
  output.Close()

if __name__ == "__main__":
  start_time = time.time()
  train_nn(folder='tmva_nn_test',
           train_filename="mva_input_ntuples/train_sample_run2_lumi.root", 
           test_filename="mva_input_ntuples/test_sample_run2_lumi.root", 
           tmva_filename="mva_output_ntuples/TMVA_nn.root")
  elapsed_time = time.time() - start_time
  print(f'Training time: {elapsed_time}')

  #TMVA::TMVAGui("ntuples_mva/TMVA_bdt.root")
