#!/usr/bin/env python3
from ROOT import TMVA, TFile, TTree, TCut, TChain, TH1F, TString, TCanvas
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
import math
import numpy as np
import array

def train_bdt(tmva_filename):
  output = TFile.Open(tmva_filename, 'RECREATE')
  factory = TMVA.Factory('TMVAClassification', output,
                         '!V:ROC:!Correlations:!Silent:Color:'
                         '!DrawProgressBar:AnalysisType=Classification')
  dataloader = TMVA.DataLoader('dataset')
  dataloader.AddVariable("photon_mva",'F')
  dataloader.AddVariable("min_dR",'F')
  dataloader.AddVariable("pt_mass",'F')
  dataloader.AddVariable("cosTheta",'F')
  dataloader.AddVariable("costheta",'F')
  dataloader.AddVariable("photon_res",'F')
  dataloader.AddVariable("photon_rapidity",'F')
  dataloader.AddVariable("l1_rapidity",'F')
  dataloader.AddVariable("l2_rapidity",'F')
  dataloader.SetBackgroundWeightExpression("w_lumi")
  dataloader.SetSignalWeightExpression("w_lumi")
  #dataloader.SetBackgroundWeightExpression("1")
  dataloader.AddSpectator("llg_mass", 'F')
  dataloader.AddSpectator("w_lumi", 'F')

  ### Add tree
  #bkg_chain = TChain('tree')
  #bkg_chain.Add("ntuples/train_decorr_bak.root")
  ##bkg_chain = TChain('train_tree')
  ##bkg_chain.Add("bkg_sample.root")
  ##bkg_chain.Add("bkg_sample.root")
  #sig_chain = TChain('tree')
  #sig_chain.Add("ntuples/train_decorr_sig.root")
  ##sig_chain = TChain('train_tree')
  ##sig_chain.Add("signal_sample.root")
  #dataloader.AddSignalTree(sig_chain)
  #dataloader.AddBackgroundTree(bkg_chain)
  ###dataloader.AddSignalTree(sig_chain, 1)
  ###dataloader.AddBackgroundTree(bkg_chain, 1)
  #cut_s = TCut('llg_mass>120 && llg_mass < 130');
  #cut_b = TCut('llg_mass>120 && llg_mass < 130');
  #n_bkg = bkg_chain.GetEntries()
  #n_signal = sig_chain.GetEntries()
  #train_test_fraction = 0.2
  #nTrain_Signal = int(n_signal * train_test_fraction)
  #nTrain_Background = int(n_bkg * train_test_fraction)
  #nTest_Signal = n_signal - nTrain_Signal
  #nTest_Background = n_bkg - nTrain_Background
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"nTrain_Signal={nTrain_Signal}:nTrain_Background={nTrain_Background}:nTest_Signal={nTest_Signal}:nTest_Background={nTest_Background}:NormMode=NumEvents:ScaleWithPreselEff:!V");

  ## Add tree
  #bkg_chain = TChain('tree')
  #bkg_chain.Add("ntuples/train_decorr_bak_window.root")
  #sig_chain = TChain('tree')
  #sig_chain.Add("ntuples/train_decorr_sig_window.root")
  #dataloader.AddSignalTree(sig_chain)
  #dataloader.AddBackgroundTree(bkg_chain)
  #cut_s = TCut('llg_mass>120 && llg_mass < 130');
  #cut_b = TCut('llg_mass>120 && llg_mass < 130');
  #n_bkg = bkg_chain.GetEntries()
  #n_signal = sig_chain.GetEntries()
  #train_test_fraction = 0.2
  #nTrain_Signal = int(n_signal * train_test_fraction)
  #nTrain_Background = int(n_bkg * train_test_fraction)
  #nTest_Signal = n_signal - nTrain_Signal
  #nTest_Background = n_bkg - nTrain_Background
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"nTrain_Signal={nTrain_Signal}:nTrain_Background={nTrain_Background}:nTest_Signal={nTest_Signal}:nTest_Background={nTest_Background}:NormMode=NumEvents:ScaleWithPreselEff:!V:SplitSeed=100");

  #bkg_chain = TChain('tree')
  #bkg_chain.Add("ntuples/train_decorr_bak.root")
  #sig_chain = TChain('tree')
  #sig_chain.Add("ntuples/train_decorr_sig.root")
  #dataloader.AddSignalTree(sig_chain)
  #dataloader.AddBackgroundTree(bkg_chain)
  #cut_s = TCut('llg_mass>120 && llg_mass < 130');
  #cut_b = TCut('llg_mass>120 && llg_mass < 130');
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"nTrain_Signal=14410:nTrain_Background=37459:nTest_Signal=1:nTest_Background=1:NormMode=NumEvents:ScaleWithPreselEff:!V");

  #bkg_chain = TChain('train_tree')
  #bkg_chain.Add("bkg_sample.root")
  #sig_chain = TChain('train_tree')
  #sig_chain.Add("signal_sample.root")
  #dataloader.AddSignalTree(sig_chain)
  #dataloader.AddBackgroundTree(bkg_chain)
  #cut_s = TCut('llg_mass>120 && llg_mass < 130');
  #cut_b = TCut('llg_mass>120 && llg_mass < 130');
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"nTrain_Signal=0:nTrain_Background=0:nTest_Signal=1:nTest_Background=1:NormMode=NumEvents:ScaleWithPreselEff:!V");

  # Add data.
  train_chain = TChain('train_tree')
  train_chain.Add("train_sample.root")
  test_chain = TChain('test_tree')
  test_chain.Add("test_sample.root")
  dataloader.AddTree(train_chain, 'Background', 1., 'classID==0', TMVA.Types.kTraining)
  dataloader.AddTree(train_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTraining)
  dataloader.AddTree(test_chain, 'Background', 1., 'classID==0', TMVA.Types.kTesting)
  dataloader.AddTree(test_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTesting)
  cut_s = TCut('llg_mass>120 && llg_mass < 130');
  cut_b = TCut('llg_mass>120 && llg_mass < 130');
  dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=NumEvents:ScaleWithPreselEff:!V");

  ## Add data.
  #signal_train_chain = TChain('train_tree')
  #signal_train_chain.Add("signal_sample.root")
  #signal_test_chain = TChain('test_tree')
  #signal_test_chain.Add("signal_sample.root")
  #bkg_train_chain = TChain('train_tree')
  #bkg_train_chain.Add("bkg_sample.root")
  #bkg_test_chain = TChain('test_tree')
  #bkg_test_chain.Add("bkg_sample.root")
  #dataloader.AddTree(bkg_train_chain, 'Background', 1., '1', TMVA.Types.kTraining)
  #dataloader.AddTree(signal_train_chain, 'Signal', 1., '1', TMVA.Types.kTraining)
  #dataloader.AddTree(bkg_test_chain, 'Background', 1., '1', TMVA.Types.kTesting)
  #dataloader.AddTree(signal_test_chain, 'Signal', 1., '1', TMVA.Types.kTesting)
  #cut_s = TCut('llg_mass>120 && llg_mass < 130');
  #cut_b = TCut('llg_mass>120 && llg_mass < 130');
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=NumEvents:ScaleWithPreselEff:!V");


  #dataloader.SetInputTrees(chain, 'classID==1', 'classID==0')

  #cut_s = TCut('llg_mass>120 && llg_mass < 130');
  #cut_b = TCut('llg_mass>120 && llg_mass < 130');
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=2500:nTrain_Background=2500:nTest_Signal=20000:nTest_Background=20000:SplitMode=Random:NormMode=NumEvents:!V");
  # bkg = 187297, signal = 72053
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"nTrain_Signal={n_signal-1}:nTrain_Background={n_bkg-1}:nTest_Signal=1:nTest_Background=1:SplitMode=Random:NormMode=NumEvents:ScaleWithPreselEff:!V");
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"nTrain_Signal=0:nTrain_Background=0:nTest_Signal=1:nTest_Background=1:SplitMode=Random:NormMode=NumEvents:ScaleWithPreselEff:!V");
  ## Use 5% for training 95% for evaluation
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=3603:nTrain_Background=9365:nTest_Signal=68450:nTest_Background=177932:SplitMode=Random:NormMode=NumEvents:!V");
  ## Use 20% for training 80% for evaluation
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=14410:nTrain_Background=37459:nTest_Signal=57643:nTest_Background=149838:SplitMode=Random:NormMode=NumEvents:!V");
  ## Use 80% for training 20% for evaluation
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=57642:nTrain_Background=149837:nTest_Signal=14411:nTest_Background=37460:SplitMode=Random:NormMode=NumEvents:!V");

  factory.BookMethod(dataloader,TMVA.Types.kBDT,"BDT","!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20");
  factory.TrainAllMethods();
  factory.TestAllMethods();
  factory.EvaluateAllMethods();
  output.Close()

def train_nn(tmva_filename):
  #bkg_chain = TChain('tree')
  ##bkg_chain.Add("ntuples/train_decorr_bak.root")
  #bkg_chain.Add("train_bkg.root")
  #sig_chain = TChain('tree')
  ##sig_chain.Add("ntuples/train_decorr_sig.root")
  #sig_chain.Add("train_signal.root")
  
  output = TFile.Open(tmva_filename, 'RECREATE')
  factory = TMVA.Factory('TMVAClassification', output,
                         '!V:ROC:!Correlations:!Silent:Color:'
                         '!DrawProgressBar:AnalysisType=Classification')
  dataloader = TMVA.DataLoader('dataset')
  dataloader.AddVariable("photon_mva",'F')
  dataloader.AddVariable("min_dR",'F')
  dataloader.AddVariable("pt_mass",'F')
  dataloader.AddVariable("cosTheta",'F')
  dataloader.AddVariable("costheta",'F')
  #dataloader.AddVariable("photon_res",'F')
  dataloader.AddVariable("llg_mass_err",'F')
  dataloader.AddVariable("photon_rapidity",'F')
  dataloader.AddVariable("l1_rapidity",'F')
  dataloader.AddVariable("l2_rapidity",'F')
  dataloader.AddVariable("llg_flavor",'F')
  dataloader.SetBackgroundWeightExpression("w_lumi")
  dataloader.SetSignalWeightExpression("w_lumi")
  #dataloader.SetBackgroundWeightExpression("1")
  dataloader.AddSpectator("llg_mass", 'F')
  dataloader.AddSpectator("w_lumi", 'F')
  #dataloader.AddSignalTree(sig_chain)
  #dataloader.AddBackgroundTree(bkg_chain)
  #dataloader.AddSignalTree(sig_chain, 1)
  #dataloader.AddBackgroundTree(bkg_chain, 1)

  # Add data.
  train_chain = TChain('train_tree')
  train_chain.Add("train_sample.root")
  test_chain = TChain('test_tree')
  test_chain.Add("test_sample.root")
  dataloader.AddTree(train_chain, 'Background', 1., 'classID==0', TMVA.Types.kTraining)
  dataloader.AddTree(train_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTraining)
  dataloader.AddTree(test_chain, 'Background', 1., 'classID==0', TMVA.Types.kTesting)
  dataloader.AddTree(test_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTesting)
  cut_s = TCut('llg_mass>120 && llg_mass < 130');
  cut_b = TCut('llg_mass>120 && llg_mass < 130');
  dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=NumEvents:ScaleWithPreselEff:!V");

  #cut_s = TCut('llg_mass>120 && llg_mass < 130')
  #cut_b = TCut('llg_mass>120 && llg_mass < 130');
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"nTrain_Signal=0:nTrain_Background=0:nTest_Signal=1:nTest_Background=1:SplitMode=Random:NormMode=NumEvents:ScaleWithPreselEff:!V");
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=2500:nTrain_Background=2500:nTest_Signal=20000:nTest_Background=20000:SplitMode=Random:NormMode=NumEvents:!V");
  ## Use 5% for training 95% for evaluation
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=3603:nTrain_Background=9365:nTest_Signal=68450:nTest_Background=177932:SplitMode=Random:NormMode=NumEvents:!V");
  ## Use 20% for training 80% for evaluation
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=14410:nTrain_Background=37459:nTest_Signal=57643:nTest_Background=149838:SplitMode=Random:NormMode=NumEvents:!V");
  # Use 80% for training 20% for evaluation
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,"nTrain_Signal=57642:nTrain_Background=149837:nTest_Signal=14411:nTest_Background=37460:SplitMode=Random:NormMode=NumEvents:!V");

  layoutString = TString("Layout=TANH|40,LINEAR")
  #trainingString = TString("LearningRate=1e-6,ConvergenceSteps=100,BatchSize=1,TestRepetitions=10,MaxEpochs=3000")
  #trainingString = TString("LearningRate=1e-6,ConvergenceSteps=100,BatchSize=1,TestRepetitions=10,MaxEpochs=80000")
  trainingString = TString("LearningRate=1e-6,ConvergenceSteps=100,BatchSize=1,TestRepetitions=10,MaxEpochs=100")
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

def find_signal_fraction_thresholds(signal_fractions, tmva_chain, mva_name):
  # Find MVA threshold where signal is the below fractions
  mva_thresholds = []
  mva_range = [-1.,1.]
  niter = 5000
  signal_entries = tmva_chain.Draw("llg_mass>>hist","classID==1","goff")
  #print("All signal entries: "+str(signal_entries))
  iFraction = 0
  for iRange in range(niter):
    mva_threshold = (mva_range[1] - mva_range[0])/niter * iRange + mva_range[0]
    entries = tmva_chain.Draw("llg_mass>>hist","classID==1&&"+mva_name+">"+str(mva_threshold)+"","goff")
    fraction = entries *1. / signal_entries
    if (fraction < signal_fractions[iFraction]):
      mva_thresholds.append(mva_threshold)
      iFraction += 1
      #print('fraction: '+str(fraction)+" mva_threshold: "+str(mva_threshold))
      if (iFraction == len(signal_fractions)): break
  return mva_thresholds

def evaluate_mva_significance(tmva_filename, mva_name):
  tmva_chain = TChain("dataset/TestTree")
  tmva_chain.Add(tmva_filename)
  #luminosity = 137. + 110.
  luminosity = 1.
  hist = TH1F("hist","hist",80,100,180)
  # Find mva thresholds
  signal_fractions = [0.95,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
  mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mva_name)
  # mva_signal_width[threshold] = signal_width
  mva_signal_widths = {}
  # Find width of signals with MVA threshold
  for mva_threshold in mva_thresholds:
    entries = tmva_chain.Draw("llg_mass>>hist","classID==1&&"+mva_name+">"+str(mva_threshold)+"","goff")
    mva_signal_width = hist.GetStdDev()
    mva_signal_widths[mva_threshold] = mva_signal_width
    #print("mva threshold: "+str(mva_threshold)+" signal_width: "+str(mva_signal_width)+" entries: "+str(entries))
  # Find signal and background within 2 sigma of signal width
  for mva_threshold in mva_thresholds:
    sigma = 2.5
    mva_signal_width = mva_signal_widths[mva_threshold]
    tmva_chain.Draw("llg_mass>>hist","(classID==1&&"+mva_name+">"+str(mva_threshold)+"&&llg_mass<(125+"+str(mva_signal_width*sigma)+")&&llg_mass>(125-"+str(mva_signal_width*sigma)+"))*w_lumi*"+str(luminosity),"goff")
    nevents_signal = hist.GetSum()
    tmva_chain.Draw("llg_mass>>hist","(classID==0&&"+mva_name+">"+str(mva_threshold)+"&&llg_mass<(125+"+str(mva_signal_width*sigma)+")&&llg_mass>(125-"+str(mva_signal_width*sigma)+"))*w_lumi*"+str(luminosity),"goff")
    nevents_background = hist.GetSum()
    #print("mva_threshold: "+str(mva_threshold)+" nSig: "+str(nevents_signal)+" nBkg: "+str(nevents_background))
    # Calculate significance
    print("mva_threshold: "+str(mva_threshold)+" significance [s/sqrt(b)]: "+str(nevents_signal/math.sqrt(nevents_background)))
    print("mva_threshold: "+str(mva_threshold)+" purity (s/(b+s)): "+str(nevents_signal/(nevents_background+nevents_signal)))

def evaluate_mva_correlation(tmva_filename, mva_name):
  tmva_chain = TChain("dataset/TestTree")
  tmva_chain.Add(tmva_filename)
  # Find mva thresholds
  signal_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mva_name)
  # Make mass histograms with thresholds
  mva_hists = {}
  for mva_threshold in mva_thresholds:
    hist = TH1F("hist_"+str(mva_threshold),"hist_"+str(mva_threshold),80,100,180)
    entries = tmva_chain.Draw("llg_mass>>hist_"+str(mva_threshold),"classID==0&&"+mva_name+">"+str(mva_threshold)+"","goff")
    # Normalize histogram
    sum_weight = hist.GetSumOfWeights()
    hist.Scale(1/sum_weight)
    c1 = TCanvas("c1")
    hist.Draw()
    c1.SaveAs("plots/hist_mva_"+str(mva_threshold)+".pdf")
    mva_hists[mva_threshold] = hist
  # Get values of histogram bin. 1 is first bin, nbin is last bin
  #print(mva_hists[mva_thresholds[0]].GetNbinsX())
  std_values = []
  for iBin in range(mva_hists[mva_thresholds[0]].GetNbinsX()):
    bin_values = []
    for mva_threshold in mva_thresholds:
      hist_ibin_entry = mva_hists[mva_threshold].GetBinContent(iBin+1)
      bin_values.append(hist_ibin_entry)
      #print("iBin: "+str(iBin+1)+" mva_threshold: "+str(mva_threshold)+" "+str(hist_ibin_entry))
    stdev = np.std(bin_values, dtype=np.float64)
    mean = np.mean(bin_values, dtype=np.float64)
    if (stdev == 0 and mean == 0): continue # Ignore case where hist bins are empty
    print("iBin: "+str(iBin+1)+" x: "+str(mva_hists[mva_thresholds[0]].GetBinCenter(iBin+1))+" stdev: "+str(stdev))
    std_values.append(stdev)
  print("Mean stdev: "+str(np.mean(std_values)))

if __name__ == "__main__":
  
  #train_bdt("ntuples_mva/TMVA_bdt.root")
  train_nn("ntuples_mva/TMVA_nn.root")

  #evaluate_mva_significance("ntuples_mva/TMVA_bdt.root", "BDT")
  #evaluate_mva_correlation("ntuples_mva/TMVA_bdt.root", "BDT")
  #evaluate_mva_significance("ntuples_mva/TMVA_nn.root", "DNN")
  #evaluate_mva_correlation("ntuples_mva/TMVA_nn.root", "DNN")


  #output = TFile.Open('TMVA.root')
  #output.cd('dataset')
  #output.ls()
  #testTree = output.Get("TestTree")
  #testTree.Print()
  #TMVA::TMVAGui("ntuples_mva/TMVA_bdt.root")
