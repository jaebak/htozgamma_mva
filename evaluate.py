#!/usr/bin/env python3
import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)
import math
import numpy as np
from RootDataset import RootDataset
import sklearn.metrics
import uproot
import matplotlib.pyplot as plt
import array
#import slugify
import rootutils
import ctypes

def find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name = ''):
  ## Find MVA threshold where signal is the below fractions
  #mva_thresholds = []
  #mva_range = [-1.,1.]
  #niter = 5000
  #signal_entries = tmva_chain.Draw(mass_name+">>hist",label_name+"==1","goff")
  ##print("All signal entries: "+str(signal_entries))
  #iFraction = 0
  #for iRange in range(niter):
  #  mva_threshold = (mva_range[1] - mva_range[0])/niter * iRange + mva_range[0]
  #  entries = tmva_chain.Draw(mass_name+">>hist",label_name+"==1&&"+mva_name+">"+str(mva_threshold)+"","goff")
  #  fraction = entries *1. / signal_entries
  #  if (fraction < signal_fractions[iFraction]):
  #    mva_thresholds.append(mva_threshold)
  #    iFraction += 1
  #    #print('fraction: '+str(fraction)+" mva_threshold: "+str(mva_threshold))
  #    if (iFraction == len(signal_fractions)): break
  #print(mva_thresholds)
  hist_mva = ROOT.TH1F("hist_mva","hist_mva",10000,1,1)
  if weight_name == '': n_entries = tmva_chain.Draw(mva_name+">>hist_mva", label_name+"==1", 'goff')
  else: n_entries = tmva_chain.Draw(mva_name+">>hist_mva", f'({label_name}==1)*{weight_name}', 'goff')
  mva_quantiles = array.array('d', [0.]*len(signal_fractions))
  mva_fractions = array.array('d', [1.-signal_fraction for signal_fraction in signal_fractions])
  hist_mva.GetQuantiles(len(signal_fractions), mva_quantiles, mva_fractions)
  mva_thresholds = mva_quantiles.tolist()
  return mva_thresholds

def evaluate_mva_significance(tmva_filename, tree_name, mass_name, mva_name, label_name, weight_name):
  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(tmva_filename)
  #luminosity = 137. + 110.
  luminosity = 1.
  hist = ROOT.TH1F("hist","hist",80,100,180)
  # Find mva thresholds
  signal_fractions = [0.95,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
  mva_thresholds, signal_fractions = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name)
  # mva_signal_width[threshold] = signal_width
  mva_signal_widths = {}
  # Find width of signals with MVA threshold
  for mva_threshold in mva_thresholds:
    entries = tmva_chain.Draw(mass_name+">>hist",label_name+"==1&&"+mva_name+">"+str(mva_threshold)+"","goff")
    mva_signal_width = hist.GetStdDev()
    mva_signal_widths[mva_threshold] = mva_signal_width
    #print("mva threshold: "+str(mva_threshold)+" signal_width: "+str(mva_signal_width)+" entries: "+str(entries))
  # Find signal and background within 2 sigma of signal width
  for mva_threshold in mva_thresholds:
    sigma = 2.5
    mva_signal_width = mva_signal_widths[mva_threshold]
    tmva_chain.Draw(mass_name+">>hist","("+label_name+"==1&&"+mva_name+">"+str(mva_threshold)+"&&"+mass_name+"<(125+"+str(mva_signal_width*sigma)+")&&"+mass_name+">(125-"+str(mva_signal_width*sigma)+"))*"+weight_name+"*"+str(luminosity),"goff")
    nevents_signal = hist.GetSum()
    tmva_chain.Draw(mass_name+">>hist","("+label_name+"==0&&"+mva_name+">"+str(mva_threshold)+"&&"+mass_name+"<(125+"+str(mva_signal_width*sigma)+")&&"+mass_name+">(125-"+str(mva_signal_width*sigma)+"))*"+weight_name+"*"+str(luminosity),"goff")
    nevents_background = hist.GetSum()
    #print("mva_threshold: "+str(mva_threshold)+" nSig: "+str(nevents_signal)+" nBkg: "+str(nevents_background))
    # Calculate significance
    if nevents_background != 0:
      print("mva_threshold: "+str(mva_threshold)+" significance [s/sqrt(b)]: "+str(nevents_signal/math.sqrt(nevents_background)))
    if nevents_background+nevents_signal !=0:
      print("  mva_threshold: "+str(mva_threshold)+" purity (s/(b+s)): "+str(nevents_signal/(nevents_background+nevents_signal)))

def evaluate_mva_correlation(tmva_filename, tree_name, mass_name, mva_name, label_name):
  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(tmva_filename)
  # Find mva thresholds
  signal_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name)
  # Make mass histograms with thresholds
  mva_hists = {}
  for mva_threshold in mva_thresholds:
    hist = ROOT.TH1F("hist_"+str(mva_threshold),"hist_"+str(mva_threshold),80,100,180)
    entries = tmva_chain.Draw(mass_name+">>hist_"+str(mva_threshold),label_name+"==0&&"+mva_name+">"+str(mva_threshold)+"","goff")
    # Normalize histogram
    sum_weight = hist.GetSumOfWeights()
    hist.Scale(1/sum_weight)
    c1 = ROOT.TCanvas("c1")
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

def find_nearest(array_in,value):
  idx = (np.abs(array_in-value)).argmin()
  return idx, array_in[idx]

def evaluate_roc(info_mvas, no_weight=False):
  roc_infos = []
  # mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':}
  for mva_info in info_mvas:
    # Evaluate with ROC curve
    if no_weight:
      # roc with no weight
      fpr, tpr, threshold = sklearn.metrics.roc_curve(mva_info['test']['y'], mva_info['test']['yhat'])
    else:
      # roc with weights probably not working due to negative weights.
      sample_weight = np.array(mva_info['test']['weight'])
      sample_weight[sample_weight<0] = 0.
      fpr, tpr, threshold = sklearn.metrics.roc_curve(mva_info['test']['y'], mva_info['test']['yhat'], sample_weight=sample_weight)
    name = mva_info['name']
    roc_infos.append([fpr, tpr, threshold, name])
    #print(f'fpr: {fpr}\ntpr: {tpr}\nthresh: {threshold}')

  plt.figure()
  for roc_info in roc_infos:
    fpr, tpr, threshold, name = roc_info
    plt.plot(tpr, fpr, lw=2.5, label=name+", AUC = {:.1f}%".format(sklearn.metrics.auc(fpr, tpr)*100))
  plt.xlabel(r'True positive rate')
  plt.ylabel(r'False positive rate')
  #plt.semilogy()
  plt.ylim(0.001,1)
  plt.xlim(0,1)
  plt.grid(True)
  plt.legend(loc='upper left')
  if no_weight: filename = "plots/roc_noweight_higgsToZGamma_classifiers.pdf"
  else: filename = "plots/roc_higgsToZGamma_classifiers.pdf"
  plt.savefig(filename)
  print(f"Saved to {filename}")

def load_tmva_dict(filename, mva_var_name, name):
  tmva_file = uproot.open(filename)
  tmva_train = {}
  tmva_train['x'] = tmva_file['dataset/TrainTree'].arrays(['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'photon_res', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'])
  tmva_train['y'] = tmva_file['dataset/TrainTree']['classID'].array()
  tmva_train['yhat'] = tmva_file['dataset/TrainTree'][mva_var_name].array()
  tmva_train['observable'] = tmva_file['dataset/TrainTree']['llg_mass'].array()
  tmva_train['weight'] = tmva_file['dataset/TrainTree']['w_lumi'].array()
  tmva_test = {}
  tmva_test['x'] = tmva_file['dataset/TestTree'].arrays(['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'photon_res', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'])
  tmva_test['y'] = tmva_file['dataset/TestTree']['classID'].array()
  tmva_test['yhat'] = tmva_file['dataset/TestTree'][mva_var_name].array()
  tmva_test['observable'] = tmva_file['dataset/TestTree']['llg_mass'].array()
  tmva_test['weight'] = tmva_file['dataset/TestTree']['w_lumi'].array()
  tmva = {'train': tmva_train, 'test': tmva_test, 'name': name, 
    'names': {'train_filename': filename, 'test_filename': filename,
      'train_tree': 'dataset/TrainTree', 'test_tree': 'dataset/TestTree', 'y': 'classID', 
    'x': ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'photon_res', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'], 
    'yhat': mva_var_name, 'observable': 'llg_mass', 'weight': 'w_lumi'}}
  tmva_file.close()
  return tmva

def load_tmva_eval_dict(train_test_filename, test_full_filename, mva_var_name, name, dataset_name='dataset', mva_input=['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'photon_res', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity']):
  tmva_train = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_train['x'] = tmva_file[f'{dataset_name}/TrainTree'].arrays(mva_input)
    tmva_train['y'] = tmva_file[f'{dataset_name}/TrainTree']['classID'].array()
    tmva_train['yhat'] = tmva_file[f'{dataset_name}/TrainTree'][mva_var_name].array()
    tmva_train['observable'] = tmva_file[f'{dataset_name}/TrainTree']['llg_mass'].array()
    tmva_train['weight'] = tmva_file[f'{dataset_name}/TrainTree']['w_lumiXyear'].array()
  tmva_test = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_test['x'] = tmva_file[f'{dataset_name}/TestTree'].arrays(mva_input)
    tmva_test['y'] = tmva_file[f'{dataset_name}/TestTree']['classID'].array()
    tmva_test['yhat'] = tmva_file[f'{dataset_name}/TestTree'][mva_var_name].array()
    tmva_test['observable'] = tmva_file[f'{dataset_name}/TestTree']['llg_mass'].array()
    tmva_test['weight'] = tmva_file[f'{dataset_name}/TestTree']['w_lumiXyear'].array()
  tmva_test_full = {}
  with uproot.open(test_full_filename) as tmva_file:
    tmva_test_full['x'] = tmva_file['test_tree'].arrays(mva_input)
    tmva_test_full['y'] = tmva_file['test_tree']['classID'].array()
    tmva_test_full['yhat'] = tmva_file['test_tree'][mva_var_name].array()
    tmva_test_full['observable'] = tmva_file['test_tree']['llg_mass'].array()
    tmva_test_full['weight'] = tmva_file['test_tree']['w_lumiXyear'].array()
  tmva = {'train': tmva_train, 'test': tmva_test, 'test_full': tmva_test_full, 'name': name,
    'names': {'train_filename': train_test_filename, 'test_filename': train_test_filename, 'test_full_filename': test_full_filename,
    'train_tree': f'{dataset_name}/TrainTree', 'test_tree': f'{dataset_name}/TestTree', 'test_full_tree': 'test_tree',
    'y': 'classID', 
    'x': ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'photon_res', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'], 
    'yhat': mva_var_name, 'observable': 'llg_mass', 'weight': 'w_lumiXyear'}}
  return tmva

def load_kfold_tmva_eval_dict(train_test_filename, test_full_filename, mva_var_name, name, dataset_name='dataset', mva_input=['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'photon_res', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity']):
  tmva_train = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_train['x'] = tmva_file[f'{dataset_name}/TrainTree'].arrays(mva_input)
    tmva_train['y'] = tmva_file[f'{dataset_name}/TrainTree']['classID'].array()
    tmva_train['yhat'] = tmva_file[f'{dataset_name}/TrainTree'][mva_var_name].array()
    tmva_train['observable'] = tmva_file[f'{dataset_name}/TrainTree']['llg_mass'].array()
    tmva_train['weight'] = tmva_file[f'{dataset_name}/TrainTree']['w_lumiXyear'].array()
  tmva_test = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_test['x'] = tmva_file[f'{dataset_name}/TestTree'].arrays(mva_input)
    tmva_test['y'] = tmva_file[f'{dataset_name}/TestTree']['classID'].array()
    tmva_test['yhat'] = tmva_file[f'{dataset_name}/TestTree'][mva_var_name].array()
    tmva_test['observable'] = tmva_file[f'{dataset_name}/TestTree']['llg_mass'].array()
    tmva_test['weight'] = tmva_file[f'{dataset_name}/TestTree']['w_lumiXyear'].array()
  tmva_test_full = {}
  with uproot.open(test_full_filename) as tmva_file:
    tmva_test_full['x'] = tmva_file['test_full_tree'].arrays(mva_input)
    tmva_test_full['y'] = tmva_file['test_full_tree']['classID'].array()
    tmva_test_full['yhat'] = tmva_file['test_full_tree'][mva_var_name].array()
    tmva_test_full['observable'] = tmva_file['test_full_tree']['llg_mass'].array()
    tmva_test_full['weight'] = tmva_file['test_full_tree']['w_lumiXyear'].array()
  tmva = {'train': tmva_train, 'test': tmva_test, 'test_full': tmva_test_full, 'name': name,
    'names': {'train_filename': train_test_filename, 'test_filename': train_test_filename, 'test_full_filename': test_full_filename,
    'train_tree': f'{dataset_name}/TrainTree', 'test_tree': f'{dataset_name}/TestTree', 'test_full_tree': 'test_full_tree',
    'y': 'classID', 
    'x': ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'photon_res', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'], 
    'yhat': mva_var_name, 'observable': 'llg_mass', 'weight': 'w_lumiXyear'}}
  return tmva

def load_tmva_simple_eval_dict(train_test_filename, test_full_filename, mva_var_name, name):
  tmva_train = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_train['x'] = tmva_file['dataset_simple/TrainTree'].arrays(['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
    tmva_train['y'] = tmva_file['dataset_simple/TrainTree']['classID'].array()
    tmva_train['yhat'] = tmva_file['dataset_simple/TrainTree'][mva_var_name].array()
    tmva_train['observable'] = tmva_file['dataset_simple/TrainTree']['llg_mass'].array()
    tmva_train['weight'] = tmva_file['dataset_simple/TrainTree']['w_lumi'].array()
  tmva_test = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_test['x'] = tmva_file['dataset_simple/TestTree'].arrays(['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
    tmva_test['y'] = tmva_file['dataset_simple/TestTree']['classID'].array()
    tmva_test['yhat'] = tmva_file['dataset_simple/TestTree'][mva_var_name].array()
    tmva_test['observable'] = tmva_file['dataset_simple/TestTree']['llg_mass'].array()
    tmva_test['weight'] = tmva_file['dataset_simple/TestTree']['w_lumi'].array()
  tmva_test_full = {}
  with uproot.open(test_full_filename) as tmva_file:
    tmva_test_full['x'] = tmva_file['test_tree'].arrays(['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
    tmva_test_full['y'] = tmva_file['test_tree']['classID'].array()
    tmva_test_full['yhat'] = tmva_file['test_tree'][mva_var_name].array()
    tmva_test_full['observable'] = tmva_file['test_tree']['llg_mass'].array()
    tmva_test_full['weight'] = tmva_file['test_tree']['w_lumi'].array()
  tmva = {'train': tmva_train, 'test': tmva_test, 'test_full': tmva_test_full, 'name': name,
    'names': {'train_filename': train_test_filename, 'test_filename': train_test_filename, 'test_full_filename': test_full_filename,
    'train_tree': 'dataset_simple/TrainTree', 'test_tree': 'dataset_simple/TestTree', 'test_full_tree': 'test_tree',
    'y': 'classID', 
    'x': ['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'], 
    'yhat': mva_var_name, 'observable': 'llg_mass', 'weight': 'w_lumi'}}
  return tmva

def load_tmva_res_dict(filename, mva_var_name, name):
  tmva_file = uproot.open(filename)
  tmva_train = {}
  tmva_train['x'] = tmva_file['dataset/TrainTree'].arrays(['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'])
  tmva_train['y'] = tmva_file['dataset/TrainTree']['classID'].array()
  tmva_train['yhat'] = tmva_file['dataset/TrainTree'][mva_var_name].array()
  tmva_train['observable'] = tmva_file['dataset/TrainTree']['llg_mass'].array()
  tmva_train['weight'] = tmva_file['dataset/TrainTree']['w_lumi'].array()
  tmva_test = {}
  tmva_test['x'] = tmva_file['dataset/TestTree'].arrays(['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'])
  tmva_test['y'] = tmva_file['dataset/TestTree']['classID'].array()
  tmva_test['yhat'] = tmva_file['dataset/TestTree'][mva_var_name].array()
  tmva_test['observable'] = tmva_file['dataset/TestTree']['llg_mass'].array()
  tmva_test['weight'] = tmva_file['dataset/TestTree']['w_lumi'].array()
  tmva = {'train': tmva_train, 'test': tmva_test, 'name': name, 'filename': filename, 
    'names': {'train_tree': 'dataset/TrainTree', 'test_tree': 'dataset/TestTree', 'y': 'classID', 
    'x': ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'], 
    'yhat': mva_var_name, 'observable': 'llg_mass', 'weight': 'w_lumi'}}
  tmva_file.close()
  return tmva

def load_tmva_eval_res_dict(train_test_filename, test_full_filename, mva_var_name, name):
  tmva_train = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_train['x'] = tmva_file['dataset/TrainTree'].arrays(['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'])
    tmva_train['y'] = tmva_file['dataset/TrainTree']['classID'].array()
    tmva_train['yhat'] = tmva_file['dataset/TrainTree'][mva_var_name].array()
    tmva_train['observable'] = tmva_file['dataset/TrainTree']['llg_mass'].array()
    tmva_train['weight'] = tmva_file['dataset/TrainTree']['w_lumi'].array()
  tmva_test = {}
  with uproot.open(train_test_filename) as tmva_file:
    tmva_test['x'] = tmva_file['dataset/TestTree'].arrays(['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'])
    tmva_test['y'] = tmva_file['dataset/TestTree']['classID'].array()
    tmva_test['yhat'] = tmva_file['dataset/TestTree'][mva_var_name].array()
    tmva_test['observable'] = tmva_file['dataset/TestTree']['llg_mass'].array()
    tmva_test['weight'] = tmva_file['dataset/TestTree']['w_lumi'].array()
  with uproot.open(test_full_filename) as tmva_file:
    tmva_test['x'] = tmva_file['test_tree'].arrays(['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'])
    tmva_test['y'] = tmva_file['test_tree']['classID'].array()
    tmva_test['yhat'] = tmva_file['test_tree'][mva_var_name].array()
    tmva_test['observable'] = tmva_file['test_tree']['llg_mass'].array()
    tmva_test['weight'] = tmva_file['test_tree']['w_lumi'].array()
  tmva = {'train': tmva_train, 'test': tmva_test, 'name': name,  
    'names': {'train_filename': train_test_filename, 'test_filename': train_test_filename, 'test_full_filename': test_full_filename,
    'train_tree': 'dataset/TrainTree', 'test_tree': 'dataset/TestTree', 'test_full_tree': 'test_tree',
    'y': 'classID', 
    'x': ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity'], 
    'yhat': mva_var_name, 'observable': 'llg_mass', 'weight': 'w_lumi'}}
  return tmva

def load_mva_dict(filename, name):
  mva_file = uproot.open(filename)
  mva_train = {}
  mva_train['x'] = mva_file['train_tree']['x'].array()
  mva_train['y'] = mva_file['train_tree']['y'].array()
  mva_train['yhat'] = mva_file['train_tree']['yhat'].array()
  mva_train['observable'] = mva_file['train_tree']['mass'].array()
  mva_train['weight'] = mva_file['train_tree']['weight'].array()
  mva_test = {}
  mva_test['x'] = mva_file['test_tree']['x'].array()
  mva_test['y'] = mva_file['test_tree']['y'].array()
  mva_test['yhat'] = mva_file['test_tree']['yhat'].array()
  mva_test['observable'] = mva_file['test_tree']['mass'].array()
  mva_test['weight'] = mva_file['test_tree']['weight'].array()
  mva = {'train': mva_train, 'test': mva_test, 'name': name, 
    'names':{'train_filename': filename, 'test_filename': filename, 'test_full_filename': filename,
    'train_tree': 'train_tree', 'test_tree': 'test_tree', 'test_full_tree': 'test_full_tree',
    'y': 'y', 
    'x': ['x[0]', 'x[1]', 'x[2]', 'x[3]','x[4]', 'x[5]','x[6]', 'x[7]', 'x[8]'], 
    'yhat': 'yhat', 'observable': 'mass', 'weight': 'weight'}}
  mva_file.close()
  return mva

#def load_mva_dict(filename, name):
#  mva_file = uproot.open(filename)
#  mva_train = {}
#  mva_train['x'] = mva_file['train_tree']['x'].array()
#  mva_train['y'] = mva_file['train_tree']['y'].array()
#  mva_train['yhat'] = mva_file['train_tree']['yhat'].array()
#  mva_train['observable'] = mva_file['train_tree']['mass'].array()
#  mva_train['weight'] = mva_file['train_tree']['weight'].array()
#  mva_test = {}
#  mva_test['x'] = mva_file['test_tree']['x'].array()
#  mva_test['y'] = mva_file['test_tree']['y'].array()
#  mva_test['yhat'] = mva_file['test_tree']['yhat'].array()
#  mva_test['observable'] = mva_file['test_tree']['mass'].array()
#  mva_test['weight'] = mva_file['test_tree']['weight'].array()
#  mva = {'train': mva_train, 'test': mva_test, 'name': name, 'filename': filename, 
#    'names':{'train_tree': 'train_tree', 'test_tree': 'test_tree', 'y': 'y', 
#    'x': ['x[0]', 'x[1]', 'x[2]', 'x[3]','x[4]', 'x[5]','x[6]', 'x[7]', 'x[8]'], 
#    'yhat': 'yhat', 'observable': 'mass', 'weight': 'weight'}}
#  mva_file.close()
#  return mva

def normalize_hist(hist):
  sum_weight = hist.GetSumOfWeights()
  hist.Scale(1/sum_weight)

# Make plot of mva variable for signal/backgroundxtrain/test
def evaluate_overtraining(info_mvas):
  # mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'label':, 'yhat':, 'observable': 'weight':, 'x':}}
  for mva_info in info_mvas:
    # Find min max of mva variable
    mva_max = np.amax([mva_info['train']['yhat'], mva_info['train']['yhat']])
    mva_min = np.amin([mva_info['train']['yhat'], mva_info['train']['yhat']])
    # Test tree
    test_root_file = ROOT.TFile(mva_info['names']['test_filename'])
    test_tree = test_root_file.Get(mva_info['names']['test_tree'])
    yhat_signal_test = ROOT.TH1F('signal_test', 'signal_test;mva', 25, mva_min, mva_max)
    yhat_bkg_test = ROOT.TH1F('bkg_test', 'bkg_test', 25, mva_min, mva_max)
    test_tree.Draw(mva_info['names']['yhat']+'>>signal_test', mva_info['names']['y']+'==1', 'goff')
    test_tree.Draw(mva_info['names']['yhat']+'>>bkg_test', mva_info['names']['y']+'==0', 'goff')
    # Train tree
    train_root_file = ROOT.TFile(mva_info['names']['train_filename'])
    train_tree = train_root_file.Get(mva_info['names']['train_tree'])
    yhat_signal_train = ROOT.TH1F('signal_train', 'signal_train', 25, mva_min, mva_max)
    yhat_bkg_train = ROOT.TH1F('bkg_train', 'bkg_train', 25, mva_min, mva_max)
    train_tree.Draw(mva_info['names']['yhat']+'>>signal_train', mva_info['names']['y']+'==1', 'goff')
    train_tree.Draw(mva_info['names']['yhat']+'>>bkg_train', mva_info['names']['y']+'==0', 'goff')
    # Normalize
    normalize_hist(yhat_signal_test)
    normalize_hist(yhat_bkg_test)
    normalize_hist(yhat_signal_train)
    normalize_hist(yhat_bkg_train)
    y_max = np.amax([yhat_signal_test.GetMaximum(), yhat_signal_train.GetMaximum(), yhat_bkg_test.GetMaximum(), yhat_bkg_train.GetMaximum()])
    blank_hist = ROOT.TH1F('blank_hist','Overtraining '+mva_info['name'], 25, mva_min, mva_max)
    blank_hist.SetMaximum(y_max * 1.1)
    blank_hist.SetMinimum(0.)

    c1 = rootutils.new_canvas()
    up_pad = ROOT.TPad('up_pad', 'up_pad', 0, 0.3, 1, 1)
    low_pad = ROOT.TPad('low_pad', 'low_pad', 0, 0, 1, 0.3)
    up_pad.Draw()
    low_pad.Draw()

    up_pad.cd()
    ROOT.gStyle.SetOptStat(0)
    blank_hist.Draw()
    yhat_signal_test.SetLineColor(ROOT.kBlue)
    yhat_signal_test.Draw('hist same E1')
    yhat_bkg_test.SetLineColor(ROOT.kGreen)
    yhat_bkg_test.Draw('same hist E1')
    yhat_signal_train.SetLineColor(ROOT.kRed)
    yhat_signal_train.Draw('same')
    yhat_bkg_train.SetLineColor(ROOT.kBlack)
    yhat_bkg_train.Draw('same')
    legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    legend.AddEntry(yhat_signal_test)
    legend.AddEntry(yhat_signal_train)
    legend.AddEntry(yhat_bkg_test)
    legend.AddEntry(yhat_bkg_train)
    legend.Draw()

    low_pad.cd()
    n_bins = blank_hist.GetNbinsX()
    #x_min = blank_hist.GetXaxis().GetBinLowEdge(blank_hist.GetXaxis().GetFirst())
    #x_max = blank_hist.GetXaxis().GetBinUpEdge(blank_hist.GetXaxis().GetLast())
    residual_y_signal = array.array('d',[0.]*n_bins)
    residual_y_bkg = array.array('d',[0.]*n_bins)
    residual_x = array.array('d',[0.]*n_bins)
    for index in range(n_bins):
      residual_x[index] = mva_min + (mva_max-mva_min)/n_bins * index + (mva_max-mva_min)/n_bins/2
    ROOT.gErrorIgnoreLevel = ROOT.kError
    pvalue_signal = yhat_signal_train.Chi2Test(yhat_signal_test, 'WW', residual_y_signal)
    pvalue_bkg = yhat_bkg_train.Chi2Test(yhat_bkg_test, 'WW', residual_y_bkg)
    #print(mva_info['name']+f' pvalue signal: {pvalue_signal} bkg: {pvalue_bkg}')
    ROOT.gErrorIgnoreLevel = ROOT.kPrint
    #print('pvalue_signal: '+str(pvalue_signal))
    #print('pvalue_bkg: '+str(pvalue_bkg))
    residual_signal = ROOT.TGraph(n_bins, residual_x, residual_y_signal)
    # Set range on computed graph. Set twice because TGraph's axis looks strange otherwise
    residual_signal.GetXaxis().SetLimits(mva_min, mva_max)
    residual_signal.SetTitle('')
    residual_signal.GetYaxis().SetRangeUser(-3.5, 3.5)
    residual_signal.GetYaxis().SetTitle("Normalized residuals")
    residual_signal.GetYaxis().CenterTitle()
    residual_signal.SetMarkerStyle(21)
    residual_signal.SetMarkerSize(0.3)
    residual_signal.SetMarkerColor(ROOT.kRed)
    residual_signal.Draw("AP")
    residual_bkg = ROOT.TGraph(n_bins, residual_x, residual_y_bkg)
    # Set range on computed graph. Set twice because TGraph's axis looks strange otherwise
    residual_bkg.GetXaxis().SetLimits(mva_min, mva_max)
    residual_bkg.SetTitle('')
    residual_bkg.GetYaxis().SetRangeUser(-3.5, 3.5)
    residual_bkg.GetYaxis().SetTitle("Normalized residuals")
    residual_bkg.GetYaxis().CenterTitle()
    residual_bkg.SetMarkerStyle(21)
    residual_bkg.SetMarkerSize(0.3)
    residual_bkg.SetMarkerColor(ROOT.kBlack)
    residual_bkg.Draw("P same")
    zero_line = ROOT.TLine(mva_min, 0, mva_max, 0)
    zero_line.Draw()
    box = ROOT.TPaveText(0.12, 0.8, 0.5, 0.9, 'NDC NB')
    box.SetFillColorAlpha(0,0)
    box.AddText('#chi^{2} test p-value signal: '+f'{pvalue_signal:.3f} bkg: {pvalue_bkg:.3f}')
    box.Draw()
    c1.SaveAs('plots/overtrain_'+rootutils.slugify(mva_info['name'])+'.pdf')
    ROOT.gStyle.SetOptStat(1)

    test_root_file.Close()
    train_root_file.Close()

# Try evaluate significance in fixed window

# mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
def evaluate_significance_bins_with_resolution(info_mvas, draw=True, tree_type='test_tree', fixed_width = False):
  verbose = False
  signal_fractions = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05] #Last is due to binning code below
  signal_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1] #Last is due to binning code below
  #signal_fractions = [0.6, 0.3, 0.3] #Last is due to binning code below
  # Draw how significance changes for each mva
  x_array = array.array('d', signal_fractions)
  # significance_arrays[mva_name] = [signficance, ...]
  significance_arrays = {}
  significance_squared_arrays = {}
  purity_arrays = {}
  resolution_arrays = {}
  resolution_windows = {}
  significance_graphs = {}
  purity_graphs = {}
  resolution_graphs = {}
  best_significances = []
  integrated_significances= []
  purity_sigeff70s = []
  significance_min_max = []
  purity_min_max = []
  resolution_min_max = []
  for mva_info in info_mvas:
    tree_name = mva_info['names'][tree_type]
    if tree_type == 'train_tree': tmva_filename = mva_info['names']['train_filename']
    elif tree_type == 'test_full_tree': tmva_filename = mva_info['names']['test_full_filename']
    else: tmva_filename = mva_info['names']['test_filename']
    mass_name = mva_info['names']['observable']
    label_name = mva_info['names']['y']
    weight_name = mva_info['names']['weight']
    mva_name = mva_info['names']['yhat']
    mva_tag = mva_info['name']
    significance_arrays[mva_tag] = array.array('d')
    significance_squared_arrays[mva_tag] = array.array('d')
    purity_arrays[mva_tag] = array.array('d')
    resolution_arrays[mva_tag] = array.array('d')

    tmva_chain = ROOT.TChain(tree_name)
    tmva_chain.Add(tmva_filename)
    #luminosity = 137. + 110.
    luminosity = 1.
    hist = ROOT.TH1F("hist","hist",160,100,180)
    # Find mva thresholds
    mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name)
    sigeff70_mva_threshold = mva_thresholds[signal_fractions.index(0.7)]
    if verbose: print(f'signal_fractions: {signal_fractions} mva_thresholds: {mva_thresholds}')
    # mva_signal_width[threshold] = signal_width
    mva_signal_widths = {}
    # Find width of signals for MVA threshold bin
    for ithresh, mva_threshold in enumerate(mva_thresholds):
      if ithresh == 0:
        mva_window = f'{mva_name}<{mva_threshold}'
      elif ithresh == len(mva_thresholds)-1:
        mva_window = f'{mva_name}>{mva_threshold}'
      else:
        mva_window = f'{mva_name}<{mva_threshold}&&{mva_name}>{mva_thresholds[ithresh-1]}'
      entries = tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_window})*{weight_name}',"goff")
      mva_signal_width = hist.GetStdDev()
      mva_signal_widths[mva_threshold] = mva_signal_width
      #resolution_arrays[mva_tag].append(mva_signal_width)
      #mass_fractions = [0.975, 0.84, 0.16, 0.025]
      mass_fractions = [0.95, 0.84, 0.16, 0.05] # 90%, 65%
      mass_quantiles = array.array('d', [0.]*len(mass_fractions))
      mass_fractions = array.array('d', [1.-mass_fraction for mass_fraction in mass_fractions])
      hist.GetQuantiles(len(mass_fractions), mass_quantiles, mass_fractions)
      mass_thresholds = mass_quantiles.tolist()
      if verbose: print(f'mass thresh: {mass_thresholds}')
      if verbose: print(f'90 window: {mass_thresholds[3]-mass_thresholds[0]} 68 window: {mass_thresholds[2]-mass_thresholds[1]}')
      resolution_arrays[mva_tag].append(mass_thresholds[2]-mass_thresholds[1]) # 65% of signal
      if mva_tag not in resolution_windows: resolution_windows[mva_tag] = {}
      resolution_windows[mva_tag][mva_threshold] = mass_thresholds
      if verbose: print("mva window: "+str(mva_window)+" signal_width: "+str(resolution_arrays[mva_tag][-1])+" entries: "+str(entries))
    # Find signal and background within 2 sigma of signal width
    significances = {}
    purities = {}
    sigma = 2
    # Find total signal and background events
    mva_signal_width = 2.5
    mass_window = f'{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})'
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mass_window})*{weight_name}',"goff")
    nevents_signal = hist.GetSum()
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mass_window})*{weight_name}',"goff")
    nevents_background = hist.GetSum()
    if verbose: print(f'All nSig: {nevents_signal} nBkg: {nevents_background} signi: {nevents_signal/math.sqrt(nevents_background)}')
    for ithresh, mva_threshold in enumerate(mva_thresholds):
      mva_signal_width = mva_signal_widths[mva_threshold]
      if fixed_width: mva_signal_width = 2.
      #mass_window = f'{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})'
      mass_window = f'{mass_name}<{resolution_windows[mva_tag][mva_threshold][3]}&&{mass_name}>{resolution_windows[mva_tag][mva_threshold][0]}' #90% of signal
      #mass_window = f'{mass_name}<{resolution_windows[mva_tag][mva_threshold][2]}&&{mass_name}>{resolution_windows[mva_tag][mva_threshold][1]}' #65% of signal
      if ithresh == 0:
        mva_window = f'{mva_name}<{mva_threshold}'
      elif ithresh == len(mva_thresholds)-1:
        mva_window = f'{mva_name}>{mva_threshold}'
      else:
        mva_window = f'{mva_name}<{mva_threshold}&&{mva_name}>{mva_thresholds[ithresh-1]}'
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_window}&&{mass_window})*{weight_name}*{luminosity}',"goff")
      nevents_signal = hist.GetSum()
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mva_window}&&{mass_window})*{weight_name}*{luminosity}',"goff")
      nevents_background = hist.GetSum()
      #if verbose: print(f'signal width: {mva_signal_width} sigma: {sigma} mva_threshold: {mva_window} nSig: {nevents_signal} nBkg: {nevents_background}')
      if verbose: print(f'mva_threshold: {mva_window} nSig: {nevents_signal} nBkg: {nevents_background}')
      if verbose: print(f'  mass window: {mass_window}')
      # Calculate significance
      if nevents_background != 0:
        if verbose: print(f"mva_threshold: {mva_threshold:.4f} significance [s/sqrt(b)]: {nevents_signal/math.sqrt(nevents_background):.4f}")
        significances[mva_threshold] = nevents_signal/math.sqrt(nevents_background)
        significance_arrays[mva_tag].append(nevents_signal/math.sqrt(nevents_background))
        significance_squared_arrays[mva_tag].append(nevents_signal**2/nevents_background)
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have background events. Setting significance to 0.')
        significances[mva_threshold] = 0
        significance_arrays[mva_tag].append(0)
        significance_squared_arrays[mva_tag].append(0)
      # Calculate purity
      if nevents_background+nevents_signal !=0:
        if verbose: print(f"  mva_threshold: {mva_threshold:.4f} purity (s/(b+s)): {nevents_signal/(nevents_background+nevents_signal):.4f}")
        purity_arrays[mva_tag].append(nevents_signal/(nevents_background+nevents_signal))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have signal+background events. Setting purity to 0.')
        purity_arrays[mva_tag].append(0)
      if mva_threshold == sigeff70_mva_threshold:
        purity_sigeff70 = purity_arrays[mva_tag][-1]
    # Find best significance
    best_mva = -999
    best_significance = -999
    for mva_threshold in mva_thresholds:
      if significances[mva_threshold] > best_significance:
        best_significance = significances[mva_threshold]
        best_mva = mva_threshold
    #print(f'{mva_tag} Best mva threshold: {best_mva:.4f} significance: {best_significance:.4f}')
    # Find integrated significance
    integrated_significance = 0.
    for significance in significance_arrays[mva_tag]:
      integrated_significance += significance**2
    integrated_significance = math.sqrt(integrated_significance)
    integrated_significances.append(integrated_significance)
    print(f'{mva_tag} Integrated significance: {integrated_significance:.4f}')
    best_significances.append(best_significance)
    purity_sigeff70s.append(purity_sigeff70)
    # Make graph
    #significance_graph = ROOT.TGraph(len(signal_fractions), x_array, significance_squared_arrays[mva_tag])
    significance_graph = ROOT.TGraph(len(signal_fractions), x_array, significance_arrays[mva_tag])
    significance_graphs[mva_tag] = significance_graph
    purity_graph = ROOT.TGraph(len(signal_fractions), x_array, purity_arrays[mva_tag])
    purity_graphs[mva_tag] = purity_graph
    resolution_graph = ROOT.TGraph(len(signal_fractions), x_array, resolution_arrays[mva_tag])
    resolution_graphs[mva_tag] = resolution_graph
    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    significance_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(significance_min_max) == 0: significance_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
      if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {significance_min_max[0]}, min_max[1]: {significance_min_max[1]}')
    purity_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(purity_min_max) == 0: purity_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < purity_min_max[0]: purity_min_max[0] = y_min.value
      if y_max.value > purity_min_max[1]: purity_min_max[1] = y_max.value
    resolution_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(resolution_min_max) == 0: resolution_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < resolution_min_max[0]: resolution_min_max[0] = y_min.value
      if y_max.value > resolution_min_max[1]: resolution_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {purity_min_max[0]}, min_max[1]: {purity_min_max[1]}')
  # Draw graphs
  if draw:
    c1 = rootutils.new_canvas()
    c1.SetLeftMargin(0.15)
    significance_legend = ROOT.TLegend(0.5, 0.6, 0.9, 0.98)
    colors = [1,2,3,4,6,7,8,9,46,49,41,38]
    for iMva, mva_tag in enumerate(significance_graphs):
      #significance_graphs[mva_tag].SetTitle(mva_tag)
      integrated_significance = integrated_significances[iMva]
      significance_graphs[mva_tag].SetTitle(f';signal eff.;significance')
      if iMva == 0: significance_graphs[mva_tag].Draw("APL same")
      else: significance_graphs[mva_tag].Draw("PL")
      significance_graphs[mva_tag].SetLineColor(colors[iMva])
      significance_graphs[mva_tag].SetMarkerColor(colors[iMva])
      significance_graphs[mva_tag].SetMarkerStyle(21)
      significance_graphs[mva_tag].SetMarkerSize(0.3)
      significance_graphs[mva_tag].GetYaxis().SetRangeUser(significance_min_max[0]-0.01, significance_min_max[1]+0.01)
      significance_legend.AddEntry(significance_graphs[mva_tag], f'{mva_tag}, comb. signi= {integrated_significance:.3f}')
      significance_legend.Draw()
    c1.SaveAs('plots/significances_with_resolution.pdf')

    c2 = rootutils.new_canvas()
    c2.SetLeftMargin(0.15)
    purity_legend = ROOT.TLegend(0.5, 0.6, 0.9, 0.98)
    colors = [1,2,3,4,6,7,8,9,46,49,41,38]
    for iMva, mva_tag in enumerate(purity_graphs):
      purity_graphs[mva_tag].SetTitle(f';signal eff.;purity')
      if iMva == 0: purity_graphs[mva_tag].Draw("APL")
      else: purity_graphs[mva_tag].Draw("PL")
      purity_graphs[mva_tag].SetLineColor(colors[iMva])
      purity_graphs[mva_tag].SetMarkerColor(colors[iMva])
      purity_graphs[mva_tag].SetMarkerStyle(21)
      purity_graphs[mva_tag].SetMarkerSize(0.3)
      purity_graphs[mva_tag].GetYaxis().SetRangeUser(0, purity_min_max[1]+0.01)
      purity_legend.AddEntry(purity_graphs[mva_tag], f'{mva_tag}')
      purity_legend.Draw()
    c2.SaveAs('plots/purity_with_resolution.pdf')

    c3 = rootutils.new_canvas()
    c3.SetLeftMargin(0.15)
    resolution_legend = ROOT.TLegend(0.2, 0.6, 0.5, 0.98)
    colors = [1,2,3,4,6,7,8,9,46,49,41,38]
    for iMva, mva_tag in enumerate(resolution_graphs):
      resolution_graphs[mva_tag].SetTitle(f';signal eff.;signal resolution')
      if iMva == 0: resolution_graphs[mva_tag].Draw("APL")
      else: resolution_graphs[mva_tag].Draw("PL")
      resolution_graphs[mva_tag].SetLineColor(colors[iMva])
      resolution_graphs[mva_tag].SetMarkerColor(colors[iMva])
      resolution_graphs[mva_tag].SetMarkerStyle(21)
      resolution_graphs[mva_tag].SetMarkerSize(0.3)
      resolution_graphs[mva_tag].GetYaxis().SetRangeUser(resolution_min_max[0]-0.1, resolution_min_max[1]+0.1)
      resolution_legend.AddEntry(resolution_graphs[mva_tag], f'{mva_tag}')
      resolution_legend.Draw()
    c3.SaveAs('plots/signal_resolution.pdf')

  return integrated_significances, purity_sigeff70s

# mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
def evaluate_significance_with_resolution(info_mvas, draw=True, tree_type='test_tree'):
  verbose = False
  signal_fractions = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
  # Draw how significance changes for each mva
  x_array = array.array('d', signal_fractions)
  # significance_arrays[mva_name] = [signficance, ...]
  significance_arrays = {}
  purity_arrays = {}
  resolution_arrays = {}
  significance_graphs = {}
  purity_graphs = {}
  resolution_graphs = {}
  best_significances = []
  purity_sigeff70s = []
  significance_min_max = []
  purity_min_max = []
  resolution_min_max = []
  for mva_info in info_mvas:
    tree_name = mva_info['names'][tree_type]
    if tree_type == 'train_tree': tmva_filename = mva_info['names']['train_filename']
    elif tree_type == 'test_full_tree': tmva_filename = mva_info['names']['test_full_filename']
    else: tmva_filename = mva_info['names']['test_filename']
    mass_name = mva_info['names']['observable']
    label_name = mva_info['names']['y']
    weight_name = mva_info['names']['weight']
    mva_name = mva_info['names']['yhat']
    mva_tag = mva_info['name']
    significance_arrays[mva_tag] = array.array('d')
    purity_arrays[mva_tag] = array.array('d')
    resolution_arrays[mva_tag] = array.array('d')

    tmva_chain = ROOT.TChain(tree_name)
    tmva_chain.Add(tmva_filename)
    #luminosity = 137. + 110.
    luminosity = 1.
    hist = ROOT.TH1F("hist","hist",80,100,180)
    # Find mva thresholds
    mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name)
    sigeff70_mva_threshold = mva_thresholds[signal_fractions.index(0.7)]
    if verbose: print(f'signal_fractions: {signal_fractions} mva_thresholds: {mva_thresholds}')
    # mva_signal_width[threshold] = signal_width
    mva_signal_widths = {}
    # Find width of signals with MVA threshold
    for mva_threshold in mva_thresholds:
      entries = tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_name}>{mva_threshold})*{weight_name}',"goff")
      mva_signal_width = hist.GetStdDev()
      mva_signal_widths[mva_threshold] = mva_signal_width
      resolution_arrays[mva_tag].append(mva_signal_width)
      if verbose: print("mva threshold: "+str(mva_threshold)+" signal_width: "+str(mva_signal_width)+" entries: "+str(entries))
    # Find signal and background within 2 sigma of signal width
    significances = {}
    purities = {}
    # Find total signal and background events
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1)*{weight_name}',"goff")
    nevents_signal = hist.GetSum()
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0)*{weight_name}',"goff")
    nevents_background = hist.GetSum()
    if verbose: print(f'All nSig: {nevents_signal} nBkg: {nevents_background} signi: {nevents_signal/math.sqrt(nevents_background)}')
    for mva_threshold in mva_thresholds:
      sigma = 2.5
      mva_signal_width = mva_signal_widths[mva_threshold]
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_name}>{mva_threshold}&&{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma}))*{weight_name}*{luminosity}',"goff")
      nevents_signal = hist.GetSum()
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mva_name}>{mva_threshold}&&{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma}))*{weight_name}*{luminosity}',"goff")
      nevents_background = hist.GetSum()
      if verbose: print(f'signal width: {mva_signal_width} sigma: {sigma} mva_threshold: {mva_threshold} nSig: {nevents_signal} nBkg: {nevents_background}')
      # Calculate significance
      if nevents_background != 0:
        if verbose: print(f"mva_threshold: {mva_threshold:.4f} significance [s/sqrt(b)]: {nevents_signal/math.sqrt(nevents_background):.4f}")
        significances[mva_threshold] = nevents_signal/math.sqrt(nevents_background)
        significance_arrays[mva_tag].append(nevents_signal/math.sqrt(nevents_background))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have background events. Setting significance to 0.')
        significances[mva_threshold] = 0
        significance_arrays[mva_tag].append(0)
      # Calculate purity
      if nevents_background+nevents_signal !=0:
        if verbose: print(f"  mva_threshold: {mva_threshold:.4f} purity (s/(b+s)): {nevents_signal/(nevents_background+nevents_signal):.4f}")
        purity_arrays[mva_tag].append(nevents_signal/(nevents_background+nevents_signal))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have signal+background events. Setting purity to 0.')
        purity_arrays[mva_tag].append(0)
      if mva_threshold == sigeff70_mva_threshold:
        purity_sigeff70 = purity_arrays[mva_tag][-1]
    # Find best significance
    best_mva = -999
    best_significance = -999
    for mva_threshold in mva_thresholds:
      if significances[mva_threshold] > best_significance:
        best_significance = significances[mva_threshold]
        best_mva = mva_threshold
    print(f'{mva_tag} Best mva threshold: {best_mva:.4f} significance: {best_significance:.4f}')
    best_significances.append(best_significance)
    purity_sigeff70s.append(purity_sigeff70)
    # Make graph
    significance_graph = ROOT.TGraph(len(signal_fractions), x_array, significance_arrays[mva_tag])
    significance_graphs[mva_tag] = significance_graph
    purity_graph = ROOT.TGraph(len(signal_fractions), x_array, purity_arrays[mva_tag])
    purity_graphs[mva_tag] = purity_graph
    resolution_graph = ROOT.TGraph(len(signal_fractions), x_array, resolution_arrays[mva_tag])
    resolution_graphs[mva_tag] = resolution_graph
    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    significance_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(significance_min_max) == 0: significance_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
      if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {significance_min_max[0]}, min_max[1]: {significance_min_max[1]}')
    purity_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(purity_min_max) == 0: purity_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < purity_min_max[0]: purity_min_max[0] = y_min.value
      if y_max.value > purity_min_max[1]: purity_min_max[1] = y_max.value
    resolution_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(resolution_min_max) == 0: resolution_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < resolution_min_max[0]: resolution_min_max[0] = y_min.value
      if y_max.value > resolution_min_max[1]: resolution_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {purity_min_max[0]}, min_max[1]: {purity_min_max[1]}')
  # Draw graphs
  if draw:
    c1 = rootutils.new_canvas()
    c1.SetLeftMargin(0.15)
    significance_legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    colors = [1,2,3,4,6,7,8]
    for iMva, mva_tag in enumerate(significance_graphs):
      #significance_graphs[mva_tag].SetTitle(mva_tag)
      significance_graphs[mva_tag].SetTitle(f'{mva_tag};signal eff.;significance')
      if iMva == 0: significance_graphs[mva_tag].Draw("APL")
      else: significance_graphs[mva_tag].Draw("PL")
      significance_graphs[mva_tag].SetLineColor(colors[iMva])
      significance_graphs[mva_tag].SetMarkerColor(colors[iMva])
      significance_graphs[mva_tag].SetMarkerStyle(21)
      significance_graphs[mva_tag].SetMarkerSize(0.3)
      significance_graphs[mva_tag].GetYaxis().SetRangeUser(significance_min_max[0]-0.01, significance_min_max[1]+0.01)
      significance_legend.AddEntry(significance_graphs[mva_tag])
      significance_legend.Draw()
    c1.SaveAs('plots/significances_with_resolution.pdf')

    c2 = rootutils.new_canvas()
    c2.SetLeftMargin(0.15)
    purity_legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    colors = [1,2,3,4,6,7,8]
    for iMva, mva_tag in enumerate(purity_graphs):
      purity_graphs[mva_tag].SetTitle(f'{mva_tag};signal eff.;purity')
      if iMva == 0: purity_graphs[mva_tag].Draw("APL")
      else: purity_graphs[mva_tag].Draw("PL")
      purity_graphs[mva_tag].SetLineColor(colors[iMva])
      purity_graphs[mva_tag].SetMarkerColor(colors[iMva])
      purity_graphs[mva_tag].SetMarkerStyle(21)
      purity_graphs[mva_tag].SetMarkerSize(0.3)
      purity_graphs[mva_tag].GetYaxis().SetRangeUser(0, purity_min_max[1]+0.01)
      purity_legend.AddEntry(purity_graphs[mva_tag])
      purity_legend.Draw()
    c2.SaveAs('plots/purity_with_resolution.pdf')

    c3 = rootutils.new_canvas()
    c3.SetLeftMargin(0.15)
    resolution_legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    colors = [1,2,3,4,6,7,8]
    for iMva, mva_tag in enumerate(resolution_graphs):
      resolution_graphs[mva_tag].SetTitle(f'{mva_tag};signal eff.;signal resolution')
      if iMva == 0: resolution_graphs[mva_tag].Draw("APL")
      else: resolution_graphs[mva_tag].Draw("PL")
      resolution_graphs[mva_tag].SetLineColor(colors[iMva])
      resolution_graphs[mva_tag].SetMarkerColor(colors[iMva])
      resolution_graphs[mva_tag].SetMarkerStyle(21)
      resolution_graphs[mva_tag].SetMarkerSize(0.3)
      resolution_graphs[mva_tag].GetYaxis().SetRangeUser(resolution_min_max[0]-0.1, resolution_min_max[1]+0.1)
      resolution_legend.AddEntry(resolution_graphs[mva_tag])
      resolution_legend.Draw()
    c3.SaveAs('plots/signal_resolution.pdf')

  return best_significances, purity_sigeff70s

# mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
def evaluate_significance(info_mvas, draw=True, tree_type='test_tree'):
  verbose = False
  signal_fractions = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
  # Draw how significance changes for each mva
  x_array = array.array('d', signal_fractions)
  # significance_arrays[mva_name] = [signficance, ...]
  significance_arrays = {}
  purity_arrays = {}
  significance_graphs = {}
  purity_graphs = {}
  best_significances = []
  purity_sigeff70s = []
  significance_min_max = []
  purity_min_max = []
  for mva_info in info_mvas:
    tree_name = mva_info['names'][tree_type]
    if tree_type == 'train_tree': tmva_filename = mva_info['names']['train_filename']
    elif tree_type == 'test_full_tree': tmva_filename = mva_info['names']['test_full_filename']
    else: tmva_filename = mva_info['names']['test_filename']
    mass_name = mva_info['names']['observable']
    label_name = mva_info['names']['y']
    weight_name = mva_info['names']['weight']
    mva_name = mva_info['names']['yhat']
    mva_tag = mva_info['name']
    significance_arrays[mva_tag] = array.array('d')
    purity_arrays[mva_tag] = array.array('d')

    tmva_chain = ROOT.TChain(tree_name)
    tmva_chain.Add(tmva_filename)
    #luminosity = 137. + 110.
    luminosity = 1.
    hist = ROOT.TH1F("hist","hist",80,100,180)
    # Find mva thresholds
    mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name)
    sigeff70_mva_threshold = mva_thresholds[signal_fractions.index(0.7)]
    if verbose: print(f'signal_fractions: {signal_fractions} mva_thresholds: {mva_thresholds}')
    # Find signal and background within 2 sigma of signal width
    significances = {}
    purities = {}
    # Find total signal and background events
    tmva_chain.Draw(mass_name+">>hist","("+label_name+"==1)*"+weight_name+"*"+str(luminosity),"goff")
    nevents_signal = hist.GetSum()
    tmva_chain.Draw(mass_name+">>hist","("+label_name+"==0)*"+weight_name+"*"+str(luminosity),"goff")
    nevents_background = hist.GetSum()
    if verbose: print("All nSig: "+str(nevents_signal)+" nBkg: "+str(nevents_background))
    for mva_threshold in mva_thresholds:
      sigma = 2.5
      tmva_chain.Draw(mass_name+">>hist","("+label_name+"==1&&"+mva_name+">"+str(mva_threshold)+")*"+weight_name+"*"+str(luminosity),"goff")
      nevents_signal = hist.GetSum()
      tmva_chain.Draw(mass_name+">>hist","("+label_name+"==0&&"+mva_name+">"+str(mva_threshold)+")*"+weight_name+"*"+str(luminosity),"goff")
      nevents_background = hist.GetSum()
      if verbose: print("mva_threshold: "+str(mva_threshold)+" nSig: "+str(nevents_signal)+" nBkg: "+str(nevents_background))
      # Calculate significance
      if nevents_background != 0:
        if verbose: print(f"mva_threshold: {mva_threshold:.4f} significance [s/sqrt(b)]: {nevents_signal/math.sqrt(nevents_background):.4f}")
        significances[mva_threshold] = nevents_signal/math.sqrt(nevents_background)
        significance_arrays[mva_tag].append(nevents_signal/math.sqrt(nevents_background))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have background events. Setting significance to 0.')
        significances[mva_threshold] = 0
        significance_arrays[mva_tag].append(0)
      if nevents_background+nevents_signal !=0:
        if verbose: print(f"  mva_threshold: {mva_threshold:.4f} purity (s/(b+s)): {nevents_signal/(nevents_background+nevents_signal):.4f}")
        purity_arrays[mva_tag].append(nevents_signal/(nevents_background+nevents_signal))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have signal+background events. Setting purity to 0.')
        purity_arrays[mva_tag].append(0)
      if mva_threshold == sigeff70_mva_threshold:
        purity_sigeff70 = purity_arrays[mva_tag][-1]
    # Find best significance
    best_mva = -999
    best_significance = -999
    for mva_threshold in mva_thresholds:
      if significances[mva_threshold] > best_significance:
        best_significance = significances[mva_threshold]
        best_mva = mva_threshold
    #print(f'Best mva threshold: {best_mva:.4f} significance: {best_significance:.4f}')
    best_significances.append(best_significance)
    purity_sigeff70s.append(purity_sigeff70)
    # Make graph
    significance_graph = ROOT.TGraph(len(signal_fractions), x_array, significance_arrays[mva_tag])
    significance_graphs[mva_tag] = significance_graph
    purity_graph = ROOT.TGraph(len(signal_fractions), x_array, purity_arrays[mva_tag])
    purity_graphs[mva_tag] = purity_graph
    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    significance_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(significance_min_max) == 0: significance_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
      if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {significance_min_max[0]}, min_max[1]: {significance_min_max[1]}')
    purity_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(purity_min_max) == 0: purity_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < purity_min_max[0]: purity_min_max[0] = y_min.value
      if y_max.value > purity_min_max[1]: purity_min_max[1] = y_max.value
  # Draw graphs
  if draw:
    c1 = rootutils.new_canvas()
    c1.SetLeftMargin(0.15)
    significance_legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    colors = [1,2,3,4,6,7,8]
    for iMva, mva_tag in enumerate(significance_graphs):
      significance_graphs[mva_tag].SetTitle(f'{mva_tag};signal eff.;significance')
      if iMva == 0: significance_graphs[mva_tag].Draw("APL")
      else: significance_graphs[mva_tag].Draw("PL")
      significance_graphs[mva_tag].SetLineColor(colors[iMva])
      significance_graphs[mva_tag].SetMarkerColor(colors[iMva])
      significance_graphs[mva_tag].SetMarkerStyle(21)
      significance_graphs[mva_tag].SetMarkerSize(0.3)
      significance_graphs[mva_tag].GetYaxis().SetRangeUser(significance_min_max[0]-0.01, significance_min_max[1]+0.01)
      significance_legend.AddEntry(significance_graphs[mva_tag])
      significance_legend.Draw()
    c1.SaveAs('plots/significances.pdf')

    c2 = ROOT.TCanvas('c2', 'c2', 500, 500)
    c2.SetLeftMargin(0.15)
    purity_legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    colors = [1,2,3,4,6,7,8]
    for iMva, mva_tag in enumerate(purity_graphs):
      purity_graphs[mva_tag].SetTitle(f'{mva_tag};signal eff.;purity')
      if iMva == 0: purity_graphs[mva_tag].Draw("APL")
      else: purity_graphs[mva_tag].Draw("PL")
      purity_graphs[mva_tag].SetLineColor(colors[iMva])
      purity_graphs[mva_tag].SetMarkerColor(colors[iMva])
      purity_graphs[mva_tag].SetMarkerStyle(21)
      purity_graphs[mva_tag].SetMarkerSize(0.3)
      purity_graphs[mva_tag].GetYaxis().SetRangeUser(0, purity_min_max[1]+0.01)
      purity_legend.AddEntry(purity_graphs[mva_tag])
      purity_legend.Draw()
    c2.SaveAs('plots/purity.pdf')
  return best_significances, purity_sigeff70s

# mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
def evaluate_correlation(info_mvas, draw=True, tree_type='test_tree'):
  verbose = False
  std_divs = []
  for mva_info in info_mvas:
    if tree_type == 'train_tree': tmva_filename = mva_info['names']['train_filename']
    elif tree_type == 'test_full_tree': tmva_filename = mva_info['names']['test_full_filename']
    else: tmva_filename = mva_info['names']['test_filename']
    tree_name = mva_info['names'][tree_type]
    mass_name = mva_info['names']['observable']
    label_name = mva_info['names']['y']
    mva_name = mva_info['names']['yhat']
    mva_tag = mva_info['name']
    weight_name = mva_info['names']['weight']

    tmva_chain = ROOT.TChain(tree_name)
    tmva_chain.Add(tmva_filename)
    # Find mva thresholds
    signal_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name)
    mva_max = tmva_chain.GetMaximum(mva_name)
    mva_thresholds.append(mva_max)
    # Make mass histograms with thresholds
    mva_hists = {}
    signal_hists = {}
    ROOT.gStyle.SetOptStat(0)
    #print(f'mva_thresholds: {mva_thresholds}')
    # Find min mva
    for imva, mva_threshold in enumerate(mva_thresholds):
      # Background hists
      hist = ROOT.TH1F("hist_"+str(mva_threshold),"hist_"+str(mva_threshold),160,100,180)
      if imva == 0:
        entries = tmva_chain.Draw(mass_name+">>hist_"+str(mva_threshold),f'{label_name}==0&&{mva_name}<{mva_threshold}',"goff")
        #print(f'{label_name}==0&&{mva_name}<{mva_threshold}, {entries}')
      elif imva == len(mva_thresholds)-1:
        entries = tmva_chain.Draw(mass_name+">>hist_"+str(mva_threshold),f'{label_name}==0&&{mva_name}>{mva_threshold}',"goff")
        #print(f'{label_name}==0&&{mva_name}>{prev_mva_threshold}, {entries}')
      else:
        entries = tmva_chain.Draw(mass_name+">>hist_"+str(mva_threshold),f'{label_name}==0&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}',"goff")
        #print(f'{label_name}==0&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}, {entries}')
      # Normalize histogram
      sum_weight = hist.GetSumOfWeights()
      if sum_weight != 0:
        hist.Scale(1/sum_weight)
        mva_hists[mva_threshold] = hist
      # Signal hists
      signal_hist = ROOT.TH1F("signal_hist_"+str(mva_threshold),"signal_hist_"+str(mva_threshold),160,100,180)
      if imva == 0:
        #print(f'{label_name}==1&&{mva_name}<{mva_threshold}')
        entries = tmva_chain.Draw(mass_name+">>signal_hist_"+str(mva_threshold),f'{label_name}==1&&{mva_name}<{mva_threshold}',"goff")
      elif imva == len(mva_thresholds)-1:
        #print(f'{label_name}==1&&{mva_name}>{prev_mva_threshold}')
        entries = tmva_chain.Draw(mass_name+">>signal_hist_"+str(mva_threshold),f'{label_name}==1&&{mva_name}>{mva_threshold}',"goff")
      else:
        #print(f'{label_name}==1&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}')
        entries = tmva_chain.Draw(mass_name+">>signal_hist_"+str(mva_threshold),f'{label_name}==1&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}',"goff")
      prev_mva_threshold = mva_threshold
      # Normalize histogram
      sum_weight = signal_hist.GetSumOfWeights()
      if sum_weight != 0:
        signal_hist.Scale(1/sum_weight)
        signal_hists[mva_threshold] = signal_hist
    # Get values of histogram bin. 1 is first bin, nbin is last bin
    #print(mva_hists[mva_thresholds[0]].GetNbinsX())
    std_values = []
    for iBin in range(mva_hists[mva_thresholds[0]].GetNbinsX()):
      bin_values = []
      for mva_threshold in mva_hists:
        hist_ibin_entry = mva_hists[mva_threshold].GetBinContent(iBin+1)
        bin_values.append(hist_ibin_entry)
        #print("iBin: "+str(iBin+1)+" mva_threshold: "+str(mva_threshold)+" "+str(hist_ibin_entry))
      stdev = np.std(bin_values, dtype=np.float64)
      mean = np.mean(bin_values, dtype=np.float64)
      if (stdev == 0 and mean == 0): continue # Ignore case where hist bins are empty
      if verbose: print("iBin: "+str(iBin+1)+" x: "+str(mva_hists[mva_thresholds[0]].GetBinCenter(iBin+1))+" stdev: "+str(stdev))
      std_values.append(stdev)
    #print(mva_tag+" mean stdev: "+str(np.mean(std_values)))
    std_divs.append(np.mean(std_values))
    if draw:
      # background
      c1 = rootutils.new_canvas()
      legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.98)
      bkg_hist = ROOT.TH1F("bkg_hist","Background dist.;m_{llg} [GeV]",160,100,180)
      bkg_hist.Draw()
      colors = [ROOT.kGreen-8, ROOT.kGreen-5, ROOT.kGreen+4,
                ROOT.kBlue-10, ROOT.kBlue-7, ROOT.kBlue,
                ROOT.kRed-10, ROOT.kRed-7, ROOT.kRed,]
      for imva, mva_threshold in enumerate(mva_hists):
        hist = mva_hists[mva_threshold]
        hist.SetLineColor(colors[imva])
        hist.Draw('same')
        legend.AddEntry(hist)
      legend.Draw()
      box = ROOT.TPaveText(0.12, 0.1, 0.5, 0.2, 'NDC NB')
      box.SetFillColorAlpha(0,0)
      box.AddText(f'Mean stdev: {np.mean(std_values):.5f}')
      box.Draw()
      rootutils.set_max_th1()
      c1.SaveAs("plots/bkg_hist_mva_"+rootutils.slugify(mva_tag)+".pdf")

      # signal
      c2 = rootutils.new_canvas()
      legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.98)
      signal_hist = ROOT.TH1F("signal_hist","Signal dist.;m_{llg} [GeV]",160,100,180)
      signal_hist.Draw('hist')
      for imva, mva_threshold in enumerate(signal_hists):
        hist = signal_hists[mva_threshold]
        hist.SetLineColor(colors[imva])
        hist.Draw('same hist')
        legend.AddEntry(hist)
      legend.Draw()
      rootutils.set_max_th1()
      c2.SaveAs("plots/signal_hist_mva_"+rootutils.slugify(mva_tag)+".pdf")
      ROOT.gStyle.SetOptStat(1)

  return std_divs

# mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
def evaluate_tprfpr(info_mvas, draw=True, tree_type='test_tree'):
  verbose = False
  signal_fractions = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
  # Draw how significance changes for each mva
  x_array = array.array('d', signal_fractions)
  # significance_arrays[mva_name] = [signficance, ...]
  significance_arrays = {}
  purity_arrays = {}
  tpr_arrays = {}
  fpr_arrays = {}
  resolution_arrays = {}
  significance_graphs = {}
  purity_graphs = {}
  tpr_graphs = {}
  fpr_graphs = {}
  resolution_graphs = {}
  best_significances = []
  purity_sigeff70s = []
  significance_min_max = []
  purity_min_max = []
  tpr_min_max = []
  fpr_min_max = []
  resolution_min_max = []
  for mva_info in info_mvas:
    tree_name = mva_info['names'][tree_type]
    if tree_type == 'train_tree': tmva_filename = mva_info['names']['train_filename']
    elif tree_type == 'test_full_tree': tmva_filename = mva_info['names']['test_full_filename']
    else: tmva_filename = mva_info['names']['test_filename']
    mass_name = mva_info['names']['observable']
    label_name = mva_info['names']['y']
    weight_name = mva_info['names']['weight']
    mva_name = mva_info['names']['yhat']
    mva_tag = mva_info['name']
    significance_arrays[mva_tag] = array.array('d')
    purity_arrays[mva_tag] = array.array('d')
    tpr_arrays[mva_tag] = array.array('d')
    fpr_arrays[mva_tag] = array.array('d')
    resolution_arrays[mva_tag] = array.array('d')

    tmva_chain = ROOT.TChain(tree_name)
    tmva_chain.Add(tmva_filename)
    #luminosity = 137. + 110.
    luminosity = 1.
    hist = ROOT.TH1F("hist","hist",160,100,180)
    # Find mva thresholds
    mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name)
    sigeff70_mva_threshold = mva_thresholds[signal_fractions.index(0.7)]
    if verbose: print(f'signal_fractions: {signal_fractions} mva_thresholds: {mva_thresholds}')
    # mva_signal_width[threshold] = signal_width
    mva_signal_widths = {}
    # Find width of signals with MVA threshold
    for mva_threshold in mva_thresholds:
      entries = tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_name}>{mva_threshold})*{weight_name}',"goff")
      mva_signal_width = hist.GetStdDev()
      mva_signal_widths[mva_threshold] = mva_signal_width
      resolution_arrays[mva_tag].append(mva_signal_width)
      if verbose: print("mva threshold: "+str(mva_threshold)+" signal_width: "+str(mva_signal_width)+" entries: "+str(entries))
    # Find signal and background within 2 sigma of signal width
    significances = {}
    purities = {}
    sigma = 2
    #weight_name = '1'
    cut = f'1'
    #cut = f'{weight_name}>0'
    # Find total signal and background events
    mva_signal_width = 2.5
    mass_window = f'{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})'
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mass_window})*{weight_name}',"goff")
    nevents_signal = hist.GetSum()
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mass_window})*{weight_name}',"goff")
    nevents_background = hist.GetSum()
    if verbose: print(f'All nSig: {nevents_signal} nBkg: {nevents_background} signi: {nevents_signal/math.sqrt(nevents_background)}')
    for mva_threshold in mva_thresholds:
      #mva_signal_width = mva_signal_widths[mva_threshold]
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_name}>{mva_threshold}&&{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})&&{cut})*{weight_name}*{luminosity}',"goff")
      nevents_signal = hist.GetSum()
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mva_name}>{mva_threshold}&&{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})&&{cut})*{weight_name}*{luminosity}',"goff")
      nevents_background = hist.GetSum()

      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_name}<{mva_threshold}&&{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})&&{cut})*{weight_name}*{luminosity}',"goff")
      nevents_missed_signal = hist.GetSum()
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mva_name}<{mva_threshold}&&{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})&&{cut})*{weight_name}*{luminosity}',"goff")
      nevents_removed_background = hist.GetSum()

      if verbose: print(f'signal width: {mva_signal_width} sigma: {sigma} mva_threshold: {mva_threshold} nSig: {nevents_signal} nBkg: {nevents_background}')
      # Calculate significance
      if nevents_background != 0:
        if verbose: print(f"mva_threshold: {mva_threshold:.4f} significance [s/sqrt(b)]: {nevents_signal/math.sqrt(nevents_background):.4f}")
        significances[mva_threshold] = nevents_signal/math.sqrt(nevents_background)
        significance_arrays[mva_tag].append(nevents_signal/math.sqrt(nevents_background))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have background events. Setting significance to 0.')
        significances[mva_threshold] = 0
        significance_arrays[mva_tag].append(0)
      # Calculate purity
      if nevents_background+nevents_signal !=0:
        if verbose: print(f"  mva_threshold: {mva_threshold:.4f} purity (s/(b+s)): {nevents_signal/(nevents_background+nevents_signal):.4f}")
        purity_arrays[mva_tag].append(nevents_signal/(nevents_background+nevents_signal))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have signal+background events. Setting purity to 0.')
        purity_arrays[mva_tag].append(0)
      if mva_threshold == sigeff70_mva_threshold:
        purity_sigeff70 = purity_arrays[mva_tag][-1]
      # Calculate tpr fpr
      tpr_arrays[mva_tag].append(nevents_signal/(nevents_signal+nevents_missed_signal))
      fpr_arrays[mva_tag].append(nevents_background/(nevents_background+nevents_removed_background))

    # Make graph
    significance_graph = ROOT.TGraph(len(signal_fractions), x_array, significance_arrays[mva_tag])
    significance_graphs[mva_tag] = significance_graph
    purity_graph = ROOT.TGraph(len(signal_fractions), x_array, purity_arrays[mva_tag])
    purity_graphs[mva_tag] = purity_graph
    resolution_graph = ROOT.TGraph(len(signal_fractions), x_array, resolution_arrays[mva_tag])
    resolution_graphs[mva_tag] = resolution_graph
    tpr_graph = ROOT.TGraph(len(signal_fractions), x_array, tpr_arrays[mva_tag])
    tpr_graphs[mva_tag] = tpr_graph
    fpr_graph = ROOT.TGraph(len(signal_fractions), x_array, fpr_arrays[mva_tag])
    fpr_graphs[mva_tag] = fpr_graph
    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    significance_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(significance_min_max) == 0: significance_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
      if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {significance_min_max[0]}, min_max[1]: {significance_min_max[1]}')
    purity_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(purity_min_max) == 0: purity_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < purity_min_max[0]: purity_min_max[0] = y_min.value
      if y_max.value > purity_min_max[1]: purity_min_max[1] = y_max.value
    resolution_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(resolution_min_max) == 0: resolution_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < resolution_min_max[0]: resolution_min_max[0] = y_min.value
      if y_max.value > resolution_min_max[1]: resolution_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {purity_min_max[0]}, min_max[1]: {purity_min_max[1]}')
    tpr_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(tpr_min_max) == 0: tpr_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < tpr_min_max[0]: tpr_min_max[0] = y_min.value
      if y_max.value > tpr_min_max[1]: tpr_min_max[1] = y_max.value
    fpr_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(fpr_min_max) == 0: fpr_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < fpr_min_max[0]: fpr_min_max[0] = y_min.value
      if y_max.value > fpr_min_max[1]: fpr_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {purity_min_max[0]}, min_max[1]: {purity_min_max[1]}')
  # Draw graphs
  if draw:
    c1 = rootutils.new_canvas()
    c1.SetLeftMargin(0.15)
    tpr_legend = ROOT.TLegend(0.2, 0.6, 0.5, 0.98)
    colors = [1,2,3,4,6,7,8,9,46,49,41,38]
    for iMva, mva_tag in enumerate(tpr_graphs):
      tpr_graphs[mva_tag].SetTitle(f';signal eff.;tpr')
      if iMva == 0: tpr_graphs[mva_tag].Draw("APL")
      else: tpr_graphs[mva_tag].Draw("PL")
      tpr_graphs[mva_tag].SetLineColor(colors[iMva])
      tpr_graphs[mva_tag].SetMarkerColor(colors[iMva])
      tpr_graphs[mva_tag].SetMarkerStyle(21)
      tpr_graphs[mva_tag].SetMarkerSize(0.3)
      tpr_graphs[mva_tag].GetYaxis().SetRangeUser(0, tpr_min_max[1]+0.01)
      tpr_legend.AddEntry(tpr_graphs[mva_tag], f'{mva_tag}')
      tpr_legend.Draw()
    c1.SaveAs('plots/tpr_with_resolution.pdf')

    c2 = rootutils.new_canvas()
    c2.SetLeftMargin(0.15)
    fpr_legend = ROOT.TLegend(0.2, 0.6, 0.5, 0.98)
    colors = [1,2,3,4,6,7,8,9,46,49,41,38]
    for iMva, mva_tag in enumerate(fpr_graphs):
      fpr_graphs[mva_tag].SetTitle(f';signal eff.;fpr')
      if iMva == 0: fpr_graphs[mva_tag].Draw("APL")
      else: fpr_graphs[mva_tag].Draw("PL")
      fpr_graphs[mva_tag].SetLineColor(colors[iMva])
      fpr_graphs[mva_tag].SetMarkerColor(colors[iMva])
      fpr_graphs[mva_tag].SetMarkerStyle(21)
      fpr_graphs[mva_tag].SetMarkerSize(0.3)
      fpr_graphs[mva_tag].GetYaxis().SetRangeUser(0, fpr_min_max[1]+0.01)
      fpr_legend.AddEntry(fpr_graphs[mva_tag], f'{mva_tag}')
      fpr_legend.Draw()
    c2.SaveAs('plots/fpr_with_resolution.pdf')

  return best_significances, purity_sigeff70s


if __name__ == "__main__":

  # Load data
  # mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
  #tmva_bdt = load_tmva_dict('ntuples_mva/TMVA_bdt.root', 'BDT', 'tmva bdt')
  #tmva_bdt = load_tmva_eval_dict('ntuples_mva/TMVA_bdt.root', 'tmva_evaluate_bdt.root', 'BDT', 'tmva bdt')
  #tmva_nn = load_tmva_res_dict('ntuples_mva/TMVA_nn.root', 'DNN', 'tmva nn')
  #tmva_nn = load_tmva_eval_res_dict('ntuples_mva/TMVA_nn.root', 'tmva_evaluate_nn.root','DNN', 'tmva nn')
  #gbdt = load_mva_dict('ntuples_mva/gbdt.root', 'gbdt')
  #xgbdt = load_mva_dict('ntuples_mva/xgbdt.root', 'xgbdt')
  #torch_nn = load_mva_dict('ntuples_mva/torch_nn.root', 'torch nn')
  #fine_nn = load_mva_dict('ntuples_mva/fine_nn.root', 'fine nn')
  #torch_nn_batch32 = load_mva_dict('ntuples_mva/torch_nn_batch32.root', 'torch nn (batch=32)')
  #torch_nn_batch4096 = load_mva_dict('ntuples_mva/torch_nn_batch4096.root', 'torch nn (batch=4096)')
  #fine_nn_batch32 = load_mva_dict('ntuples_mva/fine_nn_batch32.root', 'fine nn (batch=32)')
  #fine_nn_batch32_loss_signi = load_mva_dict('ntuples_mva/fine_nn_loss1_batch32.root', 'fine nn loss signi (batch=32)')
  #fine_nn_batch32_loss_z = load_mva_dict('ntuples_mva/fine_nn_loss2_batch32.root', 'fine nn loss z (batch=32)')
  #fine_nn_batch32_loss_purity = load_mva_dict('ntuples_mva/fine_nn_loss3_batch32.root', 'fine nn loss purity (batch=32)')
  #fine_nn_batch128_loss_z = load_mva_dict('ntuples_mva/fine_nn_loss2_batch128.root', 'fine nn loss z (batch=128)')
  #fine_nn_batch128_loss_signi_res = load_mva_dict('ntuples_mva/fine_nn_loss100_batch128.root', 'fine nn res loss (batch=128)')
  #test_mva = load_mva_dict('nn_evaluate.root', 'test mva')

  #info_mvas = [tmva_bdt, tmva_nn, gbdt, xgbdt, torch_nn, fine_nn, torch_nn_batch32, fine_nn_batch32]
  #info_mvas = [tmva_bdt, tmva_nn, gbdt, xgbdt, torch_nn_batch32, fine_nn]
  #info_mvas = [tmva_bdt, torch_nn_batch32]
  #info_mvas = [tmva_bdt, tmva_nn, fine_nn_batch128_loss_signi_res, gbdt, xgbdt]
  #info_mvas = [tmva_bdt, tmva_nn, gbdt, xgbdt, torch_nn_batch4096, test_mva]
  #info_mvas = [tmva_bdt, gbdt, xgbdt, test_mva]
  #info_mvas = [tmva_nn, fine_nn_batch32, fine_nn_batch32_loss_signi, fine_nn_batch32_loss_z, fine_nn_batch32_loss_purity]
  #info_mvas = [tmva_bdt, tmva_nn, fine_nn_batch128_loss_z, fine_nn_batch128_loss_signi_res, test_mva]
  #info_mvas = [tmva_bdt, tmva_nn, gbdt, xgbdt, fine_nn_batch128_loss_signi_res]

  #tmva_simple_bdt = load_tmva_simple_eval_dict('ntuples_mva/TMVA_simple_bdt_run2.root', 'tmva_evaluate_simple_bdt_run2.root', 'BDT', 'tmva simple bdt')
  #tmva_bdt = load_tmva_eval_dict('ntuples_mva/TMVA_bdt_run2.root', 'tmva_evaluate_bdt_run2.root', 'BDT', 'tmva bdt')
  #gbdt = load_mva_dict('ntuples_mva/gbdt_run2.root', 'gbdt')
  #xgbdt = load_mva_dict('ntuples_mva/xgbdt_run2.root', 'xgbdt')
  ##disco_mva = load_mva_dict('nn_evaluate.root', 'nn disco')
  #entropy_mva = load_mva_dict('nn_loss0_run2.root', 'nn')
  #disco_mva = load_mva_dict('nn_loss201_run2.root', 'nn disco')
  #signi_mva = load_mva_dict('nn_loss203_run2.root', 'signi mva')
  ##signi_i11_mva = load_mva_dict('nn_loss203_i11_run2.root', 'signi_i11 mva')
  #signi_i11_fullwin_mva = load_mva_dict('nn_loss203_i11_fullwin_run2.root', 'signi i11 fullwin mva')

  run2_bdt = load_tmva_eval_dict('ntuples_mva/run2_bdt.root', 'tmva_evaluate_run2_bdt.root', 'BDT', 'run2 bdt', dataset_name='run2_bdt', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  reduced_bdt = load_tmva_eval_dict('ntuples_mva/reduced_bdt.root', 'tmva_evaluate_reduced_bdt.root', 'BDT', 'reduced bdt', dataset_name='reduced_bdt')
  raw_bdt = load_tmva_eval_dict('ntuples_mva/raw_bdt.root', 'tmva_evaluate_raw_bdt.root', 'BDT', 'raw bdt', dataset_name='raw_bdt', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  var12_bdt = load_tmva_eval_dict('ntuples_mva/var12_bdt.root', 'tmva_evaluate_var12_bdt.root', 'BDT', 'var12 bdt', dataset_name='var12_bdt', mva_input=['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 'llg_mass_err', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity', 'llg_flavor', 'max_dR', 'llg_ptt'])
  #signi_mva = load_mva_dict('nn_loss203_run2.root', 'signi mva')
  #signi_i11_fullwin_mva = load_mva_dict('nn_loss203_i11_fullwin_run2.root', 'signi i11 fullwin mva')
  gbdt_decorr = load_mva_dict('ntuples_mva/gbdt_run2_decorr.root', 'gbdt decorr')
  xgbdt_decorr = load_mva_dict('ntuples_mva/xgbdt_run2_decorr.root', 'xgbdt decorr')
  gbdt = load_mva_dict('ntuples_mva/gbdt_run2.root', 'gbdt')
  xgbdt = load_mva_dict('ntuples_mva/xgbdt_run2.root', 'xgbdt')
  gbdt_noweight_decorr = load_mva_dict('ntuples_mva/gbdt_run2_noweight_decorr.root', 'gbdt nowgt decorr')
  xgbdt_noweight_decorr = load_mva_dict('ntuples_mva/xgbdt_run2_noweight_decorr.root', 'xgbdt nowgt decorr')
  gbdt_noweight = load_mva_dict('ntuples_mva/gbdt_run2_noweight.root', 'gbdt nowgt ')
  xgbdt_noweight = load_mva_dict('ntuples_mva/xgbdt_run2_noweight.root', 'xgbdt nowgt')
  entropy_mva = load_mva_dict('nn_loss5_run2.root', 'nn')
  disco_mva = load_mva_dict('nn_loss201_run2.root', 'nn disco')
  #disco_mva = load_mva_dict('ntuples_mva/nn_runs_loss201.root', 'disco')
  #logsigni_mva = load_mva_dict('ntuples_mva/nn_runs_loss202.root', 'logsigni')
  #signi_mva = load_mva_dict('ntuples_mva/nn_runs_loss300.root', 'signi')
  #signi_mva = load_mva_dict('nn_loss300_run2.root', 'signi')
  signi_mva = load_mva_dict('nn_loss207_run2.root', 'signi')

  scaled_entropy_mva = load_mva_dict('nn_loss6_run2.root', 'scaled entropy')
  signi_only_mva = load_mva_dict('nn_loss7_run2.root', 'signi')
  signi_res_mva = load_mva_dict('nn_loss8_run2.root', 'signi res')
  scaled_entropy_signi_mva = load_mva_dict('nn_loss9_run2.root', 'entropy+signi res')

  #k1_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s1.root', 'ntuples_kfold/eval_reduced_bdt_s1.root', 'BDT', 'reduced bdt kfold1', dataset_name='reduced_bdt_s1')
  #k2_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s2.root', 'ntuples_kfold/eval_reduced_bdt_s2.root', 'BDT', 'reduced bdt kfold2', dataset_name='reduced_bdt_s2')
  #k3_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s3.root', 'ntuples_kfold/eval_reduced_bdt_s3.root', 'BDT', 'reduced bdt kfold3', dataset_name='reduced_bdt_s3')
  #k4_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s4.root', 'ntuples_kfold/eval_reduced_bdt_s4.root', 'BDT', 'reduced bdt kfold4', dataset_name='reduced_bdt_s4')
  #k5_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s5.root', 'ntuples_kfold/eval_reduced_bdt_s5.root', 'BDT', 'reduced bdt kfold5', dataset_name='reduced_bdt_s5')
  #k6_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s6.root', 'ntuples_kfold/eval_reduced_bdt_s6.root', 'BDT', 'reduced bdt kfold6', dataset_name='reduced_bdt_s6')
  #k7_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s7.root', 'ntuples_kfold/eval_reduced_bdt_s7.root', 'BDT', 'reduced bdt kfold7', dataset_name='reduced_bdt_s7')
  #k8_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s8.root', 'ntuples_kfold/eval_reduced_bdt_s8.root', 'BDT', 'reduced bdt kfold8', dataset_name='reduced_bdt_s8')
  #k9_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s9.root', 'ntuples_kfold/eval_reduced_bdt_s9.root', 'BDT', 'reduced bdt kfold9', dataset_name='reduced_bdt_s9')
  #k10_reduced_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/reduced_bdt_s10.root', 'ntuples_kfold/eval_reduced_bdt_s10.root', 'BDT', 'reduced bdt kfold10', dataset_name='reduced_bdt_s10')

  #k1_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s1.root', 'ntuples_kfold/eval_raw_bdt_s1.root', 'BDT', 'raw bdt kfold1', dataset_name='raw_bdt_s1', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k2_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s2.root', 'ntuples_kfold/eval_raw_bdt_s2.root', 'BDT', 'raw bdt kfold2', dataset_name='raw_bdt_s2', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k3_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s3.root', 'ntuples_kfold/eval_raw_bdt_s3.root', 'BDT', 'raw bdt kfold3', dataset_name='raw_bdt_s3', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k4_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s4.root', 'ntuples_kfold/eval_raw_bdt_s4.root', 'BDT', 'raw bdt kfold4', dataset_name='raw_bdt_s4', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k5_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s5.root', 'ntuples_kfold/eval_raw_bdt_s5.root', 'BDT', 'raw bdt kfold5', dataset_name='raw_bdt_s5', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k6_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s6.root', 'ntuples_kfold/eval_raw_bdt_s6.root', 'BDT', 'raw bdt kfold6', dataset_name='raw_bdt_s6', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k7_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s7.root', 'ntuples_kfold/eval_raw_bdt_s7.root', 'BDT', 'raw bdt kfold7', dataset_name='raw_bdt_s7', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k8_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s8.root', 'ntuples_kfold/eval_raw_bdt_s8.root', 'BDT', 'raw bdt kfold8', dataset_name='raw_bdt_s8', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k9_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s9.root', 'ntuples_kfold/eval_raw_bdt_s9.root', 'BDT', 'raw bdt kfold9', dataset_name='raw_bdt_s9', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])
  #k10_raw_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/raw_bdt_s10.root', 'ntuples_kfold/eval_raw_bdt_s10.root', 'BDT', 'raw bdt kfold10', dataset_name='raw_bdt_s10', mva_input=['photon_mva', 'llg_ptt', 'llg_eta', 'llg_phi', 'z_eta', 'z_phi', 'l1_rapidity', 'l1_phi', 'l2_rapidity', 'l2_phi', 'gamma_eta', 'gamma_phi'])

  #k1_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s1.root', 'ntuples_kfold/eval_run2_bdt_s1.root', 'BDT', 'run2 bdt kfold1', dataset_name='run2_bdt_s1', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k2_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s2.root', 'ntuples_kfold/eval_run2_bdt_s2.root', 'BDT', 'run2 bdt kfold2', dataset_name='run2_bdt_s2', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k3_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s3.root', 'ntuples_kfold/eval_run2_bdt_s3.root', 'BDT', 'run2 bdt kfold3', dataset_name='run2_bdt_s3', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k4_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s4.root', 'ntuples_kfold/eval_run2_bdt_s4.root', 'BDT', 'run2 bdt kfold4', dataset_name='run2_bdt_s4', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k5_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s5.root', 'ntuples_kfold/eval_run2_bdt_s5.root', 'BDT', 'run2 bdt kfold5', dataset_name='run2_bdt_s5', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k6_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s6.root', 'ntuples_kfold/eval_run2_bdt_s6.root', 'BDT', 'run2 bdt kfold6', dataset_name='run2_bdt_s6', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k7_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s7.root', 'ntuples_kfold/eval_run2_bdt_s7.root', 'BDT', 'run2 bdt kfold7', dataset_name='run2_bdt_s7', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k8_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s8.root', 'ntuples_kfold/eval_run2_bdt_s8.root', 'BDT', 'run2 bdt kfold8', dataset_name='run2_bdt_s8', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k9_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s9.root', 'ntuples_kfold/eval_run2_bdt_s9.root', 'BDT', 'run2 bdt kfold9', dataset_name='run2_bdt_s9', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])
  #k10_run2_bdt = load_kfold_tmva_eval_dict('ntuples_kfold/run2_bdt_s10.root', 'ntuples_kfold/eval_run2_bdt_s10.root', 'BDT', 'run2 bdt kfold10', dataset_name='run2_bdt_s10', mva_input=['photon_mva', 'min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 'phi', 'photon_res_e', 'photon_rapidity', 'l1_rapidity', 'l2_rapidity',])


  print('Loaded dicts')

  info_mvas = [scaled_entropy_mva, signi_only_mva, signi_res_mva, scaled_entropy_signi_mva]
  #info_mvas = [run2_bdt,reduced_bdt,raw_bdt,var12_bdt,signi_mva]
  #info_mvas = [reduced_bdt,k1_reduced_bdt,k2_reduced_bdt,k3_reduced_bdt,k4_reduced_bdt,k5_reduced_bdt,k6_reduced_bdt,k7_reduced_bdt,k8_reduced_bdt,k9_reduced_bdt,k10_reduced_bdt,]
  #info_mvas = [raw_bdt,k1_raw_bdt,k2_raw_bdt,k3_raw_bdt,k4_raw_bdt,k5_raw_bdt,k6_raw_bdt,k7_raw_bdt,k8_raw_bdt,k9_raw_bdt,k10_raw_bdt,]
  #info_mvas = [run2_bdt,k1_run2_bdt,k2_run2_bdt,k3_run2_bdt,k4_run2_bdt,k5_run2_bdt,k6_run2_bdt,k7_run2_bdt,k8_run2_bdt,k9_run2_bdt,k10_run2_bdt,]
  #info_mvas= [reduced_bdt,gbdt,xgbdt,gbdt_decorr,xgbdt_decorr,entropy_mva,disco_mva,signi_mva]
  #info_mvas = [run2_bdt]
  #info_mvas = [run2_bdt,raw_bdt,gbdt,xgbdt,signi_mva,signi_i11_fullwin_mva]
  #info_mvas = [run2_bdt,xgbdt,entropy_mva,disco_mva]
  #info_mvas = [tmva_bdt, gbdt, disco_mva, signi_mva, signi_i11_mva, signi_i11_fullwin_mva, tmva_simple_bdt]
  #info_mvas = [tmva_bdt, tmva_simple_bdt, gbdt, entropy_mva, disco_mva, signi_mva, signi_i11_fullwin_mva]
  #info_mvas = [tmva_bdt]
  #info_mvas = [tmva_bdt, tmva_simple_bdt]
  #info_mvas = [entropy_mva]
  #info_mvas = [run2_bdt, xgbdt, xgbdt_noweight]

  # To check overtraining
  print('On train tree')
  evaluate_significance_bins_with_resolution(info_mvas, tree_type='train_tree', draw=False)
  print('On test tree')
  evaluate_significance_bins_with_resolution(info_mvas, tree_type='test_tree', draw=False)

  evaluate_roc(info_mvas) # negative weights set to 0
  #evaluate_roc(info_mvas, no_weight=True) # unweighted roc 
  evaluate_overtraining(info_mvas)
  evaluate_significance_bins_with_resolution(info_mvas, tree_type='test_full_tree')
  evaluate_tprfpr(info_mvas, tree_type='test_full_tree')
  evaluate_correlation(info_mvas, tree_type='test_full_tree')


  #evaluate_significance_with_resolution(info_mvas, tree_type='test_full_tree')
  #evaluate_significance(info_mvas, tree_type='test_full_tree')
  #evaluate_significance_with_resolution(info_mvas, tree_type='train_tree')
  #evaluate_significance(info_mvas, tree_type='train_tree')
  #evaluate_correlation(info_mvas, tree_type='train_tree')


