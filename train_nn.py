#!/usr/bin/env python3
import ROOT
from torch import nn
import torch.utils.data
import torch
import uproot
import numpy as np
import xml.etree.ElementTree as ET
from array import array
import math
import copy
import sklearn.metrics
import sklearn.ensemble
import matplotlib.pyplot as plt
import xgboost
from torch.utils.tensorboard import SummaryWriter
import evaluate
import Disco
import time

def unnormalize(values, norm_weights):
  feature_array = copy.deepcopy(values)
  for ifeat, [min_x, max_x] in enumerate(norm_weights):
    feature_array[:,ifeat] = (values[:,ifeat]+1)*(max_x-min_x)*1./2 + min_x
  return feature_array

def find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name):
  # Find MVA threshold where signal is the below fractions
  mva_thresholds = []
  mva_range = [-1.,1.]
  niter = 5000
  signal_entries = tmva_chain.Draw(mass_name+">>hist",label_name+"==1","goff")
  #print("All signal entries: "+str(signal_entries))
  iFraction = 0
  for iRange in range(niter):
    mva_threshold = (mva_range[1] - mva_range[0])/niter * iRange + mva_range[0]
    entries = tmva_chain.Draw(mass_name+">>hist",label_name+"==1&&"+mva_name+">"+str(mva_threshold)+"","goff")
    fraction = entries *1. / signal_entries
    if (fraction < signal_fractions[iFraction]):
      mva_thresholds.append(mva_threshold)
      iFraction += 1
      #print('fraction: '+str(fraction)+" mva_threshold: "+str(mva_threshold))
      if (iFraction == len(signal_fractions)): break
  return mva_thresholds

def evaluate_mva_significance(tmva_filename, tree_name, mass_name, mva_name, label_name, weight_name):
  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(tmva_filename)
  #luminosity = 137. + 110.
  luminosity = 1.
  hist = ROOT.TH1F("hist","hist",80,100,180)
  # Find mva thresholds
  signal_fractions = [0.95,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
  mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name)
  #print('mva_thresholds: ',mva_thresholds)
  # mva_signal_width[threshold] = signal_width
  mva_signal_widths = {}
  # Find width of signals with MVA threshold
  for mva_threshold in mva_thresholds:
    entries = tmva_chain.Draw(mass_name+">>hist",label_name+"==1&&"+mva_name+">"+str(mva_threshold)+"","goff")
    mva_signal_width = hist.GetStdDev()
    mva_signal_widths[mva_threshold] = mva_signal_width
    #print("mva threshold: "+str(mva_threshold)+" signal_width: "+str(mva_signal_width)+" entries: "+str(entries))
  # Find signal and background within 2 sigma of signal width
  significances = {}
  purities = {}
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
      significances[mva_threshold] = nevents_signal/math.sqrt(nevents_background)
    if nevents_background+nevents_signal !=0:
      print("  mva_threshold: "+str(mva_threshold)+" purity (s/(b+s)): "+str(nevents_signal/(nevents_background+nevents_signal)))
      purities[mva_threshold] = nevents_signal/(nevents_background+nevents_signal)
  # Find best significance
  best_mva = -999
  best_significance = -999
  for mva_threshold in mva_thresholds:
    if significances[mva_threshold] > best_significance:
      best_significance = significances[mva_threshold]
      best_mva = mva_threshold
  print('Best mva threshold: '+str(best_mva)+' significance: '+str(best_significance))

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

def mse_loss(output, target):
  print('output: '+str(output))
  print('target: '+str(target))
  loss = torch.mean((output - target)**2)
  return loss

class entropy_disco_loss(nn.Module):
  def __init__(self, disco_factor = 5.):
    super(entropy_disco_loss, self).__init__()
    self.bce_loss = nn.BCELoss()
    self.disco_factor = disco_factor

  def forward(self, output, target, mass):
    # output/target/mass = [[single-value], [single-value], ...]
    output_list = output.squeeze()
    mass_list = mass.squeeze()
    normedweight = torch.tensor([1.]*len(mass)).to(device)
    # TODO: Should ignore signal mass correlation
    # mask = (target.squeeze() == 0)
    # mass_list = mass_list[mask]
    # output_list = output_list[mask]
    # normedweight = normedweight[mask]
    disco = Disco.distance_corr(mass_list, output_list, normedweight)
    bce = self.bce_loss(output, target)
    loss = bce + self.disco_factor * disco
    #print(f'bce: {bce}, disco: {disco}, loss: {loss}')
    return loss

class entropy_bkg_disco_loss(nn.Module):
  def __init__(self, disco_factor = 5.):
    super(entropy_bkg_disco_loss, self).__init__()
    self.bce_loss = nn.BCELoss()
    self.disco_factor = disco_factor

  def forward(self, output, target, mass):
    # output/target/mass = [[single-value], [single-value], ...]
    output_list = output.squeeze()
    mass_list = mass.squeeze()
    normedweight = torch.tensor([1.]*len(mass)).to(device)
    # Ignore signal mass correlation
    mask = (target.squeeze() == 0)
    mass_list = mass_list[mask]
    output_list = output_list[mask]
    normedweight = normedweight[mask]
    #print(target.squeeze()[mask], mass_list)
    disco = Disco.distance_corr(mass_list, output_list, normedweight)
    bce = self.bce_loss(output, target)
    loss = bce + self.disco_factor * disco
    #print(f'bce: {bce}, disco: {disco}, loss: {loss}')
    return loss

class entropy_bkg_disco_signi_res_loss(nn.Module):
  def __init__(self, disco_factor = 5.):
    super(entropy_bkg_disco_signi_res_loss, self).__init__()
    self.bce_loss = nn.BCELoss()
    self.disco_factor = disco_factor

  def forward(self, output, target, weight, resolution, mass):
    # output/target/mass = [[single-value], [single-value], ...]
    output_list = output.squeeze()
    mass_list = mass.squeeze()
    normedweight = torch.tensor([1.]*len(mass)).to(device)
    # Ignore signal mass correlation
    mask = (target.squeeze() == 0)
    mass_list = mass_list[mask]
    output_list = output_list[mask]
    normedweight = normedweight[mask]
    #print(target.squeeze()[mask], mass_list)
    disco = Disco.distance_corr(mass_list, output_list, normedweight)
    bce = self.bce_loss(output, target)
    signal = torch.sum(output * target * weight)
    #print(f'resolution: {resolution}, output: {output.squeeze()}, target: {target.squeeze()}')
    avg_signal_res = torch.sum(output.squeeze() * target.squeeze() * resolution) / torch.sum(output.squeeze() * target.squeeze())
    bkg = torch.sum(output * (1-target) * weight)
    bkg_with_res = bkg * 2. * avg_signal_res
    signi_res = -torch.log(signal/torch.sqrt(signal+bkg_with_res))
    #print(f'signal: {signal} bkg: {bkg} avg_signal_res: {avg_signal_res} bkg_with_res: {bkg_with_res} signi_res: {signi_res}')
    #print(f'sum_res: {torch.sum(output * target * resolution)}, sum: {torch.sum(output * target)}')
    loss = bce + self.disco_factor * disco + signi_res / 30
    #print(f'bce: {bce}, disco: {disco}, signi_res: {signi_res}, loss: {loss}')
    return loss

class entropy_bkg_disco_signi_res2_loss(nn.Module):
  def __init__(self, disco_factor = 5.):
    super(entropy_bkg_disco_signi_res2_loss, self).__init__()
    self.bce_loss = nn.BCELoss()
    self.disco_factor = disco_factor

  def forward(self, output, target, weight, resolution, mass):
    # output/target/mass = [[single-value], [single-value], ...]
    output_list = output.squeeze()
    mass_list = mass.squeeze()
    normedweight = torch.tensor([1.]*len(mass)).to(device)
    # Ignore signal mass correlation
    mask = (target.squeeze() == 0)
    mass_list = mass_list[mask]
    output_list = output_list[mask]
    normedweight = normedweight[mask]
    #print(target.squeeze()[mask], mass_list)
    disco = Disco.distance_corr(mass_list, output_list, normedweight)
    bce = self.bce_loss(output, target)
    signal = torch.sum(output * target * weight)
    #print(f'resolution: {resolution}, output: {output.squeeze()}, target: {target.squeeze()}')
    avg_signal_res = torch.sum(output.squeeze() * target.squeeze() * resolution) / torch.sum(output.squeeze() * target.squeeze())
    bkg = torch.sum(output * (1-target) * weight)
    bkg_with_res = bkg * 2. * avg_signal_res
    signi_res = torch.sqrt(signal+bkg_with_res)/signal
    #print(f'signal: {signal} bkg: {bkg} avg_signal_res: {avg_signal_res} bkg_with_res: {bkg_with_res} signi_res: {signi_res}')
    #print(f'sum_res: {torch.sum(output * target * resolution)}, sum: {torch.sum(output * target)}')
    loss = bce + self.disco_factor * disco + signi_res / 1000
    #print(f'bce: {bce}, disco: {disco}, signi_res: {signi_res}, loss: {loss}')
    return loss

class entropy_bkg_disco_signi_res3_loss(nn.Module):
  # Try 20. to increase decorrelation
  def __init__(self, disco_factor = 10., signi_divide = 6000.):
    super(entropy_bkg_disco_signi_res3_loss, self).__init__()
    self.bce_loss = nn.BCELoss()
    self.disco_factor = disco_factor
    self.signi_divide = signi_divide

  def forward(self, output, target, weight, resolution, mass):
    # output/target/mass = [[single-value], [single-value], ...]
    output_list = output.squeeze()
    mass_list = mass.squeeze()
    normedweight = torch.tensor([1.]*len(mass)).to(device)
    # Ignore signal mass correlation
    mask = (target.squeeze() == 0)
    mass_list = mass_list[mask]
    output_list = output_list[mask]
    normedweight = normedweight[mask]
    #print(target.squeeze()[mask], mass_list)
    disco = Disco.distance_corr(mass_list, output_list, normedweight)
    bce = self.bce_loss(output, target)
    signal = torch.sum(output * target * weight)
    #print(f'resolution: {resolution}, output: {output.squeeze()}, target: {target.squeeze()}')
    avg_signal_res = torch.sum(output.squeeze() * target.squeeze() * resolution) / torch.sum(output.squeeze() * target.squeeze())
    bkg = torch.sum(output * (1-target) * weight)
    bkg_with_res = bkg * 2. * avg_signal_res
    signi_res = torch.sqrt(signal+bkg_with_res)/signal
    #print(f'signal: {signal} bkg: {bkg} avg_signal_res: {avg_signal_res} bkg_with_res: {bkg_with_res} signi_res: {signi_res}')
    #print(f'sum_res: {torch.sum(output * target * resolution)}, sum: {torch.sum(output * target)}')
    loss = bce + self.disco_factor * disco + signi_res / self.signi_divide
    #print(f'bce: {bce}, disco: {disco*self.disco_factor}, signi_res: {signi_res/self.signi_divide}, loss: {loss}')
    return loss

class significance_loss(nn.Module):
  def __init__(self, eps = 1e-7):
    super(significance_loss, self).__init__()
    self.eps = eps

  def forward(self, output, target, weight):
    signal = torch.sum(output * target * weight)
    bkg = torch.sum(output * (1-target) * weight)
    loss = (signal + bkg) / (signal * signal)
    #print(f'signal: {signal} bkg: {bkg} loss: {loss}')
    return loss

class significance_res_loss(nn.Module):
  def __init__(self, eps = 1e-7, res_weight = 1):
    super(significance_res_loss, self).__init__()
    self.eps = eps
    self.res_weight = res_weight


  def forward(self, output, target, weight, resolution):
    signal = torch.sum(output * target * weight)
    bkg = torch.sum(output * (1-target) * weight)
    avg_signal_res = torch.sum(output * target * resolution) / torch.sum(output * target)
    bkg_with_res = bkg * self.res_weight * avg_signal_res
    loss = (signal + bkg_with_res) / (signal * signal)
    org_loss = (signal + bkg) / (signal * signal)
    #print(f'signal: {signal} bkg: {bkg} avg_signal_res: {avg_signal_res} bkg_with_res: {bkg_with_res} org_loss: {org_loss} loss: {loss}')
    return loss

class z_loss(nn.Module):
  def __init__(self, eps = 1e-7):
    super(z_loss, self).__init__()
    self.eps = eps

  def forward(self, output, target, weight):
    signal = torch.sum(output * target * weight)
    bkg = torch.sum(output * (1-target) * weight)
    loss = 1./(2*((signal+bkg)*torch.log(1+(signal/bkg))-signal)) 
    #print(f'signal: {signal} bkg: {bkg} loss: {loss}')
    return loss

class purity_loss(nn.Module):
  def __init__(self, eps = 1e-7):
    super(purity_loss, self).__init__()
    self.eps = eps

  def forward(self, output, target, weight):
    signal = torch.sum(output * target * weight)
    bkg = torch.sum(output * (1-target) * weight)
    loss =  (signal + bkg) / signal
    #print(f'signal: {signal} bkg: {bkg} loss: {loss}')
    return loss
    

class RootDataset(torch.utils.data.Dataset):
  def __init__(self, root_filename, tree_name, features, cut, spectators, class_branch, entry_stop=None, entry_start=None, normalize=None, transform=None):
    self.root_file = uproot.open(root_filename)
    self.tree = self.root_file[tree_name]
    self.transform = transform
    feature_array = self.tree.arrays(features,
                                cut,
                                library='np')
    # feature_array = {feat1_name : [feat1_event0, feat1_event2, ...], }
    # self.feature_array = [ (feat1_event0, feat2_event0, ...),  ]
    self.feature_array = np.stack([feature_array[feat][0] for feat in features], axis=1)
    spec_array = self.tree.arrays(spectators,
                             cut,
                             library='np')
    # self.spec_array = [ (spec1_event0, spec2_event0, ...),  ]
    self.spec_array = np.stack([spec_array[spec][0] for spec in spectators], axis=1)
    # label_array = {classId: [1, 1, 0, ... ]}
    label_array = self.tree.arrays(class_branch,
                              cut,
                              library='np')
    # label_hotencoding = [ (0, 1), (0, 1), ... ]
    label_array = label_array['classID'][0]
    label_hotencoding = np.zeros((label_array.size, label_array.max()+1))
    label_hotencoding[np.arange(label_array.size), label_array] = 1
    self.label_array = np.array(label_hotencoding, dtype=int)
    # self.label_array = [ [label0_v1, label0_v2, ..], [label1_v1, label1_v2, ...] ]
    #self.label_array = np.array(list(zip(*label_hotencoding)))
    #print(label_hotencoding)
    # Combine labels into [bool, ...]
    # self.label_array = [ (0, 1), (0, 1), ... ]
    #self.label_array = np.stack([label_array[label] for label in labels], axis=1)
    #njets = self.feature_array.shape[0]
    #self.label_array = np.zeros((njets, 2))
    #self.label_array[:, 0] = label_array_all['sample_isQCD'] * (label_array_all['label_QCD_b'] +
    #                                                       label_array_all['label_QCD_bb'] +
    #                                                       label_array_all['label_QCD_c'] +
    #                                                       label_array_all['label_QCD_cc'] +
    #                                                       label_array_all['label_QCD_others'])
    #self.label_array[:, 1] = label_array_all['label_H_bb']

    # remove unlabeled data
    # np.sum(array, axis=1) => result[i] += array[i][j] over j
    # np.array[ [False, True, True] ] =>  [ value, value ]
    self.feature_array = self.feature_array[np.sum(self.label_array, axis=1) == 1]
    self.spec_array = self.spec_array[np.sum(self.label_array, axis=1) == 1]
    self.label_array = self.label_array[np.sum(self.label_array, axis=1) == 1]

    # convert [(bool, ...)] to [index of 1]
    # pytorch tensor method
    #self.label_array = (self.label_array == 1).nonzero(as_tuple=False)[:,1]
    # numpy array method
    #self.label_array = np.where(self.label_array == 1)[1]

    # normalize
    #print('org')
    #print(self.feature_array)
    if normalize:
      feat_min = np.amin(self.feature_array,0)
      feat_max = np.amax(self.feature_array,0)
      for ifeat, [min_x, max_x] in enumerate(normalize):
        #plt.figure()
        #plt.hist(self.feature_array[:,ifeat])
        #plt.savefig(f'plots/feat_{ifeat}.pdf')
        #print(f'Saved plots/feat_{ifeat}.pdf')
        #plt.close()
        print(f'[Info] ifeat[{ifeat}] data min: {feat_min[ifeat]} max: {feat_max[ifeat]} norm min: {min_x} max: {max_x}')
        self.feature_array[:,ifeat] = 2.*(self.feature_array[:,ifeat]-min_x)/(max_x-min_x) - 1.
        #self.feature_array[:,ifeat] = np.clip(self.feature_array[:,ifeat], -1, 1)
        #plt.figure()
        #plt.hist(self.feature_array[:,ifeat])
        #plt.savefig(f'plots/feat_{ifeat}_norm.pdf')
        #print(f'Saved plots/feat_{ifeat}_norm.pdf')
        #plt.close()
    #print('mod')
    #print(self.feature_array)

    # Split data
    if entry_stop and entry_stop:
      self.feature_array = self.feature_array[entry_start:entry_stop]
      self.spec_array = self.spec_array[entry_start:entry_stop]
      self.label_array = self.label_array[entry_start:entry_stop]
    elif entry_stop:
      self.feature_array = self.feature_array[:entry_stop]
      self.spec_array = self.spec_array[:entry_stop]
      self.label_array = self.label_array[:entry_stop]
    elif entry_start:
      self.feature_array = self.feature_array[entry_start:]
      self.spec_array = self.spec_array[entry_start:]
      self.label_array = self.label_array[entry_start:]

  def __len__(self):
    return len(self.label_array)

  def __getitem__(self, idx):
    #sample = {'feature_array': self.feature_array[idx], 'label_array': self.label_array[idx], 'spec_array': self.spec_array[idx]}
    sample = [self.feature_array[idx], self.label_array[idx], self.spec_array[idx]]
    if self.transform:
      sample = self.transform(sample)
    return sample

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        #torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
        #torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        layer1_out = self.tanh(hidden)
        layer2 = self.fc2(layer1_out)
        layer2_out = self.sigmoid(layer2)
        return layer2_out

def train(dataloader, model, loss_fn, optimizer, use_weight_in_loss = False, use_res_in_loss = False, use_mass_in_loss = False):
    size = len(dataloader.dataset)
    model.train()
    avg_loss = 0.
    for batch, (feature, label, spec) in enumerate(dataloader):
        # feature: [nbatch, nfeature], no_device
        # label: [nbatch, nlabel], no_device
        # spec: [nbatch, nspec], no_device

        # torch.max returns ([max value], [index of max])
        X, y = feature.to(device), torch.max(label,1)[1].to(device)
        #X, y = feature.to(device), label.to(device)
        # X: [batch, nfeature], device
        # y: [batch], device

        #print(f'pre-y max: {torch.max(label,1)}')

        # Forward pass
        # Compute prediction error
        pred = model(X)
        # pred: [batch, 1], device
        #print('out:', y.unsqueeze(1))
        #print('feature:', feature)
        #print('label:', label)
        #print('res:', X[:,5], (X[:,5]+1)*(normalize_max_min[5][1]-normalize_max_min[5][0])/2+normalize_max_min[5][0])
        #print('y:', y)
        #print('pred:', pred)
        #print(f'y: {y}, size: {y.size()}')
        #print('pred squeeze: ',pred.squeeze())
        #loss = loss_fn(pred, y)

        loss = calculate_loss(X, y, pred, spec, loss_fn, use_weight_in_loss, use_res_in_loss, use_mass_in_loss)
        #if use_weight_in_loss == False: 
        #  loss = loss_fn(pred.to(torch.float32), y.unsqueeze(1).to(torch.float32))
        #else:
        #  weight = spec[:,1].to(device).unsqueeze(1).type(torch.float32)
        #  if use_res_in_loss == False:
        #    loss = loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32))
        #  else:
        #    resolution = (X[:,5]+1)*(normalize_max_min[5][1]-normalize_max_min[5][0])/2+normalize_max_min[5][0]
        #    loss = loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32), resolution.to(torch.float32))
        ##pred_binary = pred.squeeze().to(torch.float32)
        ##y_binary = y.to(torch.float32)
        ##loss = loss_fn(pred_binary, y_binary)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() * len(X) # loss is average over batch
        #loss_batch, current = loss.item(), batch * len(X)
        #print(f"loss: {loss_batch:>7f}  [{current:>5d}/{size:>5d}]")
        #if batch % int((size-1)*1./2) == 0:
        #    loss_batch, current = loss.item(), batch * len(X)
        #    print(f"loss for batch: {loss_batch:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = avg_loss / size
    print(f'avg loss: {avg_loss:>7f}')
    return {'loss': avg_loss}

def test(dataloader, model, loss_fn, use_weight_in_loss = None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    x_array, y_array, pred_array, mass_array, w_lumi_array = [], [], [], [], []
    iEntry = 0
    with torch.no_grad():
        for feature, label, spec in dataloader:
            X, y = feature.to(device), torch.max(label,1)[1].to(device)
            pred = model(X)
            x_array.extend(X.numpy())
            y_array.extend(y.numpy())
            if (pred.squeeze().shape==torch.Size([])): pred_array.append(pred.squeeze().numpy())
            else: pred_array.extend(pred.squeeze().numpy())
            mass_array.extend(spec[:,0])
            w_lumi_array.extend(spec[:,1])
            #print('y', y.unsqueeze(1))
            #print('feature', feature)
            #print('pred', pred)
            #print('y', y.unsqueeze(1).type(torch.float))
            #print('pred', (pred>0.7).squeeze())
            # Remove entries below threshold
            #print(pred[(pred>threshold).squeeze()])
            #print(y[(pred>threshold).squeeze()])
            #print(pred>0.7)
            #print(pred[pred>0.7])
            #test_loss += loss_fn(pred, y).item()
            if use_weight_in_loss == False: test_loss += loss_fn(pred, y.unsqueeze(1).type(torch.float)).item()
            else: 
              weight = spec[:,1].to(device).unsqueeze(1).type(torch.float)
              #print('weight', weight)
              test_loss += loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32)).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            #correct += ((pred > 0.8) == y).type(torch.float).sum().item()
            iEntry += 1
            #if iEntry == 3: break
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    pred_array = np.array(pred_array)
    #root_file = uproot.recreate("trash/test.root")
    #root_file["tree"] = {'x': x_array, 'y': y_array, 'pred': pred_array, 'mass': mass_array, 'w_lumi': w_lumi_array}
    #evaluate_mva_significance('trash/test.root', 'tree', 'mass', 'pred', 'y', 'w_lumi')
    #evaluate_mva_correlation('trash/test.root', 'tree', 'mass', 'pred', 'y')

    test_loss /= num_batches
    print(f"Test loss: {test_loss:>8f} \n")
    return {'loss': test_loss}
    #correct /= size
    #print(nCorrect, nTotal, correct)
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def calculate_loss(X, y, pred, spec, loss_fn, use_weight_in_loss, use_res_in_loss, use_mass_in_loss):
  if use_weight_in_loss == False and use_res_in_loss == False and use_mass_in_loss == False:
    loss = loss_fn(pred.to(torch.float32), y.unsqueeze(1).to(torch.float32))
  elif use_weight_in_loss == False and use_res_in_loss == False and use_mass_in_loss == True:
    mass = spec[:,0].to(device).unsqueeze(1).type(torch.float32)
    loss = loss_fn(pred.to(torch.float32), y.unsqueeze(1).to(torch.float32), mass)
  elif use_weight_in_loss == False and use_res_in_loss == True and use_mass_in_loss == False:
    resolution = (X[:,5]+1)*(normalize_max_min[5][1]-normalize_max_min[5][0])/2+normalize_max_min[5][0]
    loss = loss_fn(pred, y.unsqueeze(1).type(torch.float), resolution.to(torch.float32))
  elif use_weight_in_loss == False and use_res_in_loss == True and use_mass_in_loss == True:
    mass = spec[:,0].to(device).unsqueeze(1).type(torch.float32)
    resolution = (X[:,5]+1)*(normalize_max_min[5][1]-normalize_max_min[5][0])/2+normalize_max_min[5][0]
    loss = loss_fn(pred.to(torch.float32), y.unsqueeze(1).to(torch.float32), resolution.to(torch.float32), mass.to(torch.float32))
  elif use_weight_in_loss == True and use_res_in_loss == False and use_mass_in_loss == False:
    weight = spec[:,1].to(device).unsqueeze(1).type(torch.float32)
    loss = loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32))
  elif use_weight_in_loss == True and use_res_in_loss == False and use_mass_in_loss == True:
    weight = spec[:,1].to(device).unsqueeze(1).type(torch.float32)
    mass = spec[:,0].to(device).unsqueeze(1).type(torch.float32)
    loss = loss_fn(pred.to(torch.float32), y.unsqueeze(1).to(torch.float32), weight.to(torch.float32), mass.to(torch.float32))
  elif use_weight_in_loss == True and use_res_in_loss == True and use_mass_in_loss == False:
    weight = spec[:,1].to(device).unsqueeze(1).type(torch.float32)
    resolution = (X[:,5]+1)*(normalize_max_min[5][1]-normalize_max_min[5][0])/2+normalize_max_min[5][0]
    loss = loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32), resolution.to(torch.float32))
  elif use_weight_in_loss == True and use_res_in_loss == True and use_mass_in_loss == True:
    weight = spec[:,1].to(device).unsqueeze(1).type(torch.float32)
    resolution = (X[:,5]+1)*(normalize_max_min[5][1]-normalize_max_min[5][0])/2+normalize_max_min[5][0]
    mass = spec[:,0].to(device).unsqueeze(1).type(torch.float32)
    loss = loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32), resolution.to(torch.float32), mass.to(torch.float32))
  return loss

def evaluate_sample(dataloader, device, model, loss_fn, use_weight_in_loss = False, use_res_in_loss = False, use_mass_in_loss=False):
    nBatches = len(dataloader)
    model.eval()
    loss = 0.
    # Collect information for evaluation
    x_array, y_array, pred_array, mass_array, w_lumi_array = [], [], [], [], []
    with torch.no_grad():
      # Evaluate train sample
      for feature, label, spec in dataloader:
        X, y = feature.to(device), torch.max(label,1)[1].to(device)
        pred = model(X)
        loss += calculate_loss(X, y, pred, spec, loss_fn, use_weight_in_loss, use_res_in_loss, use_mass_in_loss).item()
        #if use_weight_in_loss == False: 
        #  loss += loss_fn(pred, y.unsqueeze(1).type(torch.float)).item()
        #else: 
        #  weight = spec[:,1].to(device).unsqueeze(1).type(torch.float)
        #  if use_res_in_loss == False:
        #    loss += loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32)).item()
        #  else:
        #    resolution = (X[:,5]+1)*(normalize_max_min[5][1]-normalize_max_min[5][0])/2+normalize_max_min[5][0]
        #    loss += loss_fn(pred, y.unsqueeze(1).type(torch.float), weight.to(torch.float32), resolution.to(torch.float32)).item()
        # Collect information for evaluation
        x_array.extend(X.cpu().numpy())
        y_array.extend(y.cpu().numpy())
        if (pred.squeeze().shape==torch.Size([])): pred_array.append(pred.squeeze().numpy())
        else: pred_array.extend(pred.squeeze().cpu().numpy())
        mass_array.extend(spec[:,0])
        w_lumi_array.extend(spec[:,1])
    # Post batch loop operations
    loss /= nBatches
    # Save results
    results = {}
    results['loss'] = loss
    results['x'] = x_array
    results['y'] = y_array
    results['yhat'] = pred_array
    results['mass'] = mass_array
    results['weight'] = w_lumi_array
    return results

def load_tmva_model(weight_xml, model, min_max_array):
  print("Loading tmva model from :"+weight_xml)
  # Load TMVA model
  tmva_weights_tree = ET.parse(weight_xml)
  tmva_weights = tmva_weights_tree.getroot()
  # Load min max of input variables
  for ranges in tmva_weights.findall(".//*[@ClassIndex='2']/Ranges/Range"):
    min_max_array.append([float(ranges.get('Min')), float(ranges.get('Max'))])
  #print(min_max_array)
  # Load weights of network
  with torch.no_grad():
    nLayer = 0
    for tmva_weight in tmva_weights.findall("./Weights"):
      for denseLayer in tmva_weight:
        #print('weight row: '+str(denseLayer.findall("./Weights")[0].get("Rows")))
        #print('weight column: '+str(denseLayer.findall("./Weights")[0].get("Columns")))
        #print('bias row: '+str(denseLayer.findall("./Biases")[0].get("Rows")))
        #print('bias column: '+str(denseLayer.findall("./Biases")[0].get("Columns")))
        weights = denseLayer.findall("./Weights")[0].text.split()
        nRow_weights = int(denseLayer.findall("./Weights")[0].get("Rows"))
        nColumn_weights = int(denseLayer.findall("./Weights")[0].get("Columns"))
        for iRow in range(nRow_weights):
          for iColumn in range(nColumn_weights):
            #print(weights[iColumn*nRow_weights+iRow])
            if nLayer == 0:
              #model.fc1.weight[iRow][iColumn] = float(weights[iColumn*nRow_weights+iRow])
              model.fc1.weight[iRow][iColumn] = float(weights[iColumn+iRow*nColumn_weights])
            else:
              #model.fc2.weight[iRow][iColumn] = float(weights[iColumn*nRow_weights+iRow])
              model.fc2.weight[iRow][iColumn] = float(weights[iColumn+iRow*nColumn_weights])
              #model.fc2.weight[1][iColumn] = -1*float(weights[iColumn+iRow*nColumn_weights])
        biases = denseLayer.findall("./Biases")[0].text.split()
        nRow_biases = int(denseLayer.findall("./Biases")[0].get("Rows"))
        nColumn_biases = int(denseLayer.findall("./Biases")[0].get("Columns"))
        for iRow in range(nRow_biases):
         #print(biases[iRow])
         if nLayer == 0:
           model.fc1.bias[iRow] = float(biases[iRow])
         else:
           model.fc2.bias[iRow] = float(biases[iRow])
           #model.fc2.bias[1] = -1*float(biases[iRow])
        #if nLayer == 1:
        #  print(model.fc2.weight)
        #  print(model.fc2.bias)
        nLayer += 1

def find_nearest(array,value):
  idx = (np.abs(array-value)).argmin()
  return idx, array[idx]

if __name__ == "__main__":
  ROOT.EnableImplicitMT()
  # 1: Use tmva training, 2: Use previous nn training
  do_fine_tune = 0
  model_filename = 'runs/May20_01-59-34_hepmacprojb.local/model_epoch_4990.pt'
  #batch_size = 128
  #batch_size = 4096
  batch_size = 8192
  #batch_size = 16384 # crashes with memory issue

  epochs = 10 * batch_size # Keeps number of updates on model constant
  eval_epoch = 100

  #epochs = 100
  #eval_epoch = 10

  #device = "cpu"
  device = "cuda"

  train_filename = 'train_sample_run2_winfull.root'
  test_filename = 'test_sample_run2_winfull.root'
  test_full_filename = 'test_full_sample_run2_winfull.root'
  log_dir = None
  output_name = 'torch_nn_winfull_decorH'

  #train_filename = 'train_sample_run2.root'
  #test_filename = 'test_sample_run2.root'
  #test_full_filename = 'test_full_sample_run2.root'
  #log_dir = None
  #output_name = 'torch_nn'

  #train_filename = 'train_sample_run2_0p05.root'
  #test_filename = 'test_sample_run2_0p05.root'
  #test_full_filename = 'test_full_sample_run2_0p05.root'

  do_test = True


  #batch_size = 1
  # train_loss
  # 0: cross-entropy, 
  # 1: s/sqrt(s+b), 2: Z, 3: purity
  # 100: s/sqrt(s+b*res)
  # 200: cross-entropy + disco
  # 201: cross-entropy + disco (only on bkg)
  # 202: cross-entropy + disco (only on bkg) + -ln(s/sqrt(s+b*res))
  # 203: cross-entropy + 5*disco (only on bkg) + 1/1000*sqrt(s+b*res)/s
  # 204: (tune) cross-entropy + 10*disco (only on bkg) + 1/6000*sqrt(s+b*res)/s
  # 205: (tune) cross-entropy + 20*disco (only on bkg) + 1/6000*sqrt(s+b*res)/s
  # 206: (tune) cross-entropy + 40*disco (only on bkg) + 1/6000*sqrt(s+b*res)/s
  train_loss = 206
  #log_dir = f'runs/loss{train_loss}'
  #output_name = f'nn_{log_dir}'.replace('/','_')

  use_weight_in_loss = False
  use_res_in_loss = False
  use_mass_in_loss = False
  if train_loss == 0: 
    loss_fn = nn.BCELoss()
    loss_filename = ''
  elif train_loss == 1: 
    loss_fn = significance_loss()
    use_weight_in_loss = True
    loss_filename = '_signi_loss'
  elif train_loss == 2: 
    loss_fn = z_loss()
    use_weight_in_loss = True
    loss_filename = '_z_loss'
  elif train_loss == 3: 
    loss_fn = purity_loss()
    use_weight_in_loss = True
    loss_filename = '_purity_loss'
  elif train_loss == 100: 
    loss_fn = significance_res_loss()
    use_weight_in_loss = True
    use_res_in_loss = True
    loss_filename = '_signi_res_loss'
  elif train_loss == 200: 
    loss_fn = entropy_disco_loss()
    use_mass_in_loss = True
    loss_filename = '_entropy_disco'
  elif train_loss == 201: 
    loss_fn = entropy_bkg_disco_loss()
    use_mass_in_loss = True
    loss_filename = '_entropy_bkg_disco'
  elif train_loss == 202: 
    loss_fn = entropy_bkg_disco_signi_res_loss()
    use_mass_in_loss = True
    use_res_in_loss = True
    use_weight_in_loss = True
    loss_filename = '_entropy_bkg_disco_signi_res_loss'
  elif train_loss == 203: 
    loss_fn = entropy_bkg_disco_signi_res2_loss()
    use_mass_in_loss = True
    use_res_in_loss = True
    use_weight_in_loss = True
    loss_filename = '_entropy_bkg_disco_signi_res2_loss'
  elif train_loss == 204: 
    loss_fn = entropy_bkg_disco_signi_res3_loss()
    use_mass_in_loss = True
    use_res_in_loss = True
    use_weight_in_loss = True
    loss_filename = '_entropy_bkg_disco_signi_res3_loss'
  elif train_loss == 205: 
    loss_fn = entropy_bkg_disco_signi_res3_loss(disco_factor=20.)
    use_mass_in_loss = True
    use_res_in_loss = True
    use_weight_in_loss = True
    loss_filename = '_entropy_bkg_disco_signi_loss205'
  elif train_loss == 206: 
    loss_fn = entropy_bkg_disco_signi_res3_loss(disco_factor=40.)
    use_mass_in_loss = True
    use_res_in_loss = True
    use_weight_in_loss = True
    loss_filename = '_entropy_bkg_disco_signi_loss205'


  if do_test == False:
    writer = SummaryWriter(log_dir=log_dir)
    writer_foldername = writer.get_logdir()

  print(f"Using {device} device")
  torch.manual_seed(1)
  model = SimpleNetwork(input_size=11, hidden_size=44, output_size=1).to(device)
  min_max_array = []
  if do_fine_tune == 1:
    load_tmva_model('dataset/weights/TMVAClassification_DNN.weights.xml', model, min_max_array)
  if do_fine_tune == 2:
    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict)

  feature_names = ['photon_mva', 'min_dR', 'pt_mass', 'cosTheta', 'costheta', 
                   #'photon_res', 
                   'llg_mass_err',
                   'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
                   'llg_flavor', 'gamma_pt']
  normalize_max_min = [[-0.57861328125,0.98583984375],
                      [0.400207489729,3.32512640953],
                      [0.000612989999354,4.14180803299],
                      [-0.999573588371,0.998835206032],
                      [-0.987939178944,0.983025610447],
                      #[0.00963300466537,1.51448833942], # photon_res
                      [0.53313,16.254], # llg_mass_err
                      [-2.49267578125,2.4921875],
                      [-2.49072265625,2.4814453125],
                      [-2.49072265625,2.50830078125],
                      [1., 2.],
                      [15.015657, 295.22623]]

  #model.eval()
  #with torch.no_grad():
  #  print(model(torch.from_numpy(test_dataset.feature_array[0])))
  #  print(model(torch.from_numpy(test_dataset.feature_array[1])))
  #  print(model(torch.from_numpy(test_dataset.feature_array[2])))

  #train_dataset = RootDataset(root_filename='ntuples_mva/TMVA_nn.root',
  #                          tree_name = "dataset/TrainTree",
  #                          features = feature_names,
  #                          normalize = normalize_max_min,
  #                          cut = '1',
  #                          spectators = ['llg_mass', 'w_lumi'],
  #                          class_branch = ['classID'])
  train_dataset = RootDataset(root_filename= train_filename,
                            tree_name = "train_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumi'],
                            class_branch = ['classID'])
  print(f'train entries: {len(train_dataset)}')

  #test_dataset = RootDataset(root_filename='ntuples_mva/TMVA_nn.root',
  #                          tree_name = "dataset/TestTree",
  #                          features = feature_names,
  #                          normalize = normalize_max_min,
  #                          cut = '1',
  #                          spectators = ['llg_mass', 'w_lumi'],
  #                          class_branch = ['classID'], 
  #                          entry_stop = len(train_dataset))
  test_dataset = RootDataset(root_filename= test_filename,
                            tree_name = "test_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumi'],
                            class_branch = ['classID'], 
                            entry_stop = len(train_dataset)
                            )
  print(f'test entries: {len(test_dataset)}')

  #eval_dataset = RootDataset(root_filename='ntuples_mva/TMVA_nn.root',
  #                          tree_name = "dataset/TestTree",
  #                          features = feature_names,
  #                          normalize = normalize_max_min,
  #                          cut = '1',
  #                          spectators = ['llg_mass', 'w_lumi'],
  #                          class_branch = ['classID'])
  eval_dataset = RootDataset(root_filename=test_full_filename,
                            tree_name = "test_tree",
                            features = feature_names,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass', 'w_lumi'],
                            class_branch = ['classID'])
  print(f'eval entries: {len(eval_dataset)}')
  
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  if do_fine_tune:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


  start_time = time.time()
  #epochs = 50
  #test(train_dataloader, model, loss_fn, train_loss!=0)
  if do_test: epochs = 1
  for iEpoch in range(epochs):
    if iEpoch != 0 or do_test:
      print(f"Epoch {iEpoch+1}\n-------------------------------")
      train(train_dataloader, model, loss_fn, optimizer, use_weight_in_loss, use_res_in_loss, use_mass_in_loss)
    if iEpoch % eval_epoch and not do_test == 0:
      # Evaluate
      results = {'train': {}, 'test': {}}
      filename = 'trash/evaluate.root'
      # Returns loss, sample arrays
      results['train'] = evaluate_sample(train_dataloader, device, model, loss_fn, use_weight_in_loss, use_res_in_loss, use_mass_in_loss)
      results['test'] = evaluate_sample(test_dataloader, device, model, loss_fn, use_weight_in_loss, use_res_in_loss, use_mass_in_loss)
      # Create evaluation root file
      with uproot.recreate(filename) as root_file:
        root_file['train_tree'] = {'x': results['train']['x'], 'y': results['train']['y'], 'yhat': results['train']['yhat'], 'mass': results['train']['mass'], 'weight': results['train']['weight']}
        root_file['test_tree'] = {'x': results['test']['x'], 'y': results['test']['y'], 'yhat': results['test']['yhat'], 'mass': results['test']['mass'], 'weight': results['test']['weight']}
      mva_dict = [evaluate.load_mva_dict(filename, 'mva')]
      # Evaluate train
      #train_significances, train_purities = evaluate.evaluate_significance_with_resolution(mva_dict, draw=False, tree_type='train_tree')
      train_significances, train_purities = evaluate.evaluate_significance(mva_dict, draw=False, tree_type='train_tree')
      train_significances_with_res, train_purities_with_res = evaluate.evaluate_significance_with_resolution(mva_dict, draw=False, tree_type='train_tree')
      train_std_divs = evaluate.evaluate_correlation(mva_dict, draw=False, tree_type='train_tree')
      results['train']['significance'] = train_significances[0]
      results['train']['purity'] = train_purities[0]
      results['train']['significance_res'] = train_significances_with_res[0]
      results['train']['purity_res'] = train_purities_with_res[0]
      results['train']['observable_std_div'] = train_std_divs[0]
      # Evaluate test
      #test_significances, test_purities = evaluate.evaluate_significance_with_resolution(mva_dict, draw=False, tree_type='test_tree')
      test_significances, test_purities = evaluate.evaluate_significance(mva_dict, draw=False, tree_type='test_tree')
      test_significances_with_res, test_purities_with_res = evaluate.evaluate_significance_with_resolution(mva_dict, draw=False, tree_type='test_tree')
      test_std_divs = evaluate.evaluate_correlation(mva_dict, draw=False, tree_type='test_tree')
      results['test']['significance'] = test_significances[0]
      results['test']['purity'] = test_purities[0]
      results['test']['significance_res'] = test_significances_with_res[0]
      results['test']['purity_res'] = test_purities_with_res[0]
      results['test']['observable_std_div'] = test_std_divs[0]
      # Log evaluation results
      print(f"loss: train: {results['train']['loss']:.6f}, test: {results['test']['loss']:.6f}")
      print(f"significance: train: {results['train']['significance']:.6f}, test: {results['test']['significance']:.6f}")
      print(f"purity: train: {results['train']['purity']:.6f}, test: {results['test']['purity']:.6f}")
      print(f"significance with res.: train: {results['train']['significance_res']:.6f}, test: {results['test']['significance_res']:.6f}")
      print(f"purity with res.: train: {results['train']['purity_res']:.6f}, test: {results['test']['purity_res']:.6f}")
      print(f"observable std div: train: {results['train']['observable_std_div']:.6f}, test: {results['test']['observable_std_div']:.6f}")
      if do_test == False:
        writer.add_scalars('Loss', {'train': results['train']['loss'], 'test': results['test']['loss']}, iEpoch)
        writer.add_scalars('Significance', {'train': results['train']['significance'], 'test': results['test']['significance']}, iEpoch)
        writer.add_scalars('Purity', {'train': results['train']['purity'], 'test': results['test']['purity']}, iEpoch)
        writer.add_scalars('Significance with res.', {'train': results['train']['significance_res'], 'test': results['test']['significance_res']}, iEpoch)
        writer.add_scalars('Purity with res.', {'train': results['train']['purity_res'], 'test': results['test']['purity_res']}, iEpoch)
        writer.add_scalars('Observable std. div.', {'train': results['train']['observable_std_div'], 'test': results['test']['observable_std_div']}, iEpoch)
      # Save model
      if do_test == False:
        model_filename = writer_foldername+f'/model_epoch_{iEpoch}.pt'
        torch.save(model.state_dict(), model_filename)

  elapsed_time = time.time() - start_time
  print(f'Training time: {elapsed_time}')

  # Save train 
  train_feature_array = train_dataset.feature_array
  train_hot_label_array = train_dataset.label_array
  train_label_array = train_dataset.label_array[:,1]
  train_spec_array = train_dataset.spec_array
  train_unnorm_feature_array = unnormalize(train_feature_array, normalize_max_min)
  train_mass_array = train_dataset.spec_array[:,0]
  train_weight_array = train_dataset.spec_array[:,1]
  nlabels = train_hot_label_array.shape[1]

  model.eval()
  with torch.no_grad():
    train_predict_array_nn_raw = model(torch.from_numpy(train_feature_array).to(device)).to('cpu')
    train_predict_array_nn = train_predict_array_nn_raw.squeeze()
  print(f'nn label: {train_hot_label_array[:,nlabels-1]} predict: {train_predict_array_nn}')

  # Save test
  test_feature_array = test_dataset.feature_array
  test_hot_label_array = test_dataset.label_array
  test_label_array = test_dataset.label_array[:,1]
  test_spec_array = test_dataset.spec_array
  test_unnorm_feature_array = unnormalize(test_feature_array, normalize_max_min)
  test_mass_array = test_dataset.spec_array[:,0]
  test_weight_array = test_dataset.spec_array[:,1]
  #test_random_idx = np.random.permutation(len(test_hot_label_array))
  #test_feature_array = test_feature_array[test_random_idx]

  model.eval()
  with torch.no_grad():
    test_predict_array_nn_raw = model(torch.from_numpy(test_feature_array).to(device)).to('cpu')
    test_predict_array_nn = test_predict_array_nn_raw.squeeze()
  print(f'nn label: {test_hot_label_array[:,nlabels-1]} predict: {test_predict_array_nn}')

  # Save test_full
  eval_feature_array = eval_dataset.feature_array
  eval_hot_label_array = eval_dataset.label_array
  eval_label_array = eval_dataset.label_array[:,1]
  eval_spec_array = eval_dataset.spec_array
  eval_unnorm_feature_array = unnormalize(eval_feature_array, normalize_max_min)
  eval_mass_array = eval_dataset.spec_array[:,0]
  eval_weight_array = eval_dataset.spec_array[:,1]

  model.eval()
  with torch.no_grad():
    eval_predict_array_nn_raw = model(torch.from_numpy(eval_feature_array).to(device)).to('cpu')
    eval_predict_array_nn = eval_predict_array_nn_raw.squeeze()
  print(f'nn label: {eval_hot_label_array[:,nlabels-1]} predict: {eval_predict_array_nn}')




  if do_test == False: 
    filename = 'ntuples_mva/'+output_name
    #if do_fine_tune == 1: filename += 'fine_nn'
    #elif do_fine_tune == 2: filename += 'torch_fine_nn' 
    #else: filename += 'torch_nn'
    #filename += loss_filename
    #if batch_size != 1: filename +='_batch'+str(batch_size)
    filename += '.root'
    root_file = uproot.recreate(filename)
    root_file["test_tree"] = {'x_norm': test_feature_array, 'x': test_unnorm_feature_array, 'y': test_label_array, 'yhat': test_predict_array_nn, 'mass': test_mass_array, 'weight': test_weight_array}
    root_file["train_tree"] = {'x_norm': train_feature_array, 'x': train_unnorm_feature_array, 'y': train_label_array, 'yhat': train_predict_array_nn, 'mass': train_mass_array, 'weight': train_weight_array}
    root_file["test_full_tree"] = {'x_norm': eval_feature_array, 'x': eval_unnorm_feature_array, 'y': eval_label_array, 'yhat': eval_predict_array_nn, 'mass': eval_mass_array, 'weight': eval_weight_array}
    print('Results saved to '+filename)
    writer.close()

  #for iEvent, feats in enumerate(test_feature_array):
  #  print('feat: '+','.join(str(x) for x in raw_test_feature_array[iEvent]))
  #  print('normalized feat: '+','.join(str(x) for x in test_feature_array[iEvent]))
  #  print('label: '+str(test_label_array[iEvent]))
  #  print('predict nn: '+str(test_predict_array_nn[iEvent]))
  #  print('predict bdt: '+str(test_predict_array_btree[iEvent]))
  #  print('predict xgboost: '+str(test_predict_array_bdt[iEvent]))
  #  if (iEvent==10): break

  #fpr_nn, tpr_nn, threshold_nn = sklearn.metrics.roc_curve(test_label_array[:,nlabels-1], test_predict_array_nn)
  #fpr_btree, tpr_btree, threshold_btree = sklearn.metrics.roc_curve(test_label_array[:,nlabels-1], test_predict_array_btree)
  #fpr_bdt, tpr_bdt, threshold_bdt = sklearn.metrics.roc_curve(test_label_array[:,nlabels-1], test_predict_array_bdt)
  ## ROC
  #plt.figure()
  #plt.plot(tpr_btree, fpr_btree, lw=2.5, label="Boosted tree, AUC = {:.1f}%".format(sklearn.metrics.auc(fpr_btree, tpr_btree)*100))
  #plt.plot(tpr_bdt, fpr_bdt, lw=2.5, label="XGBoost, AUC = {:.1f}%".format(sklearn.metrics.auc(fpr_bdt, tpr_bdt)*100))
  #plt.plot(tpr_nn, fpr_nn, lw=2.5, label="NN, AUC = {:.1f}%".format(sklearn.metrics.auc(fpr_nn, tpr_nn)*100))
  #plt.xlabel(r'True positive rate')
  #plt.ylabel(r'False positive rate')
  ##plt.semilogy()
  #plt.ylim(0.001,1)
  #plt.xlim(0,1)
  #plt.grid(True)
  #plt.legend(loc='upper left')
  #plt.savefig("plots/roc_higgsToZGamma_classifiers.pdf")
  #print("Saved to plots/roc_higgsToZGamma_classifiers.pdf")
