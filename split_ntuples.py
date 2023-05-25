#!/usr/bin/env python3
import uproot
import numpy as np

def split_ntuple(in_filename, in_treename, branches, cut, train_fraction, weight_branch_index, out_filename):
  # Read the ntuples
  in_file = uproot.open(in_filename)
  in_tree = in_file[in_treename]
  in_ntuples = in_tree.arrays(branches, cut, library='np')
  ntuples = np.stack([np.squeeze(in_ntuples[feat]) for feat in branches], axis=1)

  # Randomize ntuple
  random_idx = np.random.permutation(len(ntuples))
  ntuples = ntuples[random_idx]

  # Split the ntuples
  n_entries = len(ntuples)
  train_n_entries = int(train_fraction * n_entries)
  train_ntuples = ntuples[:train_n_entries]
  test_ntuples = ntuples[train_n_entries:]
  print(f'Total entries: {n_entries} train entries: {len(train_ntuples)}, test entries: {len(test_ntuples)}')
  # Modify the ntuples. Scale up weights
  train_ntuples[:,weight_branch_index] = train_ntuples[:,weight_branch_index] / train_fraction
  test_ntuples[:,weight_branch_index] = test_ntuples[:,weight_branch_index] / (1-train_fraction)
  # Create the ntuples
  out_train_ntuples = {}
  out_test_ntuples = {}
  for ibranch, branch in enumerate(branches):
    out_train_ntuples[branch] = train_ntuples[:,ibranch]
    out_test_ntuples[branch] = test_ntuples[:,ibranch]
  # Create file
  with uproot.recreate(out_filename) as out_file:
    out_file["train_tree"] = out_train_ntuples
    out_file["test_tree"] = out_test_ntuples
  print('tree saved to '+out_filename)
  in_file.close()
  
def combine_signal_bkg(bkg_filename, bkg_treename,
                       signal_filename, signal_treename,
                       cut,
                       out_filename, out_treename):
  bkg_file = uproot.open(bkg_filename)
  bkg_tree = bkg_file[bkg_treename]
  bkg_ntuples = bkg_tree.arrays(branches, cut, library='np')

  signal_file = uproot.open(signal_filename)
  signal_tree = signal_file[signal_treename]
  signal_ntuples = signal_tree.arrays(branches, cut, library='np')

  new_branches = [branch for branch in branches]
  out_ntuples = {}
  for branch in branches:
    out_ntuples[branch] = np.float32(np.concatenate((np.squeeze(bkg_ntuples[branch]), np.squeeze(signal_ntuples[branch])),0))
  # Add branch indicating signal or background
  out_ntuples['classID'] = np.int32(np.concatenate((np.array([0]*len(np.squeeze(bkg_ntuples[branches[0]]))), np.array([1]*len(np.squeeze(signal_ntuples[branches[0]])))), 0))
  new_branches.append('classID')

  # Randomize ntuple
  random_idx = np.random.permutation(len(out_ntuples['classID']))
  for branch in new_branches:
    out_ntuples[branch] = out_ntuples[branch][random_idx]

  #ntuples = np.stack([np.squeeze(out_ntuples[feat]) for feat in branches], axis=1)
  #print(ntuples)
  
  with uproot.recreate(out_filename) as out_file:
    out_file[out_treename] = out_ntuples
  print('tree saved to '+out_filename)

  bkg_file.close()
  signal_file.close()

if __name__ == '__main__':
  input_signal_filename = 'ntuples/train_decorr_sig_shapewgt_run2.root'
  input_bkg_filename = 'ntuples/train_decorr_bak_shapewgt_run2.root'

  #split_signal_filename = 'signal_sample_run2.root'
  #split_bkg_filename = 'bkg_sample_run2.root'
  #train_sample_filename = 'train_sample_run2.root'
  #test_sample_filename = 'test_sample_run2.root'
  #test_full_sample_filename = 'test_full_sample_run2.root'

  #split_signal_filename = 'signal_sample_run2_0p05.root'
  #split_bkg_filename = 'bkg_sample_run2_0p05.root'
  #train_sample_filename = 'train_sample_run2_0p05.root'
  #test_sample_filename = 'test_sample_run2_0p05.root'
  #test_full_sample_filename = 'test_full_sample_run2_0p05.root'

  split_signal_filename = 'signal_sample_run2_win10.root'
  split_bkg_filename = 'bkg_sample_run2_win10.root'
  train_sample_filename = 'train_sample_run2_win10.root'
  test_sample_filename = 'test_sample_run2_win10.root'
  test_full_sample_filename = 'test_full_sample_run2_win10.root'


  # List of branches from zgamma_preprocess.py
  branches = ['photon_mva','min_dR','max_dR','pt_mass','cosTheta',
              'costheta', 'phi','photon_res','photon_rapidity','l1_rapidity',
              'l2_rapidity', 'decorr_photon_pt','photon_pt_mass', 'w_lumi', 'llg_mass', 
              'llg_mass_err', 'llg_flavor', 'gamma_pt', 'w_llg_mass', 'weight']

  np.random.seed(1)
  # Create training and test sample. Randomize events between training and testing. Scales up weights.
  # weight index starts from 0
  split_ntuple( in_filename = input_signal_filename, in_treename = 'tree', branches = branches,
    cut = '1',
    train_fraction = 0.05, weight_branch_index = 19, out_filename = split_signal_filename)
  split_ntuple( in_filename = input_bkg_filename, in_treename = 'tree', branches = branches,
    cut = '1',
    train_fraction = 0.05, weight_branch_index = 19, out_filename = split_bkg_filename)

  # Combine signal and bkg for train sample. Randomizes events between signal and bkg.
  combine_signal_bkg(bkg_filename = split_bkg_filename, bkg_treename = 'train_tree',
    signal_filename = split_signal_filename, signal_treename = 'train_tree',
    cut = '(llg_mass>120) & (llg_mass<130)',
    #cut = '(llg_mass>110) & (llg_mass<140)',
    out_filename = train_sample_filename, out_treename = 'train_tree')

  # Combine signal and bkg for test sample. Randomizes events between signal and bkg. # Note: When trying to update root file, it created issues.
  combine_signal_bkg(bkg_filename = split_bkg_filename, bkg_treename = 'test_tree',
    signal_filename = split_signal_filename, signal_treename = 'test_tree',
    cut = '(llg_mass>120) & (llg_mass<130)',
    #cut = '(llg_mass>110) & (llg_mass<140)',
    out_filename = test_sample_filename, out_treename = 'test_tree')

  # Combine signal and bkg for test sample. Randomizes events between signal and bkg.
  # Note: When trying to update root file, it created issues.
  combine_signal_bkg(bkg_filename = split_bkg_filename, bkg_treename = 'test_tree',
    signal_filename = split_signal_filename, signal_treename = 'test_tree',
    cut = '1',
    out_filename = test_full_sample_filename, out_treename = 'test_tree')
