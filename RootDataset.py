import torch.utils.data
import uproot
import numpy as np

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
    label_array = label_array[class_branch[0]][0]
    #label_array = label_array['classID'][0]
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
