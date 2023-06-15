#!/usr/bin/env python3
'''
Function that generates n-tuples for MVA training
'''
import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)
import rootutils

def get_entries(filenames, tree_name):
  chain = ROOT.TChain(tree_name)
  for filename in filenames:
    chain.Add(filename) 
  return chain.GetEntries()
  

def write_ntuples(filenames, cuts, out_name, defines=[], tree_name='tree', branches=(), entries=1):
  '''Generate ROOT n-tuple from existing n-tuple
  
  Parameters:
  filenames - list of filenames of signal ROOT n-tuples
  cuts - list of cuts expressed as strings in order they should be applied
  out_name - output filename of n-tuple
  defines - list of 2-tuples describing new branches to define in the format (name, expr)
            note that these must be defineable before cuts
  tree_name - name of tree in ROOT file
  branches - tuple of branches to save; if empty all branches are saved
  '''
  filenames_vec = ROOT.std.vector('string')()
  for filename in filenames:
    filenames_vec.push_back(filename)
  df = ROOT.RDataFrame('tree',filenames_vec)
  for define in defines:
    df = df.Define(define[0],define[1])
  df = df.Define("entries", f'return {entries};')
  df = df.Define("luminosity", f'return 137.5;')
  df = df.DefinePerSample("cross_section", "get_cross_section(rdfslot_, rdfsampleinfo_)")
  df = df.Define("w_lumi", "cross_section / entries")
  df = df.Define("w_lumiXyear", "w_lumi * luminosity")
  for cut in cuts:
    df = df.Filter(cut)
  if (branches == ()):
    df.Snapshot(tree_name,out_name)
  else:
    df.Snapshot(tree_name,out_name,branches)
  print('Wrote '+out_name)

ROOT.gInterpreter.Declare("""
template <class C>
using RVec = ROOT::VecOps::RVec<C>;


float get_dr(float eta1, float phi1, float eta2, float phi2) {
  const double PI = 3.1415;
  double dphi = fmod(fabs(phi2-phi1), 2.*PI);
  dphi = dphi>PI ? 2.*PI-dphi : dphi;
  double deta = fabs(eta1-eta2);
  return sqrt(deta*deta+dphi*dphi);
}

float get_max_dr(RVec<float> photon_eta, RVec<float> photon_phi, 
    RVec<float> el_eta, RVec<float> el_phi, RVec<float> mu_eta,
    RVec<float> mu_phi, RVec<int> ll_lepid, RVec<int> ll_i1,
    RVec<int> ll_i2) {
  float dr1, dr2;
  if (ll_lepid[0]==11) {
    dr1 = get_dr(photon_eta[0],photon_phi[0],el_eta[ll_i1[0]],el_phi[ll_i1[0]]);
    dr2 = get_dr(photon_eta[0],photon_phi[0],el_eta[ll_i2[0]],el_phi[ll_i2[0]]);
    return dr1 > dr2 ? dr1 : dr2;
  }
  dr1 = get_dr(photon_eta[0],photon_phi[0],mu_eta[ll_i1[0]],mu_phi[ll_i1[0]]);
  dr2 = get_dr(photon_eta[0],photon_phi[0],mu_eta[ll_i2[0]],mu_phi[ll_i2[0]]);
  return dr1 > dr2 ? dr1 : dr2;
}

float get_l1_rapidity(RVec<float> el_pt, RVec<float> el_eta, 
    RVec<float> mu_pt, RVec<float> mu_eta, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_eta[ll_i1[0]] : el_eta[ll_i2[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_eta[ll_i1[0]] : mu_eta[ll_i2[0]];
}

float get_l2_rapidity(RVec<float> el_pt, RVec<float> el_eta, 
    RVec<float> mu_pt, RVec<float> mu_eta, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_eta[ll_i2[0]] : el_eta[ll_i1[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_eta[ll_i2[0]] : mu_eta[ll_i1[0]];
}

float get_l1_phi(RVec<float> el_pt, RVec<float> el_phi, 
    RVec<float> mu_pt, RVec<float> mu_phi, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_phi[ll_i1[0]] : el_phi[ll_i2[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_phi[ll_i1[0]] : mu_phi[ll_i2[0]];
}

float get_l2_phi(RVec<float> el_pt, RVec<float> el_phi, 
    RVec<float> mu_pt, RVec<float> mu_phi, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_phi[ll_i2[0]] : el_phi[ll_i1[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_phi[ll_i2[0]] : mu_phi[ll_i1[0]];
}

float get_mass_err(RVec<float> llphoton_l1_masserr, RVec<float> llphoton_l2_masserr, RVec<float> llphoton_ph_masserr) {
  return sqrt(pow(llphoton_l1_masserr[0],2)+pow(llphoton_l2_masserr[0],2)+pow(llphoton_ph_masserr[0],2));
}

float get_flavor(RVec<int> ll_lepid) {
  if (ll_lepid[0] == 11) return 1.;
  if (ll_lepid[0] == 13) return 2.;
  return 0.;
}

float get_llg_ptt(RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi, 
                  RVec<float> llphoton_pt, RVec<float> llphoton_eta, RVec<float> llphoton_phi,
                  RVec<float> ll_pt, RVec<float> ll_eta, RVec<float> ll_phi) {
  TVector3 gamma; gamma.SetPtEtaPhi(photon_pt[0], photon_eta[0], photon_phi[0]);
  TVector3 higgs; higgs.SetPtEtaPhi(llphoton_pt[0], llphoton_eta[0], llphoton_phi[0]);
  TVector3 zboson; zboson.SetPtEtaPhi(ll_pt[0], ll_eta[0], ll_phi[0]);
  gamma.SetZ(0); higgs.SetZ(0); zboson.SetZ(0);
  return higgs.Cross((zboson-gamma).Unit()).Mag();
}

float get_luminosity(unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
  float luminosity = 137.5;
  return luminosity;
}

float get_cross_section(unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
  float w_lumi = 0;
  if (id.Contains("HToZG")) w_lumi = 48.58 * 0.001533 * 0.100974 * 1000; // xs * H->ZG * Z->LL * pb to fb
  else if (id.Contains("_ZG"))  w_lumi = 55.48 * 1000; // xs * pb to fb
  return w_lumi;
}

int get_year(unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
  int year = 0;
  if (id.Contains("2016APV/mc")) year = 2016;
  else if (id.Contains("2016/mc"))  year = 2016;
  else if (id.Contains("2017/mc")) year = 2017;
  else if (id.Contains("2018/mc")) year = 2018;
  return year;
}
""")

if __name__=='__main__':
  ROOT.EnableImplicitMT()
  #parameters
  cuts = ['(ll_m>-999)&&(llg_m>-999)',
      '(ll_m>50)&&(gamma_pt_over_llg_mass>=15.0/110.0)&&((llg_m+ll_m)>185)&&(min_dR_gamma_lepton>0.4)',
      '(lead_lep_pt>25)&&(sublead_lep_pt>15)',
      #'llg_m>120&&llg_m<130',
      '(llg_m>100)&&(llg_m<180)',
      ]
  defines = [('min_dR','min_dR_gamma_lepton'),
             ('max_dR','max_dR_gamma_lepton'),
             ('pt_mass','gamma_pt_over_llg_mass'),
             ('cosTheta','llg_cosTheta'),
             ('costheta','llg_costheta'),
             ('phi','llg_Phi'),
             ('photon_rapidity','gamma_eta'),
             ('l1_rapidity','lead_lep_eta'),
             ('l2_rapidity','sublead_lep_eta'),
             ('photon_pt_mass','gamma_pt_over_llg_mass'),
             ('llg_mass','llg_m'),
             ('llg_flavor', 'e_or_mu'),
             ('z_eta', 'll_eta'),
             ('z_phi', 'll_phi'),
             ('l1_phi', 'lead_lep_phi'),
             ('l2_phi', 'sublead_lep_phi'),
             ]
  branches = ('min_dR','max_dR','pt_mass','cosTheta','costheta',
      'phi', 'photon_rapidity','l1_rapidity','l2_rapidity','photon_pt_mass', 'llg_mass', 'llg_flavor', 'gamma_pt',
      'llg_eta', 'llg_phi', 'llg_ptt', 'z_eta', 'z_phi', 'l1_phi', 'l2_phi', 'gamma_eta', 'gamma_phi', 
      'w_lumiXyear', 'w_lumi')
  signal_files = ['input_ntuples/ntuple_ggH_HToZG_ZToLL_run02.root',
                  'input_ntuples/ntuple_ggH_HToZG_ZToLL_run03.root',
                  'input_ntuples/ntuple_ggH_HToZG_ZToLL_run06.root',
                  'input_ntuples/ntuple_ggH_HToZG_ZToLL_run07.root',
                  'input_ntuples/ntuple_ggH_HToZG_ZToLL_run08.root',
                  'input_ntuples/ntuple_ggH_HToZG_ZToLL_run09.root',
                  'input_ntuples/ntuple_ggH_HToZG_ZToLL_run10.root',
                  'input_ntuples/ntuple_ggH_HToZG_ZToLL_run11.root',]
  bkg_files = ['input_ntuples/ntuple_ZG_ZToLL_run02.root',
               'input_ntuples/ntuple_ZG_ZToLL_run03.root',
               'input_ntuples/ntuple_ZG_ZToLL_run06.root',
               'input_ntuples/ntuple_ZG_ZToLL_run07.root',
               'input_ntuples/ntuple_ZG_ZToLL_run08.root',
               'input_ntuples/ntuple_ZG_ZToLL_run09.root',
               'input_ntuples/ntuple_ZG_ZToLL_run10.root',
               'input_ntuples/ntuple_ZG_ZToLL_run11.root',]
  signal_entries = get_entries(signal_files, 'tree')
  bkg_entries = get_entries(bkg_files, 'tree')
  write_ntuples(signal_files,
      cuts,
      'processed_ntuples/train_decorr_sig_run2_lumi.root',
      #'ntuples/train_decorr_sig.root',
      defines,
      'tree',
      branches,
      signal_entries)
  write_ntuples(bkg_files,
      cuts,
      'processed_ntuples/train_decorr_bak_run2_lumi.root',
      #'ntuples/train_decorr_bak.root',
      defines,
      'tree',
      branches,
      bkg_entries)

  rootutils.plot_variables('processed_ntuples/train_decorr_sig_run2_lumi.root', 'tree', branches, out_folder = 'plots')
  rootutils.plot_variables('processed_ntuples/train_decorr_bak_run2_lumi.root', 'tree', branches, out_folder = 'plots')
