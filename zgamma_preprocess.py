#!/usr/bin/env python3
'''
Function that generates n-tuples for MVA training
'''
import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)
import rootutils

def write_ntuples(filenames, cuts, out_name, defines=[], tree_name='tree', branches=()):
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

float get_mass_err(RVec<float> llphoton_l1_masserr, RVec<float> llphoton_l2_masserr, RVec<float> llphoton_ph_masserr) {
  return sqrt(pow(llphoton_l1_masserr[0],2)+pow(llphoton_l2_masserr[0],2)+pow(llphoton_ph_masserr[0],2));
}

float get_flavor(RVec<int> ll_lepid) {
  if (ll_lepid[0] == 11) return 1.;
  if (ll_lepid[0] == 13) return 2.;
  return 0.;
}

""")

if __name__=='__main__':
  ROOT.EnableImplicitMT()
  #parameters
  #TODO: move generic function to /lib/
  cuts = ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL||HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ||HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
      'll_m.size()>0&&llphoton_m.size()>0',
      'stitch_dy||(type%1000!=6)',
      '(ll_m[0]>50)&&(photon_pt[0]/llphoton_m[0]>=15.0/110.0)&&((llphoton_m[0]+ll_m[0])>185)&&(photon_drmin[0]>0.4)',
      '(ll_lepid[0]==11&&el_pt[ll_i1[0]]>25&&el_pt[ll_i2[0]]>15)||(ll_lepid[0]==13&&mu_pt[ll_i1[0]]>20&&mu_pt[ll_i2[0]]>10)',
      #'llphoton_m[0]>120&&llphoton_m[0]<130',
      'llphoton_m[0]>100&&llphoton_m[0]<180',
      ]
  #defines = [('higgsdr','llphoton_dr[0]'),('higgspt','llphoton_pt[0]'),('zpt','ll_pt[0]'),('phpt','photon_pt[0]')]
  #branches = ('higgsdr','higgspt','zpt','phpt')
  defines = [('photon_mva','photon_idmva[0]'),
             ('min_dR','photon_drmin[0]'),
             ('max_dR','get_max_dr(photon_eta,photon_phi,el_eta,el_phi,mu_eta,mu_phi,ll_lepid,ll_i1,ll_i2)'),
             ('pt_mass','llphoton_pt[0]/llphoton_m[0]'),
             ('cosTheta','llphoton_cosTheta[0]'),
             ('costheta','llphoton_costheta[0]'),
             ('phi','llphoton_phi[0]'),
             ('photon_res','photon_pterr[0]/photon_pt[0]'),
             ('photon_rapidity','photon_eta[0]'),
             ('l1_rapidity','get_l1_rapidity(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
             ('l2_rapidity','get_l2_rapidity(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
             ('decorr_photon_pt','photon_pt[0]-0.207*llphoton_m[0]'),
             ('photon_pt_mass','photon_pt[0]/llphoton_m[0]'),
             ('llg_mass','llphoton_m[0]'),
             ('llg_mass_err','get_mass_err(llphoton_l1_masserr,llphoton_l2_masserr,llphoton_ph_masserr)'),
             ('llg_flavor', 'get_flavor(ll_lepid)'),
             ('gamma_pt', 'photon_pt[0]')
             ]
  branches = ('photon_mva','min_dR','max_dR','pt_mass','cosTheta','costheta',
      'phi','photon_res','photon_rapidity','l1_rapidity','l2_rapidity','decorr_photon_pt','photon_pt_mass','w_lumi', 'llg_mass', 'llg_mass_err', 'llg_flavor', 'gamma_pt')
  #define drmax, pt_mass, first index
  #make n-tuples
  #signal_files = '/Users/jbkim/Work/nn_study/pico/NanoAODv9/htozgamma_deathvalley_v3/2017/mc/skim_llg/*GluGluHToZG*.root'
  #bkg_1_files = '/Users/jbkim/Work/nn_study/pico/NanoAODv9/htozgamma_deathvalley_v3/2017/mc/skim_llg/*DYJetsToLL_M-50*madgraphMLM*.root'
  #bkg_2_files = '/Users/jbkim/Work/nn_study/pico/NanoAODv9/htozgamma_deathvalley_v3/2017/mc/skim_llg/*ZGToLLG_01J_5f*.root'
  signal_files = []
  bkg_files = []
  for year in [2016, '2016APV', 2017, 2018]:
    signal_file = f'/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_deathvalley_v3/{year}/mc/skim_llg/*GluGluHToZG*M-125*.root'
    signal_files.append(signal_file)
    bkg_1_file = f'/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_deathvalley_v3/{year}/mc/skim_llg/*DYJetsToLL_M-50*madgraphMLM*.root'
    bkg_2_file = f'/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_deathvalley_v3/{year}/mc/skim_llg/*ZGToLLG_01J_5f_TuneCP5*.root'
    bkg_files.append(bkg_1_file)
    bkg_files.append(bkg_2_file)
  write_ntuples(signal_files,
      cuts,
      'ntuples/train_decorr_sig_run2.root',
      defines,
      'tree',
      branches)
  write_ntuples(bkg_files,
      cuts,
      'ntuples/train_decorr_bak_run2.root',
      defines,
      'tree',
      branches)

  #rootutils.plot_variables('ntuples/train_decorr_sig_run2.root', 'tree', branches, out_folder = 'plots')
  #rootutils.plot_variables('ntuples/train_decorr_bak_run2.root', 'tree', branches, out_folder = 'plots')
