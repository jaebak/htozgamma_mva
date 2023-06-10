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
  df = df.DefinePerSample("luminosity", "get_luminosity(rdfslot_, rdfsampleinfo_)")
  df = df.DefinePerSample("year", "get_year(rdfslot_, rdfsampleinfo_)")
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
  float luminosity = 0;
  if (id.Contains("2016APV/mc")) luminosity = 16.8;
  else if (id.Contains("2016/mc"))  luminosity = 19.5;
  else if (id.Contains("2017/mc")) luminosity = 41.48;
  else if (id.Contains("2018/mc")) luminosity = 59.83;
  //cout<<id.AsString()<<" "<<luminosity<<endl;
  return luminosity;
}

int get_year(unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
  int year = 0;
  if (id.Contains("2016APV/mc")) year = 2016;
  else if (id.Contains("2016/mc"))  year = 2016;
  else if (id.Contains("2017/mc")) year = 2017;
  else if (id.Contains("2018/mc")) year = 2018;
  return year;
}

// Find gen photon from Higgs and compare with reconstructed photon
float get_gamma_pt_diff(RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  //float dr = -999;
  float dpt = -999;
  float gamma_pt = photon_pt[0];
  float gamma_eta = photon_eta[0];
  float gamma_phi = photon_phi[0];
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != 22) continue;
    if (mc_status[imc] != 1) continue; 
    if (mc_mom[imc] != 25) continue;
    //float deta = gamma_eta - mc_eta[imc];
    //float dphi = TVector2::Phi_mpi_pi(gamma_phi - mc_phi[imc]);
    //dr = TMath::Sqrt(deta*deta + dphi*dphi);
    dpt = gamma_pt - mc_pt[imc];
    break;
  }
  // Limit dpt
  if (dpt>10) dpt = 10;
  if (dpt<-10) dpt = -10;
  return dpt;
}

// Find gen photon from Higgs and compare with reconstructed photon
float get_gamma_eta_diff(RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  float deta = 999;
  float gamma_pt = photon_pt[0];
  float gamma_eta = photon_eta[0];
  float gamma_phi = photon_phi[0];
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != 22) continue;
    if (mc_status[imc] != 1) continue; 
    if (mc_mom[imc] != 25) continue;
    deta = gamma_eta - mc_eta[imc];
    break;
  }
  return deta;
}

// Find gen photon from Higgs and compare with reconstructed photon
float get_gamma_phi_diff(RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  float dphi = 999;
  float gamma_pt = photon_pt[0];
  float gamma_eta = photon_eta[0];
  float gamma_phi = photon_phi[0];
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != 22) continue;
    if (mc_status[imc] != 1) continue; 
    if (mc_mom[imc] != 25) continue;
    dphi = TVector2::Phi_mpi_pi(gamma_phi - mc_phi[imc]);
    break;
  }
  return dphi;
}

// Find gen lep plus from Higgs and compare with reconstructed lep plus
float get_lep_plus_pt_diff(RVec<float> mu_pt, RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> mu_charge,
     RVec<float> el_pt, RVec<float> el_eta, RVec<float> el_phi, RVec<float> el_charge,
     RVec<int> ll_lepid, RVec<int> ll_i1, RVec<int> ll_i2, RVec<int> llphoton_ill,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  int lep_id = -999;
  float lep_plus_pt = -999;
  float lep_plus_eta = -999;
  float lep_plus_phi = -999;
  // Find rec lep plus
  lep_id = ll_lepid[llphoton_ill[0]];
  if (lep_id == 13) { // muon
    int mu1_idx = ll_i1[llphoton_ill[0]];
    int mu2_idx = ll_i2[llphoton_ill[0]];
    int mu_plus_idx = -1; 
    int mu_minus_idx = -1;
    if (mu_charge[mu1_idx] == mu_charge[mu2_idx]) return -10;
    if (mu_charge[mu1_idx] == 1) {
      mu_plus_idx = mu1_idx;
      mu_minus_idx = mu2_idx;
    } else {
      mu_plus_idx = mu2_idx;
      mu_minus_idx = mu1_idx;
    }
    lep_plus_pt = mu_pt[mu_plus_idx];
    lep_plus_eta = mu_eta[mu_plus_idx];
    lep_plus_phi = mu_phi[mu_plus_idx];
  } else { // electron
    int el1_idx = ll_i1[llphoton_ill[0]];
    int el2_idx = ll_i2[llphoton_ill[0]];
    int el_plus_idx = -1; 
    int el_minus_idx = -1;
    if (el_charge[el1_idx] == el_charge[el2_idx]) return -10;
    if (el_charge[el1_idx] == 1) {
      el_plus_idx = el1_idx;
      el_minus_idx = el2_idx;
    } else {
      el_plus_idx = el2_idx;
      el_minus_idx = el1_idx;
    }
    lep_plus_pt = el_pt[el_plus_idx];
    lep_plus_eta = el_eta[el_plus_idx];
    lep_plus_phi = el_phi[el_plus_idx];
  }
  float dr = -999;
  float dpt = -999;
  // Find gen lep plus
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != -lep_id ) continue; // Find plus charge
    if (mc_mom[imc] != 23) continue; // Note, assumed event will have 1 Z.
    float deta = lep_plus_eta - mc_eta[imc];
    float dphi = TVector2::Phi_mpi_pi(lep_plus_phi - mc_phi[imc]);
    dr = TMath::Sqrt(deta*deta + dphi*dphi);
    dpt = lep_plus_pt - mc_pt[imc];
    break;
  }
  // Limit dpt
  if (dpt>10) dpt = 10;
  if (dpt<-10) dpt = -10;
  return dpt;
}

// Find gen lep plus from Higgs and compare with reconstructed lep plus
float get_lep_plus_eta_diff(RVec<float> mu_pt, RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> mu_charge,
     RVec<float> el_pt, RVec<float> el_eta, RVec<float> el_phi, RVec<float> el_charge,
     RVec<int> ll_lepid, RVec<int> ll_i1, RVec<int> ll_i2, RVec<int> llphoton_ill,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  int lep_id = -999;
  float lep_plus_pt = -999;
  float lep_plus_eta = -999;
  float lep_plus_phi = -999;
  // Find rec lep plus
  lep_id = ll_lepid[llphoton_ill[0]];
  if (lep_id == 13) { // muon
    int mu1_idx = ll_i1[llphoton_ill[0]];
    int mu2_idx = ll_i2[llphoton_ill[0]];
    int mu_plus_idx = -1; 
    int mu_minus_idx = -1;
    if (mu_charge[mu1_idx] == mu_charge[mu2_idx]) return -999;
    if (mu_charge[mu1_idx] == 1) {
      mu_plus_idx = mu1_idx;
      mu_minus_idx = mu2_idx;
    } else {
      mu_plus_idx = mu2_idx;
      mu_minus_idx = mu1_idx;
    }
    lep_plus_pt = mu_pt[mu_plus_idx];
    lep_plus_eta = mu_eta[mu_plus_idx];
    lep_plus_phi = mu_phi[mu_plus_idx];
  } else { // electron
    int el1_idx = ll_i1[llphoton_ill[0]];
    int el2_idx = ll_i2[llphoton_ill[0]];
    int el_plus_idx = -1; 
    int el_minus_idx = -1;
    if (el_charge[el1_idx] == el_charge[el2_idx]) return -999;
    if (el_charge[el1_idx] == 1) {
      el_plus_idx = el1_idx;
      el_minus_idx = el2_idx;
    } else {
      el_plus_idx = el2_idx;
      el_minus_idx = el1_idx;
    }
    lep_plus_pt = el_pt[el_plus_idx];
    lep_plus_eta = el_eta[el_plus_idx];
    lep_plus_phi = el_phi[el_plus_idx];
  }
  // Find gen lep plus
  float deta = -999;
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != -lep_id ) continue; // Find plus charge
    if (mc_mom[imc] != 23) continue; // Note, assumed event will have 1 Z.
    deta = lep_plus_eta - mc_eta[imc];
    break;
  }
  return deta;
}

// Find gen lep plus from Higgs and compare with reconstructed lep plus
float get_lep_plus_phi_diff(RVec<float> mu_pt, RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> mu_charge,
     RVec<float> el_pt, RVec<float> el_eta, RVec<float> el_phi, RVec<float> el_charge,
     RVec<int> ll_lepid, RVec<int> ll_i1, RVec<int> ll_i2, RVec<int> llphoton_ill,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  int lep_id = -999;
  float lep_plus_pt = -999;
  float lep_plus_eta = -999;
  float lep_plus_phi = -999;
  // Find rec lep plus
  lep_id = ll_lepid[llphoton_ill[0]];
  if (lep_id == 13) { // muon
    int mu1_idx = ll_i1[llphoton_ill[0]];
    int mu2_idx = ll_i2[llphoton_ill[0]];
    int mu_plus_idx = -1; 
    int mu_minus_idx = -1;
    if (mu_charge[mu1_idx] == mu_charge[mu2_idx]) return -999;
    if (mu_charge[mu1_idx] == 1) {
      mu_plus_idx = mu1_idx;
      mu_minus_idx = mu2_idx;
    } else {
      mu_plus_idx = mu2_idx;
      mu_minus_idx = mu1_idx;
    }
    lep_plus_pt = mu_pt[mu_plus_idx];
    lep_plus_eta = mu_eta[mu_plus_idx];
    lep_plus_phi = mu_phi[mu_plus_idx];
  } else { // electron
    int el1_idx = ll_i1[llphoton_ill[0]];
    int el2_idx = ll_i2[llphoton_ill[0]];
    int el_plus_idx = -1; 
    int el_minus_idx = -1;
    if (el_charge[el1_idx] == el_charge[el2_idx]) return -999;
    if (el_charge[el1_idx] == 1) {
      el_plus_idx = el1_idx;
      el_minus_idx = el2_idx;
    } else {
      el_plus_idx = el2_idx;
      el_minus_idx = el1_idx;
    }
    lep_plus_pt = el_pt[el_plus_idx];
    lep_plus_eta = el_eta[el_plus_idx];
    lep_plus_phi = el_phi[el_plus_idx];
  }
  float dphi = -999;
  // Find gen lep plus
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != -lep_id ) continue; // Find plus charge
    if (mc_mom[imc] != 23) continue; // Note, assumed event will have 1 Z.
    dphi = TVector2::Phi_mpi_pi(lep_plus_phi - mc_phi[imc]);
    break;
  }
  return dphi;
}

// Find gen lep minus from Higgs and compare with reconstructed lep minus
float get_lep_minus_pt_diff(RVec<float> mu_pt, RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> mu_charge,
     RVec<float> el_pt, RVec<float> el_eta, RVec<float> el_phi, RVec<float> el_charge,
     RVec<int> ll_lepid, RVec<int> ll_i1, RVec<int> ll_i2, RVec<int> llphoton_ill,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  int lep_id = -999;
  float lep_minus_pt = -999;
  float lep_minus_eta = -999;
  float lep_minus_phi = -999;
  // Find rec lep minus
  lep_id = ll_lepid[llphoton_ill[0]];
  if (lep_id == 13) { // muon
    int mu1_idx = ll_i1[llphoton_ill[0]];
    int mu2_idx = ll_i2[llphoton_ill[0]];
    int mu_plus_idx = -1; 
    int mu_minus_idx = -1;
    if (mu_charge[mu1_idx] == mu_charge[mu2_idx]) return -10;
    if (mu_charge[mu1_idx] == 1) {
      mu_plus_idx = mu1_idx;
      mu_minus_idx = mu2_idx;
    } else {
      mu_plus_idx = mu2_idx;
      mu_minus_idx = mu1_idx;
    }
    lep_minus_pt = mu_pt[mu_minus_idx];
    lep_minus_eta = mu_eta[mu_minus_idx];
    lep_minus_phi = mu_phi[mu_minus_idx];
  } else { // electron
    int el1_idx = ll_i1[llphoton_ill[0]];
    int el2_idx = ll_i2[llphoton_ill[0]];
    int el_plus_idx = -1; 
    int el_minus_idx = -1;
    if (el_charge[el1_idx] == el_charge[el2_idx]) return -10;
    if (el_charge[el1_idx] == 1) {
      el_plus_idx = el1_idx;
      el_minus_idx = el2_idx;
    } else {
      el_plus_idx = el2_idx;
      el_minus_idx = el1_idx;
    }
    lep_minus_pt = el_pt[el_minus_idx];
    lep_minus_eta = el_eta[el_minus_idx];
    lep_minus_phi = el_phi[el_minus_idx];
  }
  float dr = -999;
  float dpt = -999;
  // Find gen lep minus
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != lep_id ) continue; // Find minus charge
    if (mc_mom[imc] != 23) continue; // Note, assumed event will have 1 Z.
    float deta = lep_minus_eta - mc_eta[imc];
    float dphi = TVector2::Phi_mpi_pi(lep_minus_phi - mc_phi[imc]);
    dr = TMath::Sqrt(deta*deta + dphi*dphi);
    dpt = lep_minus_pt - mc_pt[imc];
    break;
  }
  // Limit dpt
  if (dpt>10) dpt = 10;
  if (dpt<-10) dpt = -10;
  return dpt;
}

// Find gen lep minus from Higgs and compare with reconstructed lep minus
float get_lep_minus_eta_diff(RVec<float> mu_pt, RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> mu_charge,
     RVec<float> el_pt, RVec<float> el_eta, RVec<float> el_phi, RVec<float> el_charge,
     RVec<int> ll_lepid, RVec<int> ll_i1, RVec<int> ll_i2, RVec<int> llphoton_ill,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  int lep_id = -999;
  float lep_minus_pt = -999;
  float lep_minus_eta = -999;
  float lep_minus_phi = -999;
  // Find rec lep minus
  lep_id = ll_lepid[llphoton_ill[0]];
  if (lep_id == 13) { // muon
    int mu1_idx = ll_i1[llphoton_ill[0]];
    int mu2_idx = ll_i2[llphoton_ill[0]];
    int mu_plus_idx = -1; 
    int mu_minus_idx = -1;
    if (mu_charge[mu1_idx] == mu_charge[mu2_idx]) return -999;
    if (mu_charge[mu1_idx] == 1) {
      mu_plus_idx = mu1_idx;
      mu_minus_idx = mu2_idx;
    } else {
      mu_plus_idx = mu2_idx;
      mu_minus_idx = mu1_idx;
    }
    lep_minus_pt = mu_pt[mu_minus_idx];
    lep_minus_eta = mu_eta[mu_minus_idx];
    lep_minus_phi = mu_phi[mu_minus_idx];
  } else { // electron
    int el1_idx = ll_i1[llphoton_ill[0]];
    int el2_idx = ll_i2[llphoton_ill[0]];
    int el_plus_idx = -1; 
    int el_minus_idx = -1;
    if (el_charge[el1_idx] == el_charge[el2_idx]) return -999;
    if (el_charge[el1_idx] == 1) {
      el_plus_idx = el1_idx;
      el_minus_idx = el2_idx;
    } else {
      el_plus_idx = el2_idx;
      el_minus_idx = el1_idx;
    }
    lep_minus_pt = el_pt[el_minus_idx];
    lep_minus_eta = el_eta[el_minus_idx];
    lep_minus_phi = el_phi[el_minus_idx];
  }
  float deta = -999;
  // Find gen lep minus
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != lep_id ) continue; // Find minus charge
    if (mc_mom[imc] != 23) continue; // Note, assumed event will have 1 Z.
    deta = lep_minus_eta - mc_eta[imc];
    break;
  }
  return deta;
}

// Find gen lep minus from Higgs and compare with reconstructed lep minus
float get_lep_minus_phi_diff(RVec<float> mu_pt, RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> mu_charge,
     RVec<float> el_pt, RVec<float> el_eta, RVec<float> el_phi, RVec<float> el_charge,
     RVec<int> ll_lepid, RVec<int> ll_i1, RVec<int> ll_i2, RVec<int> llphoton_ill,
     RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi) {
  int lep_id = -999;
  float lep_minus_pt = -999;
  float lep_minus_eta = -999;
  float lep_minus_phi = -999;
  // Find rec lep minus
  lep_id = ll_lepid[llphoton_ill[0]];
  if (lep_id == 13) { // muon
    int mu1_idx = ll_i1[llphoton_ill[0]];
    int mu2_idx = ll_i2[llphoton_ill[0]];
    int mu_plus_idx = -1; 
    int mu_minus_idx = -1;
    if (mu_charge[mu1_idx] == mu_charge[mu2_idx]) return -999;
    if (mu_charge[mu1_idx] == 1) {
      mu_plus_idx = mu1_idx;
      mu_minus_idx = mu2_idx;
    } else {
      mu_plus_idx = mu2_idx;
      mu_minus_idx = mu1_idx;
    }
    lep_minus_pt = mu_pt[mu_minus_idx];
    lep_minus_eta = mu_eta[mu_minus_idx];
    lep_minus_phi = mu_phi[mu_minus_idx];
  } else { // electron
    int el1_idx = ll_i1[llphoton_ill[0]];
    int el2_idx = ll_i2[llphoton_ill[0]];
    int el_plus_idx = -1; 
    int el_minus_idx = -1;
    if (el_charge[el1_idx] == el_charge[el2_idx]) return -999;
    if (el_charge[el1_idx] == 1) {
      el_plus_idx = el1_idx;
      el_minus_idx = el2_idx;
    } else {
      el_plus_idx = el2_idx;
      el_minus_idx = el1_idx;
    }
    lep_minus_pt = el_pt[el_minus_idx];
    lep_minus_eta = el_eta[el_minus_idx];
    lep_minus_phi = el_phi[el_minus_idx];
  }
  float dphi = -999;
  // Find gen lep minus
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    // Require photon to be final state and from Higgs
    if (mc_id[imc] != lep_id ) continue; // Find minus charge
    if (mc_mom[imc] != 23) continue; // Note, assumed event will have 1 Z.
    dphi = TVector2::Phi_mpi_pi(lep_minus_phi - mc_phi[imc]);
    break;
  }
  return dphi;
}

""")

if __name__=='__main__':
  print('Start zgamma_preprocessing')
  ROOT.EnableImplicitMT()
  #parameters
  #TODO: move generic function to /lib/
  cuts = ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL||HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ||HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
      'll_m.size()>0&&llphoton_m.size()>0',
      'stitch_dy||(type/1000!=6)',
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
             ('photon_res_e','photon_pterr[0]/(photon_pt[0]*cosh(photon_eta[0]))'),
             ('photon_rapidity','photon_eta[0]'),
             ('l1_rapidity','get_l1_rapidity(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
             ('l2_rapidity','get_l2_rapidity(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
             ('decorr_photon_pt','photon_pt[0]-0.207*llphoton_m[0]'),
             ('photon_pt_mass','photon_pt[0]/llphoton_m[0]'),
             ('llg_mass','llphoton_m[0]'),
             ('llg_mass_err','get_mass_err(llphoton_l1_masserr,llphoton_l2_masserr,llphoton_ph_masserr)'),
             ('llg_flavor', 'get_flavor(ll_lepid)'),
             ('gamma_pt', 'photon_pt[0]'),
             ('llg_eta', 'llphoton_eta[0]'),
             ('llg_phi', 'llphoton_phi[0]'),
             ('llg_ptt', 'get_llg_ptt(photon_pt, photon_eta, photon_phi, llphoton_pt, llphoton_eta, llphoton_phi, ll_pt, ll_eta, ll_phi)'),
             ('z_eta', 'll_eta[0]'),
             ('z_phi', 'll_phi[0]'),
             ('l1_phi', 'get_l1_phi(el_pt,el_phi,mu_pt,mu_phi,ll_lepid,ll_i1,ll_i2)'),
             ('l2_phi', 'get_l2_phi(el_pt,el_phi,mu_pt,mu_phi,ll_lepid,ll_i1,ll_i2)'),
             ('gamma_eta', 'photon_eta[0]'),
             ('gamma_phi', 'photon_phi[0]'),
             ('gamma_pt_diff', 'get_gamma_pt_diff(photon_pt, photon_eta, photon_phi, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('gamma_eta_diff', 'get_gamma_eta_diff(photon_pt, photon_eta, photon_phi, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('gamma_phi_diff', 'get_gamma_phi_diff(photon_pt, photon_eta, photon_phi, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('lep_plus_pt_diff', 'get_lep_plus_pt_diff(mu_pt, mu_eta, mu_phi, mu_charge, el_pt, el_eta, el_phi, el_charge, ll_lepid, ll_i1, ll_i2, llphoton_ill, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('lep_plus_eta_diff', 'get_lep_plus_eta_diff(mu_pt, mu_eta, mu_phi, mu_charge, el_pt, el_eta, el_phi, el_charge, ll_lepid, ll_i1, ll_i2, llphoton_ill, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('lep_plus_phi_diff', 'get_lep_plus_phi_diff(mu_pt, mu_eta, mu_phi, mu_charge, el_pt, el_eta, el_phi, el_charge, ll_lepid, ll_i1, ll_i2, llphoton_ill, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('lep_minus_pt_diff', 'get_lep_minus_pt_diff(mu_pt, mu_eta, mu_phi, mu_charge, el_pt, el_eta, el_phi, el_charge, ll_lepid, ll_i1, ll_i2, llphoton_ill, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('lep_minus_eta_diff', 'get_lep_minus_eta_diff(mu_pt, mu_eta, mu_phi, mu_charge, el_pt, el_eta, el_phi, el_charge, ll_lepid, ll_i1, ll_i2, llphoton_ill, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ('lep_minus_phi_diff', 'get_lep_minus_phi_diff(mu_pt, mu_eta, mu_phi, mu_charge, el_pt, el_eta, el_phi, el_charge, ll_lepid, ll_i1, ll_i2, llphoton_ill, mc_id, mc_status, mc_mom, mc_pt, mc_eta, mc_phi)'),
             ]
  branches = ('photon_mva','min_dR','max_dR','pt_mass','cosTheta','costheta',
      'phi','photon_res','photon_res_e', 'photon_rapidity','l1_rapidity','l2_rapidity','decorr_photon_pt','photon_pt_mass','w_lumi', 'llg_mass', 'llg_mass_err', 'llg_flavor', 'gamma_pt',
      'llg_eta', 'llg_phi', 'llg_ptt', 'z_eta', 'z_phi', 'l1_phi', 'l2_phi', 'gamma_eta', 'gamma_phi', 
      'year', 'luminosity', 'w_lumiXyear', 
      'gamma_pt_diff', 'gamma_eta_diff', 'gamma_phi_diff',
      'lep_plus_pt_diff', 'lep_plus_eta_diff', 'lep_plus_phi_diff',
      'lep_minus_pt_diff', 'lep_minus_eta_diff', 'lep_minus_phi_diff',)
  #define drmax, pt_mass, first index
  #make n-tuples
  #signal_files = '/Users/jbkim/Work/nn_study/pico/NanoAODv9/htozgamma_deathvalley_v3/2017/mc/skim_llg/*GluGluHToZG*.root'
  #bkg_1_files = '/Users/jbkim/Work/nn_study/pico/NanoAODv9/htozgamma_deathvalley_v3/2017/mc/skim_llg/*DYJetsToLL_M-50*madgraphMLM*.root'
  #bkg_2_files = '/Users/jbkim/Work/nn_study/pico/NanoAODv9/htozgamma_deathvalley_v3/2017/mc/skim_llg/*ZGToLLG_01J_5f*.root'
  signal_files = []
  bkg_files = []
  for year in [2016, '2016APV', 2017, 2018]:
  #for year in [2017]:
    signal_file = f'/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_deathvalley_v3/{year}/mc/skim_llg/*GluGluHToZG*M-125*.root'
    signal_files.append(signal_file)
    bkg_1_file = f'/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_deathvalley_v3/{year}/mc/skim_llg/*DYJetsToLL_M-50*madgraphMLM*.root'
    #bkg_1_file = f'/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_deathvalley_v3/{year}/mc/skim_llg/*DYJetsToLL_M-50*amcatnloFXFX*.root'
    bkg_2_file = f'/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_deathvalley_v3/{year}/mc/skim_llg/*ZGToLLG_01J_5f_TuneCP5*.root'
    bkg_files.append(bkg_1_file)
    bkg_files.append(bkg_2_file)
  write_ntuples(signal_files,
      cuts,
      'ntuples/train_decorr_sig_run2_lumi_diff.root',
      #'ntuples/train_decorr_sig_run2_lumi.root',
      #'ntuples/train_decorr_sig.root',
      defines,
      'tree',
      branches)
  write_ntuples(bkg_files,
      cuts,
      'ntuples/train_decorr_bak_run2_lumi_diff.root',
      #'ntuples/train_decorr_bak_run2_lumi.root',
      #'ntuples/train_decorr_bak.root',
      defines,
      'tree',
      branches)
  # Combine years

  #rootutils.plot_variables('ntuples/train_decorr_sig_run2.root', 'tree', branches, out_folder = 'plots')
  #rootutils.plot_variables('ntuples/train_decorr_bak_run2.root', 'tree', branches, out_folder = 'plots')
