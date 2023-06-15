#include "TMVA/Reader.h"

void evaluate(TMVA::Reader *tmva_reader, map<string, Float_t> & float_store, map<string, Int_t> & int_store, 
              vector<string> & tmva_variables, vector<string> & tmva_spectators, string tmva_output, string label,
              TString input_filename, TString input_treename, TString output_filename, bool append_root = false) {
  TFile *input = TFile::Open(input_filename);
  TTree* tree = (TTree*)input->Get(input_treename);
  for (auto var : tmva_variables) tree->SetBranchAddress( var.c_str(), &float_store[var] );
  for (auto var : tmva_spectators) tree->SetBranchAddress( var.c_str(), &float_store[var] );
  tree->SetBranchAddress( label.c_str(), &int_store[label] );
  //tree->SetBranchAddress( "photon_mva", &float_store["photon_mva"] );
  //tree->SetBranchAddress( "min_dR", &float_store["min_dR"] );
  //tree->SetBranchAddress( "pt_mass", &float_store["pt_mass"] );
  //tree->SetBranchAddress( "cosTheta", &float_store["cosTheta"] );
  //tree->SetBranchAddress( "costheta", &float_store["costheta"] );
  //tree->SetBranchAddress( "photon_res", &float_store["photon_res"] );
  //tree->SetBranchAddress( "photon_rapidity", &float_store["photon_rapidity"] );
  //tree->SetBranchAddress( "l1_rapidity", &float_store["l1_rapidity"] );
  //tree->SetBranchAddress( "l2_rapidity", &float_store["l2_rapidity"] );
  //tree->SetBranchAddress( "llg_mass", &float_store["llg_mass"] );
  //tree->SetBranchAddress( "w_lumi", &float_store["w_lumi"] );
  //tree->SetBranchAddress( "classID", &int_store["classID"] );

  TFile * output = 0;
  if (append_root) output = new TFile(output_filename, "update");
  else output = new TFile(output_filename, "recreate");
  TTree * out_tree = new TTree(input_treename,input_treename);
  for (auto var : tmva_variables) out_tree->Branch( var.c_str(), &float_store[var] );
  for (auto var : tmva_spectators) out_tree->Branch( var.c_str(), &float_store[var] );
  out_tree->Branch( label.c_str(), &int_store[label] );
  out_tree->Branch( tmva_output.c_str(), &float_store[tmva_output] );
  //for (auto it : float_store) out_tree->SetBranchAddress( it.first.c_str(), &it.second );
  //for (auto it : int_store) out_tree->SetBranchAddress( it.first.c_str(), &it.second );
  //out_tree->Branch( "classID", &int_store["classID"] );
  //out_tree->Branch( "photon_mva", &float_store["photon_mva"] );
  //out_tree->Branch( "min_dR", &float_store["min_dR"] );
  //out_tree->Branch( "pt_mass", &float_store["pt_mass"] );
  //out_tree->Branch( "cosTheta", &float_store["cosTheta"] );
  //out_tree->Branch( "costheta", &float_store["costheta"] );
  //out_tree->Branch( "photon_res", &float_store["photon_res"] );
  //out_tree->Branch( "photon_rapidity", &float_store["photon_rapidity"] );
  //out_tree->Branch( "l1_rapidity", &float_store["l1_rapidity"] );
  //out_tree->Branch( "l2_rapidity", &float_store["l2_rapidity"] );
  //out_tree->Branch( "llg_mass", &float_store["llg_mass"] );
  //out_tree->Branch( "w_lumi", &float_store["w_lumi"] );
  //out_tree->Branch( "BDT", &float_store["MVA"] );

  // Event loop
  for (Long64_t ievt=0; ievt<tree->GetEntries();ievt++) {
    tree->GetEntry(ievt);
    float_store[tmva_output] = tmva_reader->EvaluateMVA(tmva_output);
    out_tree->Fill();
  }
  out_tree->Write();
  cout<<"Wrote to "<<output_filename<<endl;

  input->Close();
  output->Close();
}

XMLNodePointer_t getNChild(TXMLEngine & xml, XMLNodePointer_t node, int iChild=0) {
  //printf("JB node: %s %i\n", xml.GetNodeName(node), iChild);
  XMLNodePointer_t child = xml.GetChild(node);
  for (int i = 0; i < iChild; i++) {
    child = xml.GetNext(child);
  }
  return child;
}

void parse_xml(string tmva_weights, vector<string> & tmva_variables, vector<string> & tmva_spectators) {
  TXMLEngine xml;
  XMLDocPointer_t xmldoc = xml.ParseFile(tmva_weights.c_str());
  XMLNodePointer_t mainnode = xml.DocGetRootElement(xmldoc);
  // Get variables
  XMLNodePointer_t xml_variables = getNChild(xml, mainnode, 2);
  int nVar = stoi(xml.GetAttr(xml_variables, "NVar"));
  for (int iVar = 0; iVar < nVar; ++iVar) {
    XMLNodePointer_t xml_variable = getNChild(xml, xml_variables, iVar);
    tmva_variables.push_back(xml.GetAttr(xml_variable, "Expression"));
  }
  // Get spectators
  XMLNodePointer_t xml_spectators = getNChild(xml, mainnode, 3);
  int nSpec = stoi(xml.GetAttr(xml_spectators, "NSpec"));
  for (int iSpec = 0; iSpec < nSpec; ++iSpec) {
    XMLNodePointer_t xml_spectator = getNChild(xml, xml_spectators, iSpec);
    tmva_spectators.push_back(xml.GetAttr(xml_spectator, "Expression"));
  }
  xml.FreeDoc(xmldoc);
  cout<<"Variables from xml: ";
  for (auto it : tmva_variables) cout<<it<<" ";
  cout<<endl;
  cout<<"Spectators from xml: ";
  for (auto it : tmva_spectators) cout<<it<<" ";
  cout<<endl;
}

// Run: root use_tmva.C -q
using namespace TMVA;
void use_tmva() {

  // evaluate_mva = 1 is for bdt, 
  // evaluate_mva = 2 is for nn, 
  // evaluate_mva = 3 is for raw bdt, 
  // evaluate_mva = 4 is for run2 bdt
  // evaluate_mva = 5 is for reduced bdt
  // evaluate_mva = 6 is for var12 bdt
  int evaluate_mva = 1;

  // Load bdt tmva
  TMVA::Reader *tmva_reader = new TMVA::Reader( "!Color:!Silent" );

  string input_test_root = "mva_input_ntuples/test_full_sample_run2_lumi.root";
  string input_test_tree = "test_tree";
  string label = "classID";

  string output_root;
  string tmva_weights;
  string tmva_output;
  if (evaluate_mva == 1) {
    output_root = "mva_output_ntuples/tmva_evaluate_bdt_run2.root";
    tmva_weights = "tmva_run2_bdt/weights/TMVAClassification_BDT.weights.xml";
    tmva_output = "BDT";
  } else if (evaluate_mva == 2) {
    output_root = "mva_output_ntuples/tmva_evaluate_nn.root";
    tmva_weights = "tmva_nn/weights/TMVAClassification_DNN.weights.xml";
    tmva_output = "DNN";
  } else if (evaluate_mva == 3) {
    output_root = "tmva_evaluate_raw_bdt.root";
    tmva_weights = "raw_bdt/weights/TMVAClassification_BDT.weights.xml";
    tmva_output = "BDT";
  } else if (evaluate_mva == 4) {
    output_root = "tmva_evaluate_run2_bdt.root";
    tmva_weights = "run2_bdt/weights/TMVAClassification_BDT.weights.xml";
    tmva_output = "BDT";
  } else if (evaluate_mva == 5) {
    output_root = "tmva_evaluate_reduced_bdt.root";
    tmva_weights = "reduced_bdt/weights/TMVAClassification_BDT.weights.xml";
    tmva_output = "BDT";
  } else if (evaluate_mva == 6) {
    output_root = "tmva_evaluate_var12_bdt.root";
    tmva_weights = "var12_bdt/weights/TMVAClassification_BDT.weights.xml";
    tmva_output = "BDT";
  }

  // Parse xml file to get variables and spectators
  vector<string> tmva_variables;
  vector<string> tmva_spectators;
  parse_xml(tmva_weights, tmva_variables, tmva_spectators);

  // Make memory for variables
  map<string, Float_t> float_store;
  for (auto var : tmva_variables) float_store[var];
  for (auto var : tmva_spectators) float_store[var];
  float_store[tmva_output];
  map<string, Int_t> int_store;
  int_store[label];

  // Add variables to reader
  for (auto var : tmva_variables) tmva_reader->AddVariable(var.c_str(), &float_store[var]);
  for (auto var : tmva_spectators) tmva_reader->AddSpectator(var.c_str(), &float_store[var]);
  tmva_reader->BookMVA(tmva_output.c_str(), tmva_weights.c_str());

  //// Evaluate train sample
  //evaluate(tmva_reader, float_store, int_store, tmva_variables, tmva_spectators, tmva_output, label, input_train_root, input_train_tree, output_root);
  // Evaluate test sample
  evaluate(tmva_reader, float_store, int_store, tmva_variables, tmva_spectators, tmva_output, label, input_test_root, input_test_tree, output_root);

}
