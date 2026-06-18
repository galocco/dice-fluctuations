#include <TFile.h>
#include <TLorentzVector.h>
#include <TTree.h>

#include <glob.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

//---------------------------------------------
// Utility: expand wildcard pattern
//---------------------------------------------
std::vector<std::string> getFiles(const std::string& pattern) {
  glob_t glob_result;
  std::vector<std::string> files;

  glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);

  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    files.emplace_back(glob_result.gl_pathv[i]);
  }

  globfree(&glob_result);
  return files;
}
// list of PDG codes to reject (for pdgToSelect = 0)
const std::vector<int> rejectList = {22, // photon
                                      111, // pi0
                                      130, // K0L
                                      2112, // neutron
                                      -2112, // anti-neutron
                                    }; // proton and anti-proton

//---------------------------------------------
// Utility: check if PDG is in reject list
//---------------------------------------------
bool isRejected(int pdg, const std::vector<int>& rejectList) {
  for (int rejected : rejectList) {
    if (std::abs(pdg) == rejected) {
      return true;
    }
  }
  return false;
}

//---------------------------------------------
// Main parser
//---------------------------------------------
void parser(
    std::string pattern = "/home/galocco/TheFIST_PbPb_8.77GeV_HEP/job_1/PbPb.8.77.C0-5.dat",
    int pdgToSelect = 3122 // Default: select Lambda (PDG 3122), 0 means select all but those in the reject list
  ) { 

  // Output
  TFile outputFile("output.root", "recreate");
  TTree tree("kinematics", "kinematics");

  int event = -1;
  int pdg = 0;
  float px = 0.f, py = 0.f, pz = 0.f, E = 0.f;

  tree.Branch("event", &event);
  tree.Branch("pdg", &pdg);
  tree.Branch("px", &px);
  tree.Branch("py", &py);
  tree.Branch("pz", &pz);
  tree.Branch("E", &E);
  // Get all files
  auto files = getFiles(pattern);

  std::cout << "Found " << files.size() << " files" << std::endl;

  //---------------------------------------------
  // Loop over files
  //---------------------------------------------
  for (const auto& fileName : files) {

    std::cout << "Processing: " << fileName << std::endl;

    std::ifstream inFile(fileName);
    if (!inFile.is_open()) {
      std::cerr << "Cannot open file: " << fileName << std::endl;
      continue;
    }

    std::string line;

    while (std::getline(inFile, line)) {

      if (line.empty()) continue;

      // Event line
      if (line[0] == 'E') {
        event++;
        continue;
      }

      // Only particle lines
      if (line[0] != 'P') continue;

      std::istringstream split(line);

      char type;
      int barcode, parent, status;
      double mass;

      split >> type >> barcode >> parent >> pdg
            >> px >> py >> pz >> E >> mass >> status;

      TLorentzVector mom(px, py, pz, E);

      double rap = mom.Rapidity();
      double p = mom.P();

      bool rejected = isRejected(pdg, rejectList);
      // Selection
      if (std::abs(pdg) == pdgToSelect || pdgToSelect == 0 && !rejected) {
        tree.Fill();
      }
    }
  }

  //---------------------------------------------
  // Write output
  //---------------------------------------------
  outputFile.cd();
  tree.Write();
  outputFile.Close();

  std::cout << "Done. Output written to output.root" << std::endl;
}