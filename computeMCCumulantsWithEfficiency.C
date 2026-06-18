// ============================================================
// Cumulant MC processing and efficiency correction pipeline
//
// Most credits go to Mario Ciacco.
// ============================================================

#include <TEfficiency.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TLorentzVector.h>
#include <TDatabasePDG.h>
#include <TRandom3.h>
#include <TTree.h>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

// Number of subsamples used for statistical error estimation
const int nSamples = 50;
// Total number of events to process
int64_t nevents = 100000;

// ─── Cut set definition ───────────────────────────────────────────────────────
struct CutSet
{
  std::string label;
  double ptMin;
  double ptMax;
  double etaMin;
  double etaMax;
};

// ─── Per-cut-set accumulators ────────────────────────────────────────────────
struct Accumulators
{
  double nEventsPerSample[nSamples]{};

  // 1st order — Eq. (62)
  double qAcc_1_1_1[nSamples]{};

  // 2nd order — Eq. (63)
  double qAcc_2_1_1[nSamples]{};
  double qAcc_1_2_1[nSamples]{}; // <qTot(1)> = <N+ + N->  ← used also for Ntot
  double qAcc_1_2_2[nSamples]{};

  // 3rd order — Eq. (64)
  double qAcc_3_1_1[nSamples]{};
  double qAcc_1_1_1_x_1_2_1[nSamples]{};
  double qAcc_1_1_1_x_1_2_2[nSamples]{};
  double qAcc_1_3_2[nSamples]{};
  double qAcc_1_3_3[nSamples]{};

  // 4th order — Eq. (65)
  double qAcc_4_1_1[nSamples]{};
  double qAcc_2_1_1_x_1_2_1[nSamples]{};
  double qAcc_2_1_1_x_1_2_2[nSamples]{};
  double qAcc_2_2_1[nSamples]{};
  double qAcc_2_2_2[nSamples]{};
  double qAcc_1_1_1_x_1_3_2[nSamples]{};
  double qAcc_1_1_1_x_1_3_3[nSamples]{};
  double qAcc_1_2_1_x_1_2_2[nSamples]{};
  double qAcc_1_4_1[nSamples]{};
  double qAcc_1_4_2[nSamples]{};
  double qAcc_1_4_3[nSamples]{};
  double qAcc_1_4_4[nSamples]{};
};

// ─── Helper utilities ─────────────────────────────────────────────────────────
std::vector<double> buildTargetProbabilities(int nTargets)
{
  std::vector<double> probs;
  if (nTargets <= 0)
    return probs;

  std::vector<double> lOverLambda(nTargets, 0.15);
  probs.assign(nTargets, 0.0);
  double survival = 1.0;
  for (int i = 0; i < nTargets; ++i)
  {
    double l2l = lOverLambda[i];
    if (l2l < 0.)
      l2l = 0.;
    const double pInteract = 1.0 - std::exp(-l2l);
    probs[i] = survival * pInteract;
    survival *= std::exp(-l2l);
  }
  double sum = 0.;
  for (double p : probs)
    sum += p;
  if (sum <= 0.)
  {
    probs.assign(nTargets, 1.0 / nTargets);
    return probs;
  }
  for (double &p : probs)
    p /= sum;
  return probs;
}

int getTarget(const std::vector<double> &probs)
{
  if (probs.empty())
    return 0;
  double r = gRandom->Rndm(), cum = 0.;
  for (size_t i = 0; i < probs.size(); ++i)
  {
    cum += probs[i];
    if (r <= cum)
      return static_cast<int>(i);
  }
  return static_cast<int>(probs.size()) - 1;
}

// ─── Write one hQ histogram from a filled Accumulators object ────────────────
//
// Bin layout (Y axis):
//   bin 1 → nEvents
//   bin 2 → C1  (= κ₁ = <N+ - N->)
//   bin 3 → C2  (= κ₂)
//   bin 4 → C3  (= κ₃)
//   bin 5 → C4  (= κ₄)
//   bin 6 → <N+ + N->  (= qTot(1), needed for K2/<Ntot> normalisation)
//
void writeAccumulators(const Accumulators &acc, const std::string &label, TFile *fOut)
{
  TH2D hQ(Form("hQ_%s", label.c_str()),
          Form("%s;sample;order;Q_{n}", label.c_str()),
          nSamples, 0, nSamples, 6, 0, 6); // 6 bins on Y

  for (int s = 0; s < nSamples; ++s)
  {
    double nEv = acc.nEventsPerSample[s];
    if (nEv < 1.)
      continue;

    // Eq. (62): first raw moment
    double M1 = acc.qAcc_1_1_1[s] / nEv;

    // Eq. (63): second raw moment
    double M2 = (acc.qAcc_2_1_1[s] + acc.qAcc_1_2_1[s] - acc.qAcc_1_2_2[s]) / nEv;

    // Eq. (64): third raw moment
    double M3 = (acc.qAcc_3_1_1[s] + 3. * acc.qAcc_1_1_1_x_1_2_1[s] - 3. * acc.qAcc_1_1_1_x_1_2_2[s] + acc.qAcc_1_1_1[s] - 3. * acc.qAcc_1_3_2[s] + 2. * acc.qAcc_1_3_3[s]) / nEv;

    // Eq. (65): fourth raw moment
    double M4 = (acc.qAcc_4_1_1[s] + 6. * acc.qAcc_2_1_1_x_1_2_1[s] - 6. * acc.qAcc_2_1_1_x_1_2_2[s] + 4. * acc.qAcc_2_1_1[s] + 3. * acc.qAcc_2_2_1[s] + 3. * acc.qAcc_2_2_2[s] - 12. * acc.qAcc_1_1_1_x_1_3_2[s] + 8. * acc.qAcc_1_1_1_x_1_3_3[s] - 6. * acc.qAcc_1_2_1_x_1_2_2[s] + acc.qAcc_1_4_1[s] - 7. * acc.qAcc_1_4_2[s] + 12. * acc.qAcc_1_4_3[s] - 6. * acc.qAcc_1_4_4[s]) / nEv;

    // Raw moments → cumulants
    double C1 = M1;
    double C2 = M2 - M1 * M1;
    double C3 = M3 - 3. * M2 * M1 + 2. * M1 * M1 * M1;
    double C4 = M4 - 4. * M3 * M1 - 3. * M2 * M2 + 12. * M2 * M1 * M1 - 6. * M1 * M1 * M1 * M1;

    // <N+ + N-> per event = qTot(1) / nEv
    // qAcc_1_2_1 accumulates qTot(1) = N+/eff^1 + N-/eff^1 per event.
    // For the ideal slot (eff=1) this is exactly <N+> + <N->.
    // For the real slot it is the efficiency-weighted sum, consistent with
    // the rest of the correction scheme.
    double Ntot = acc.qAcc_1_2_1[s] / nEv;

    hQ.SetBinContent(s + 1, 1, nEv);
    hQ.SetBinContent(s + 1, 2, C1);
    hQ.SetBinContent(s + 1, 3, C2);
    hQ.SetBinContent(s + 1, 4, C3);
    hQ.SetBinContent(s + 1, 5, C4);
    hQ.SetBinContent(s + 1, 6, Ntot); // <N+ + N->
  }

  fOut->cd();
  hQ.Write();
}

// ─── Main macro ───────────────────────────────────────────────────────────────
void computeMCCumulantsWithEfficiency(std::vector<CutSet> cutSets = {
                                          //{"pt0.2_2_eta_2_4", 0.2, 2.0, 2.0, 4.0}, {"pt0.5_2_eta_2_4", 0.5, 2.0, 2.0, 4.0}, {"pt0.75_2_eta_2_4", 0.75, 2.0, 2.0, 4.0}, {"pt1_2_eta_2_4", 1.0, 2.0, 2.0, 4.0}, {"pt0.2_2_eta_2_3", 0.2, 2.0, 2.0, 3.0}, {"pt0.5_2_eta_2_3", 0.5, 2.0, 2.0, 3.0}, {"pt0.75_2_eta_2_3", 0.75, 2.0, 2.0, 3.0}, {"pt1_2_eta_2_3", 1.0, 2.0, 2.0, 3.0}, {"pt0.2_2_eta_2_3.5", 0.2, 2.0, 2.0, 3.5}, {"pt0.5_2_eta_2_3.5", 0.5, 2.0, 2.0, 3.5}, {"pt0.75_2_eta_2_3.5", 0.75, 2.0, 2.0, 3.5}, {"pt1_2_eta_2_3.5", 1.0, 2.0, 2.0, 3.5}, {"pt0.2_2_eta_2_2.5", 0.2, 2.0, 2.0, 2.5}, {"pt0.5_2_eta_2_2.5", 0.5, 2.0, 2.0, 2.5}, {"pt0.75_2_eta_2_2.5", 0.75, 2.0, 2.0, 2.5}, {"pt1_2_eta_2_2.5", 1.0, 2.0, 2.0, 2.5}},
                                          {"pt0.2_2.5_eta_2_3.5", 0.2, 2.5, 2.0, 3.5}},
                                      TString inputFilePath = "/data/galocco/output_PbPb.7.5.C0-5-netCharge.root", TString efficiencyFilePath = "/data/galocco/TheFIST_PbPb.7.5.C0-5-Lambda/ChargeEfficiency.root", int pdgToSelect = 0, bool doNet = true, float branchingRatio = 0.639)
{
  if (cutSets.empty())
  {
    std::cout << "No cut sets provided, nothing to do." << std::endl;
    return;
  }
  const int nCuts = static_cast<int>(cutSets.size());

  TString histEff3DPart = "hEff3D";
  TString histEff3DAntiPart = "hEffAnti3D";
  if (pdgToSelect == 0)
  {
    branchingRatio = 1.0;
    histEff3DPart = "hEff3DPos";
    histEff3DAntiPart = "hEff3DNeg";
  }

  TFile *filPow = TFile::Open(inputFilePath, "READ");
  if (!filPow || filPow->TestBit(TFile::kZombie))
  {
    std::cout << "No file or zombie!" << std::endl;
    return;
  }

  std::vector<TH2D *> hPY(nCuts), hPYr(nCuts);
  for (int ic = 0; ic < nCuts; ++ic)
  {
    const std::string &lbl = cutSets[ic].label;
    hPY[ic] = new TH2D(Form("hPY_%s", lbl.c_str()),
                       Form("Generated (%s);#it{p} (GeV/#it{c});y;Entries", lbl.c_str()),
                       100, 0, 50, 150, -5, 10);
    hPYr[ic] = new TH2D(Form("hPYr_%s", lbl.c_str()),
                        Form("Reconstructed (%s);#it{p} (GeV/#it{c});y;Entries", lbl.c_str()),
                        100, 0, 50, 150, -5, 10);
  }
  TH1D hT("hT", "Generated target, all samples;target;Entries", 10, -0.5, 9.5);

  TTree *kineTree = (TTree *)filPow->Get("kinematics");
  if (!kineTree)
  {
    std::cout << "No tree!" << std::endl;
    return;
  }

  int pdg{0}, event{0};
  float en{0.f}, px{0.f}, py{0.f}, pz{0.f};
  kineTree->SetBranchAddress("event", &event);
  kineTree->SetBranchAddress("pdg", &pdg);
  kineTree->SetBranchAddress("E", &en);
  kineTree->SetBranchAddress("px", &px);
  kineTree->SetBranchAddress("py", &py);
  kineTree->SetBranchAddress("pz", &pz);

  TFile fmc(efficiencyFilePath);
  std::cout << "Reading efficiency maps from " << efficiencyFilePath << std::endl;
  std::cout << "Efficiencies map for particles:      " << histEff3DPart << std::endl;
  std::cout << "Efficiencies map for anti-particles: " << histEff3DAntiPart << std::endl;
  TH3F *effMapPart = (TH3F *)fmc.Get(histEff3DPart);
  TH3F *effMapAntiPart = (TH3F *)fmc.Get(histEff3DAntiPart);
  if (!effMapAntiPart)
  {
    std::cout << "Warning: no anti-particle efficiency map found, using particle map." << std::endl;
    effMapAntiPart = effMapPart;
  }
  if (!effMapPart)
  {
    std::cout << "Error: " << histEff3DPart << " not found in " << efficiencyFilePath << std::endl;
    return;
  }

  const int nTargets = effMapPart->GetZaxis()->GetNbins();
  std::vector<double> targetProbabilities = buildTargetProbabilities(nTargets);

  const int nSlots = 2 * nCuts;
  std::vector<Accumulators> acc(nSlots);

  int64_t inputEntry{0};
  bool newEvent{false};
  int targetBinIdx{0};
  int ev_counter = 0;

  auto ipow = [](double base, int exp) -> double
  {
    double r = 1.;
    for (int i = 0; i < exp; ++i)
      r *= base;
    return r;
  };

  std::vector<std::array<double, 4>> qP(nSlots), qA(nSlots);
  int curSample = 0;

  auto flushSlot = [&](int slot)
  {
    auto qNet = [&](int c)
    { return qP[slot][c - 1] - qA[slot][c - 1]; };
    auto qTot = [&](int c)
    { return qP[slot][c - 1] + qA[slot][c - 1]; };
    Accumulators &a = acc[slot];
    int s = curSample;

    a.qAcc_1_1_1[s] += qNet(1);

    a.qAcc_2_1_1[s] += ipow(qNet(1), 2);
    a.qAcc_1_2_1[s] += qTot(1);
    a.qAcc_1_2_2[s] += qTot(2);

    a.qAcc_3_1_1[s] += ipow(qNet(1), 3);
    a.qAcc_1_1_1_x_1_2_1[s] += qNet(1) * qTot(1);
    a.qAcc_1_1_1_x_1_2_2[s] += qNet(1) * qTot(2);
    a.qAcc_1_3_2[s] += qNet(2);
    a.qAcc_1_3_3[s] += qNet(3);

    a.qAcc_4_1_1[s] += ipow(qNet(1), 4);
    a.qAcc_2_1_1_x_1_2_1[s] += ipow(qNet(1), 2) * qTot(1);
    a.qAcc_2_1_1_x_1_2_2[s] += ipow(qNet(1), 2) * qTot(2);
    a.qAcc_2_2_1[s] += ipow(qTot(1), 2);
    a.qAcc_2_2_2[s] += ipow(qTot(2), 2);
    a.qAcc_1_1_1_x_1_3_2[s] += qNet(1) * qNet(2);
    a.qAcc_1_1_1_x_1_3_3[s] += qNet(1) * qNet(3);
    a.qAcc_1_2_1_x_1_2_2[s] += qTot(1) * qTot(2);
    a.qAcc_1_4_1[s] += qTot(1);
    a.qAcc_1_4_2[s] += qTot(2);
    a.qAcc_1_4_3[s] += qTot(3);
    a.qAcc_1_4_4[s] += qTot(4);
  };

  auto flushEvent = [&]()
  {
    for (int slot = 0; slot < nSlots; ++slot)
    {
      acc[slot].nEventsPerSample[curSample] += 1;
      flushSlot(slot);
      qP[slot].fill(0.);
      qA[slot].fill(0.);
    }
  };

  std::cout << "kineTree->GetEntries(): " << kineTree->GetEntries() << "\n";
  std::cout << "Processing " << nCuts << " cut set(s) + ideal variants.\n";

  std::unordered_map<int, int> chargeCache;
  auto getCharge = [&](int pdg) -> int
  {
    auto it = chargeCache.find(pdg);
    if (it != chargeCache.end())
      return it->second;
    auto *p = TDatabasePDG::Instance()->GetParticle(pdg);
    int ch = p ? (int)p->Charge() : 0;
    chargeCache[pdg] = ch;
    return ch;
  };
  for (int64_t iev = 0; iev < nevents;)
  {
    targetBinIdx = getTarget(targetProbabilities);
    hT.Fill(targetBinIdx);
    curSample = static_cast<int>(gRandom->Rndm() * nSamples);

    for (int slot = 0; slot < nSlots; ++slot)
    {
      qP[slot].fill(0.);
      qA[slot].fill(0.);
    }

    int readError = 0;

    if (iev % 10000 == 0)
      std::cout << "Processing event " << iev
                << " tot events: " << ev_counter << "\n"
                << std::flush;

    while (inputEntry < kineTree->GetEntries())
    {
      if (newEvent)
      {
        newEvent = (event != iev);
      }
      else
      {
        if (kineTree->GetEntry(inputEntry++) < 0)
        {
          readError = -999;
          break;
        }
      }

      if (event != iev)
      {
        flushEvent();
        newEvent = true;
        iev = event;
        ev_counter++;
        break;
      }

      bool isPart = (pdg == pdgToSelect);
      bool isAntiPart = (pdg == -pdgToSelect) && doNet;

      if (pdgToSelect == 0)
      {
        int charge = getCharge(pdg);
        if (charge == 0)
          continue;
        isPart = charge > 0;
        isAntiPart = charge < 0 && doNet;
      }

      if (!isPart && !isAntiPart)
        continue;

      TLorentzVector mom(px, py, pz, en);
      const double rap = mom.Rapidity();
      const double p = mom.P();
      const double pt = mom.Pt();
      const double eta = mom.PseudoRapidity();

      TH3F *effMapTmp = isPart ? effMapPart : effMapAntiPart;
      const int etaBin = effMapTmp->GetXaxis()->FindBin(eta);
      const int ptBin = effMapTmp->GetYaxis()->FindBin(pt);
      const int tgtBin = targetBinIdx + 1;
      const double effReal = branchingRatio * effMapTmp->GetBinContent(etaBin, ptBin, tgtBin);

      for (int ic = 0; ic < nCuts; ++ic)
      {
        const CutSet &cs = cutSets[ic];

        if (pt < cs.ptMin || pt > cs.ptMax)
          continue;
        if (eta < cs.etaMin || eta > cs.etaMax)
          continue;

        hPY[ic]->Fill(p, rap);

        // Real efficiency
        {
          const int slot = ic;
          if (gRandom->Rndm() < effReal)
          {
            hPYr[ic]->Fill(p, rap);
            double *qArr = isPart ? qP[slot].data() : qA[slot].data();
            double invEff = 1.0 / effReal;
            double w = invEff;
            for (int c = 0; c < 4; ++c)
            {
              qArr[c] += w;
              w *= invEff;
            }
          }
        }

        // Ideal efficiency (eff = 1)
        {
          const int slot = nCuts + ic;
          double *qArr = isPart ? qP[slot].data() : qA[slot].data();
          for (int c = 0; c < 4; ++c)
            qArr[c] += 1.;
        }
      }
    } // end particle loop

    if (readError < -998)
    {
      std::cout << "Corrupted file!" << std::endl;
      return;
    }

    if (inputEntry >= kineTree->GetEntries())
    {
      flushEvent();
      std::cout << "Exiting at event " << iev
                << " as we ran out of input events." << std::endl;
      break;
    }
  } // end event loop

  TFile *fOut = TFile::Open("fOutAll.root", "RECREATE");

  for (int ic = 0; ic < nCuts; ++ic)
  {
    writeAccumulators(acc[ic], cutSets[ic].label, fOut);
    writeAccumulators(acc[nCuts + ic], cutSets[ic].label + "_ideal", fOut);
    fOut->cd();
    hPY[ic]->Write();
    hPYr[ic]->Write();
  }

  hT.Write();
  fOut->Close();

  for (int ic = 0; ic < nCuts; ++ic)
  {
    delete hPY[ic];
    delete hPYr[ic];
  }
}