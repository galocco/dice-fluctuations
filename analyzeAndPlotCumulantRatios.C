// ============================================================
// Cumulant post-processing and plotting pipeline
//
// Most credits go to Mario Ciacco.
//
// Reads raw accumulators (hRawAcc_* TH2D) written by
// computeMCCumulantsWithEfficiency.  The input file can be the
// direct output of one compute run, or the result of
//
//   hadd merged.root fOutAll_run1.root fOutAll_run2.root ...
//
// hadd sums TH2D bins, so the merged file is equivalent to a
// single run over all input events.
// ============================================================

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TGraphErrors.h>
#include <TCanvas.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TStyle.h>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <algorithm>
#include <memory>
#include <cmath>
#include <filesystem>

gErrorIgnoreLevel = kWarning;

static const double kBoxHalfWidth = 0.05; // half-width in x for ideal error boxes

// ─── Accumulator field layout ────────────────────────────────────────────────
// Must stay in sync with writeRawAccumulators in the compute macro.
static const int kNAccFields = 22;
// Y-bin → field name (for documentation only)
//  1: nEventsPerSample
//  2: qAcc_1_1_1
//  3: qAcc_2_1_1
//  4: qAcc_1_2_1
//  5: qAcc_1_2_2
//  6: qAcc_3_1_1
//  7: qAcc_1_1_1_x_1_2_1
//  8: qAcc_1_1_1_x_1_2_2
//  9: qAcc_1_3_2
// 10: qAcc_1_3_3
// 11: qAcc_4_1_1
// 12: qAcc_2_1_1_x_1_2_1
// 13: qAcc_2_1_1_x_1_2_2
// 14: qAcc_2_2_1
// 15: qAcc_2_2_2
// 16: qAcc_1_1_1_x_1_3_2
// 17: qAcc_1_1_1_x_1_3_3
// 18: qAcc_1_2_1_x_1_2_2
// 19: qAcc_1_4_1
// 20: qAcc_1_4_2
// 21: qAcc_1_4_3
// 22: qAcc_1_4_4

// ─── POD holding the cumulant ratios and errors extracted from one hRawAcc ───
struct CumulantResults {
  double k2k1{0}, k3k1{0}, k4k2{0};
  double errK2K1{0}, errK3K1{0}, errK4K2{0};    // subsample standard errors
  double errK2K1R{0}, errK3K1R{0}, errK4K2R{0}; // rescaled to targetEvents
  double nev{0};
  bool   valid{false};
};

TString getSpeciesLabel(int pdg)
{
  TString species;
  if (pdg == 3122) species = "#Lambda^{0}";
  else if (pdg == -3122) species = "#bar{#Lambda}^{0}";
  else if (pdg == 3212) species = "#Sigma^{-}";
  else if (pdg == -3212) species = "#Sigma^{+}";
  else if (pdg == 2212) species = "p";
  else if (pdg == -2212) species = "#bar{p}";
  else if (pdg == 0) species = "h^{+}-h^{-}";
  else species = Form("PDG%d", pdg);
  return species;
}

// ─── Read raw accumulators from hRawAcc_<label> and compute cumulants ────────
//
// Each X-bin of hRawAcc is one subsample; each Y-bin is one accumulator field.
// After hadd the bin contents are the *sum* over all merged files, which is
// exactly what we want before dividing by nEvents.
//
CumulantResults extractResults(TFile& f, const std::string& label,
                               double targetEvents = 4.5e9,
                               bool net = false)
{
  CumulantResults res;
  TH2D *h = (TH2D *)f.Get(Form("hRawAcc_%s", label.c_str()));
  if (!h) {
    std::cout << "Warning: hRawAcc_" << label << " not found, skipping." << std::endl;
    return res;
  }

  const int nSamples = h->GetNbinsX();

  double k2k1{0}, k3k1{0}, k4k2{0};
  double k2k1sq{0}, k3k1sq{0}, k4k2sq{0};
  int nUsed = 0;

  for (int s = 0; s < nSamples; ++s) {
    // Column s+1 of hRawAcc: read all fields
    double nEv = h->GetBinContent(s + 1,  1); // nEventsPerSample
    if (nEv < 1.) continue;

    double acc_1_1_1         = h->GetBinContent(s + 1,  2);
    double acc_2_1_1         = h->GetBinContent(s + 1,  3);
    double acc_1_2_1         = h->GetBinContent(s + 1,  4);
    double acc_1_2_2         = h->GetBinContent(s + 1,  5);
    double acc_3_1_1         = h->GetBinContent(s + 1,  6);
    double acc_1_1_1_x_1_2_1 = h->GetBinContent(s + 1,  7);
    double acc_1_1_1_x_1_2_2 = h->GetBinContent(s + 1,  8);
    double acc_1_3_2         = h->GetBinContent(s + 1,  9);
    double acc_1_3_3         = h->GetBinContent(s + 1, 10);
    double acc_4_1_1         = h->GetBinContent(s + 1, 11);
    double acc_2_1_1_x_1_2_1 = h->GetBinContent(s + 1, 12);
    double acc_2_1_1_x_1_2_2 = h->GetBinContent(s + 1, 13);
    double acc_2_2_1         = h->GetBinContent(s + 1, 14);
    double acc_2_2_2         = h->GetBinContent(s + 1, 15);
    double acc_1_1_1_x_1_3_2 = h->GetBinContent(s + 1, 16);
    double acc_1_1_1_x_1_3_3 = h->GetBinContent(s + 1, 17);
    double acc_1_2_1_x_1_2_2 = h->GetBinContent(s + 1, 18);
    double acc_1_4_1         = h->GetBinContent(s + 1, 19);
    double acc_1_4_2         = h->GetBinContent(s + 1, 20);
    double acc_1_4_3         = h->GetBinContent(s + 1, 21);
    double acc_1_4_4         = h->GetBinContent(s + 1, 22);

    // ── Raw moments (Eqs. 62–65) ─────────────────────────────────────────────
    double M1 = acc_1_1_1 / nEv;

    double M2 = (acc_2_1_1 + acc_1_2_1 - acc_1_2_2) / nEv;

    double M3 = (acc_3_1_1
                 + 3. * acc_1_1_1_x_1_2_1
                 - 3. * acc_1_1_1_x_1_2_2
                 +      acc_1_1_1
                 - 3. * acc_1_3_2
                 + 2. * acc_1_3_3) / nEv;

    double M4 = (acc_4_1_1
                 + 6. * acc_2_1_1_x_1_2_1
                 - 6. * acc_2_1_1_x_1_2_2
                 + 4. * acc_2_1_1
                 + 3. * acc_2_2_1
                 + 3. * acc_2_2_2
                 - 12.* acc_1_1_1_x_1_3_2
                 + 8. * acc_1_1_1_x_1_3_3
                 - 6. * acc_1_2_1_x_1_2_2
                 +      acc_1_4_1
                 - 7. * acc_1_4_2
                 + 12.* acc_1_4_3
                 - 6. * acc_1_4_4) / nEv;

    // ── Cumulants ────────────────────────────────────────────────────────────
    double C1 = M1;
    double C2 = M2 - M1 * M1;
    double C3 = M3 - 3. * M2 * M1 + 2. * M1 * M1 * M1;
    double C4 = M4 - 4. * M3 * M1 - 3. * M2 * M2
                   + 12. * M2 * M1 * M1 - 6. * M1 * M1 * M1 * M1;

    // <N+ + N-> per event
    double Ntot = acc_1_2_1 / nEv;

    if (C1 == 0.) continue;
    if (C2 == 0.) continue;
    if (net && Ntot == 0.) continue;

    const double firstRatio = net ? (C2 / Ntot) : (C2 / C1);
    const double r3k1 = C3 / C1;
    const double r4k2 = C4 / C2;

    res.nev += nEv;
    ++nUsed;

    k2k1   += firstRatio;   k2k1sq   += firstRatio * firstRatio;
    k3k1   += r3k1;         k3k1sq   += r3k1 * r3k1;
    k4k2   += r4k2;         k4k2sq   += r4k2 * r4k2;
  }

  if (nUsed == 0) {
    std::cout << "Warning: no valid subsamples for hRawAcc_" << label
              << ", skipping." << std::endl;
    return res;
  }

  k2k1 /= nUsed;   k3k1 /= nUsed;   k4k2 /= nUsed;
  k2k1sq /= nUsed; k3k1sq /= nUsed; k4k2sq /= nUsed;

  // Standard error of the mean: sqrt(Var / (N*(N-1)))
  auto sampleErr = [&](double meanSq, double mean) {
    if (nUsed < 2) return 0.0;
    return std::sqrt((meanSq - mean * mean) / (nUsed - 1));
  };

  res.k2k1    = k2k1;   res.k3k1    = k3k1;   res.k4k2    = k4k2;
  res.errK2K1 = sampleErr(k2k1sq, k2k1);
  res.errK3K1 = sampleErr(k3k1sq, k3k1);
  res.errK4K2 = sampleErr(k4k2sq, k4k2);

  // Scale factor: current stat → target stat (errors shrink as 1/sqrt(N))
  const double errScale = (targetEvents > 0.) ? std::sqrt(res.nev / targetEvents) : 0.;
  res.errK2K1R = res.errK2K1 * errScale;
  res.errK3K1R = res.errK3K1 * errScale;
  res.errK4K2R = res.errK4K2 * errScale;

  res.valid = true;
  return res;
}

// ─── One plot per label: ideal points + non-ideal rescaled error bars ─────────
// plotIdealErr: if true, overlay a second graph showing the ideal (unrescaled)
//               statistical errors from the ideal sample itself.
bool analyseLabel(TFile& f, const std::string& label, TFile& fOut,
                  double energy = 8.8, int pdg = 3122,
                  bool net = false, double targetEvents = 4.5e9,
                  bool plotIdealErr = false, TString dir = "./")
{
  const CumulantResults real  = extractResults(f, label, targetEvents, net);
  const CumulantResults ideal = extractResults(f, label + "_ideal", targetEvents, net);

  if (!real.valid || !ideal.valid) return false;

  // ── Console summary: ideal values with real rescaled errors ─────────────────
  const char* firstRatioLabel = net ? "k2/<Ntot>" : "k2/k1";
  std::cout << "\n=== " << label << " (ideal values, real errors) ===" << std::endl;
  std::cout << "N_ev  = " << ideal.nev << std::endl;
  std::cout << firstRatioLabel << " = " << ideal.k2k1
            << " +/- " << ideal.errK2K1
            << "  (rescaled: " << real.errK2K1R << ")"
            << (plotIdealErr ? Form("  (ideal stat: %g)", ideal.errK2K1R) : "")
            << std::endl;
  std::cout << "k3/k1 = " << ideal.k3k1
            << " +/- " << ideal.errK3K1
            << "  (rescaled: " << real.errK3K1R << ")"
            << (plotIdealErr ? Form("  (ideal stat: %g)", ideal.errK3K1R) : "")
            << std::endl;
  std::cout << "k4/k2 = " << ideal.k4k2
            << " +/- " << ideal.errK4K2
            << "  (rescaled: " << real.errK4K2R << ")"
            << (plotIdealErr ? Form("  (ideal stat: %g)", ideal.errK4K2R) : "")
            << std::endl;

  // ── Output histogram: ideal central values, real rescaled errors ───────────
  TH1D hCR(Form("hCRescaled_%s", label.c_str()), ";Order;#kappa", 3, 0, 3);
  hCR.SetBinContent(1, ideal.k2k1);  hCR.SetBinError(1, real.errK2K1R);
  hCR.SetBinContent(2, ideal.k3k1);  hCR.SetBinError(2, real.errK3K1R);
  hCR.SetBinContent(3, ideal.k4k2);  hCR.SetBinError(3, real.errK4K2R);

  // ── Canvas ─────────────────────────────────────────────────────────────────
  TCanvas cR(Form("cRescaled_%s", label.c_str()),
             Form("cRescaled_%s", label.c_str()), 800, 800);
  cR.SetLeftMargin(0.16);
  cR.SetBottomMargin(0.20);
  cR.SetTopMargin(0.04);
  cR.SetRightMargin(0.03);

  auto species = getSpeciesLabel(pdg);
  TH1D hFrame(Form("hFrame_%s", label.c_str()), ";;Cumulants ratio", 3, 0.5, 3.5);
  hFrame.SetMinimum(0.0);
  hFrame.SetMaximum(1.1);
  hFrame.GetXaxis()->SetBinLabel(1, "#frac{#kappa_{2}(" + species + ")}{<" + (pdg==0 ? "h^{+}+h^{-}" : species) + ">}");
  hFrame.GetXaxis()->SetBinLabel(2, "#frac{#kappa_{3}(" + species + ")}{<" + species + ">}");
  hFrame.GetXaxis()->SetBinLabel(3, "#frac{#kappa_{4}(" + species + ")}{#kappa_{2}(" + species + ")}");
  hFrame.GetXaxis()->SetLabelSize(0.065);
  hFrame.GetYaxis()->SetTitleSize(0.05);
  hFrame.GetYaxis()->SetLabelSize(0.04);

  double x[3]  = {1.0, 2.0, 3.0};
  double ex[3] = {0., 0., 0.};
  double y[3]  = {ideal.k2k1,    ideal.k3k1,    ideal.k4k2};
  double eyR[3] = {real.errK2K1R,  real.errK3K1R,  real.errK4K2R};  // rescaled real
  double eyI[3] = {ideal.errK2K1, ideal.errK3K1, ideal.errK4K2};    // ideal (unrescaled)
  double yPlusEyI[3] = {y[0] + eyI[0], y[1] + eyI[1], y[2] + eyI[2]};
  
  // maximum of y + ideal.err for setting the y-axis range
  double ymax = *std::max_element(yPlusEyI, yPlusEyI + 3);
  hFrame.GetYaxis()->SetRangeUser(0, ymax + 0.01);
  gStyle->SetOptStat(0);
  hFrame.Draw("AXIS");

  // Graph 1: ideal central values + rescaled real errors (classic error bars)
  TGraphErrors gR(3, x, y, ex, eyR);
  gR.SetMarkerStyle(20);
  gR.SetMarkerSize(1.0);
  gR.SetMarkerColor(kRed + 1);
  gR.SetLineColor(kRed + 1);
  gR.SetLineWidth(2);
  gR.Draw("P SAME");

  // Graph 2 (optional): ideal stat errors as boxes + markers on top
  std::unique_ptr<TGraphAsymmErrors> gIBox;
  std::unique_ptr<TGraphErrors>      gIMark;
  if (plotIdealErr) {
    double exBoxL[3] = {kBoxHalfWidth, kBoxHalfWidth, kBoxHalfWidth};
    double exBoxH[3] = {kBoxHalfWidth, kBoxHalfWidth, kBoxHalfWidth};
    gIBox = std::make_unique<TGraphAsymmErrors>(3, x, y, exBoxL, exBoxH, eyI, eyI);
    gIBox->SetFillColorAlpha(kBlue + 1, 0.25);
    gIBox->SetFillStyle(1001);
    gIBox->SetLineWidth(0);
    gIBox->Draw("2 SAME");
  }

  // Legend (only when both series are shown)
  std::unique_ptr<TLegend> leg;
  if (plotIdealErr) {
    leg = std::make_unique<TLegend>(0.19, 0.2, 0.62, 0.3);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.035);
    leg->AddEntry(&gR, "Expected statistical uncertainty", "lp");
    leg->AddEntry(gIBox.get(), "Uncertainty with the simulated data", "f");
    leg->Draw();
  }

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.04);
  latex.SetTextFont(42);
  latex.DrawLatex(0.19, 0.47,       "NA60+/DiCE Performance");
  latex.DrawLatex(0.19, 0.47-0.05,
      Form("Pb#font[122]{-}Pb,  #sqrt{s_{NN}} = %0.1f GeV, 0-5%%", energy));

  std::regex number_regex(R"([0-9]*\.?[0-9]+)");
  std::sregex_iterator it(label.begin(), label.end(), number_regex);
  std::sregex_iterator end;
  std::vector<double> numbers;
  for (; it != end; ++it)
    numbers.push_back(std::stod(it->str()));

  auto fmt = [](double x) -> const char* {
    if (std::fabs(x - std::round(x)) < 1e-6)
      return Form("%.0f", x);
    if (std::fabs(x * 10 - std::round(x * 10)) < 1e-6)
      return Form("%.1f", x);
    return Form("%.2f", x);
  };

  latex.DrawLatex(0.19, 0.47-0.10,
      Form("%s #leq #it{#eta} #leq %s",
          fmt(numbers[2]), fmt(numbers[3])));
  latex.DrawLatex(0.19, 0.47-0.15,
      Form("%s #leq #it{p}_{T} #leq %s GeV/#it{c}",
          fmt(numbers[0]), fmt(numbers[1])));

  cR.SaveAs(Form("%s/cumulants_rescaled_%s.pdf", dir.Data(), label.c_str()));
  cR.SaveAs(Form("%s/cumulants_rescaled_%s.png", dir.Data(), label.c_str()));

  fOut.cd();
  hCR.Write();
  cR.Write();

  return true;
}

// ─── Plot cumulant ratios vs eta upper cut, one canvas per pt range ───────────
void analyseVsEtaCuts(TFile& fIn, const std::vector<std::string>& labels,
                      TFile& fOut, double energy = 8.8,
                      int pdg = 3122, bool net = false,
                      double targetEvents = 4.5e9,
                      bool plotIdealErr = false, TString dir = "./")
{
  std::vector<std::string> ptTags;
  for (const auto& lbl : labels) {
    auto pos = lbl.find("_eta_");
    if (pos == std::string::npos) continue;
    std::string tag = lbl.substr(0, pos);
    if (std::find(ptTags.begin(), ptTags.end(), tag) == ptTags.end())
      ptTags.push_back(tag);
  }

  auto etaRange = [](const std::string& lbl) -> std::pair<double,double> {
    auto pos = lbl.find("_eta_");
    if (pos == std::string::npos) return {0,0};
    std::string etaPart = lbl.substr(pos + 5);
    auto us = etaPart.find('_');
    double lo = std::stod(etaPart.substr(0, us));
    double hi = std::stod(etaPart.substr(us + 1));
    return {lo, hi};
  };

  auto ptRange = [](const std::string& tag) -> std::pair<double,double> {
    std::string s = tag.substr(2);
    auto us = s.find('_');
    double lo = std::stod(s.substr(0, us));
    double hi = std::stod(s.substr(us + 1));
    return {lo, hi};
  };

  auto fmt = [](double v) -> const char* {
    if (std::fabs(v - std::round(v)) < 1e-6)        return Form("%.0f", v);
    if (std::fabs(v*10 - std::round(v*10)) < 1e-6)  return Form("%.1f", v);
    return Form("%.2f", v);
  };

  auto species = getSpeciesLabel(pdg);
  auto partSpecies = getSpeciesLabel(std::abs(pdg));
  auto antiSpecies = getSpeciesLabel(-std::abs(pdg));
  if (pdg == 0) {
    partSpecies = "h^{+}";
    antiSpecies = "h^{-}";
  }
  const std::string speciesStr = species.Data();
  const std::string partStr = partSpecies.Data();
  const std::string antiStr = antiSpecies.Data();
  const char* ratioNames[3]  = {"k2k1", "k3k1", "k4k2"};
  std::string ratioTitles[3];
  if (net) {
    ratioTitles[0] = "#kappa_{2}(" + partStr + "-" + antiStr + ")/<" + partStr + "+" + antiStr + ">";
    ratioTitles[1] = "#kappa_{3}(" + partStr + "-" + antiStr + ")/#kappa_{1}(" + partStr + "-" + antiStr + ")";
    ratioTitles[2] = "#kappa_{4}(" + partStr + "-" + antiStr + ")/#kappa_{2}(" + partStr + "-" + antiStr + ")";
  } else {
    ratioTitles[0] = "#kappa_{2}(" + speciesStr + ")/<" + speciesStr + ">";
    ratioTitles[1] = "#kappa_{3}(" + speciesStr + ")/<" + speciesStr + ">";
    ratioTitles[2] = "#kappa_{4}(" + speciesStr + ")/#kappa_{2}(" + speciesStr + ")";
  }

  for (const auto& tag : ptTags) {
    std::vector<std::pair<double,std::string>> etaLabels;
    for (const auto& lbl : labels) {
      if (lbl.find(tag + "_eta_") == 0) {
        auto [lo, hi] = etaRange(lbl);
        etaLabels.push_back({hi, lbl});
      }
    }
    std::sort(etaLabels.begin(), etaLabels.end());
    const int N = (int)etaLabels.size();
    if (N == 0) continue;

    std::vector<double> etaHi(N), ex(N, 0.);
    std::vector<double> yK2K1(N),  eyK2K1R(N),  eyK2K1I(N);
    std::vector<double> yK3K1(N),  eyK3K1R(N),  eyK3K1I(N);
    std::vector<double> yK4K2(N),  eyK4K2R(N),  eyK4K2I(N);

    double etaLo = 0.;
    for (int i = 0; i < N; ++i) {
      etaHi[i] = etaLabels[i].first;
      const auto& lbl = etaLabels[i].second;
      auto [lo, hi] = etaRange(lbl);
      etaLo = lo;

      const CumulantResults real  = extractResults(fIn, lbl, targetEvents, net);
      const CumulantResults ideal = extractResults(fIn, lbl + "_ideal", targetEvents, net);
      if (!real.valid || !ideal.valid) {
        yK2K1[i] = yK3K1[i] = yK4K2[i] = 0.;
        eyK2K1R[i] = eyK3K1R[i] = eyK4K2R[i] = 0.;
        eyK2K1I[i] = eyK3K1I[i] = eyK4K2I[i] = 0.;
        continue;
      }
      yK2K1[i]   = ideal.k2k1;    eyK2K1R[i]  = real.errK2K1R;  eyK2K1I[i]  = ideal.errK2K1;
      yK3K1[i]   = ideal.k3k1;    eyK3K1R[i]  = real.errK3K1R;  eyK3K1I[i]  = ideal.errK3K1;
      yK4K2[i]   = ideal.k4k2;    eyK4K2R[i]  = real.errK4K2R;  eyK4K2I[i]  = ideal.errK4K2;
    }

    auto [ptLo, ptHi] = ptRange(tag);

    double* yArr[3]   = {yK2K1.data(),   yK3K1.data(),   yK4K2.data()};
    double* eyArrR[3] = {eyK2K1R.data(), eyK3K1R.data(), eyK4K2R.data()};
    double* eyArrI[3] = {eyK2K1I.data(), eyK3K1I.data(), eyK4K2I.data()};

    double ymax = 0;
    for (int r = 0; r < 3; ++r) {
      for (int j = 0; j < N; ++j)
        ymax = std::max(ymax, yArr[r][j] + eyArrI[r][j]);
    }

    for (int r = 0; r < 3; ++r) {
      std::string cName = Form("cVsEta_%s_%s", tag.c_str(), ratioNames[r]);

      TCanvas c(cName.c_str(), cName.c_str(), 800, 700);
      c.SetLeftMargin(0.16);
      c.SetBottomMargin(0.13);
      c.SetTopMargin(0.05);
      c.SetRightMargin(0.04);

      double yminErr = yArr[r][0] - eyArrR[r][0];
      for (int j = 1; j < N; ++j)
        yminErr = std::min(yminErr, yArr[r][j] - eyArrR[r][j]);
      if (plotIdealErr)
        for (int j = 0; j < N; ++j)
          yminErr = std::min(yminErr, yArr[r][j] - eyArrI[r][j]);

      TH1D hF(Form("hFrameEta_%s_%s", tag.c_str(), ratioNames[r]),
              Form(";#eta_{max};%s", ratioTitles[r].c_str()),
              100,
              etaHi.front() - 0.1,
              etaHi.back()  + 0.1);

      float margin = 0.005;
      hF.SetMinimum(yminErr - margin);
      hF.SetMaximum(1 + margin);
      hF.GetXaxis()->SetTitleSize(0.05);
      hF.GetYaxis()->SetTitleSize(0.05);
      hF.GetXaxis()->SetLabelSize(0.04);
      hF.GetYaxis()->SetLabelSize(0.04);
      hF.GetYaxis()->SetRangeUser(0, ymax + 0.01);
      gStyle->SetOptStat(0);
      hF.Draw();

      // Rescaled real errors (classic error bars)
      TGraphErrors gR(N, etaHi.data(), yArr[r], ex.data(), eyArrR[r]);
      gR.SetMarkerStyle(20);
      gR.SetMarkerSize(1.2);
      gR.SetMarkerColor(kRed + 1);
      gR.SetLineColor(kRed + 1);
      gR.SetLineWidth(2);
      gR.Draw("P SAME");

      // Ideal errors (optional): box + markers on top
      std::unique_ptr<TGraphAsymmErrors> gIBox;
      std::unique_ptr<TGraphErrors>      gIMark;
      if (plotIdealErr) {
        std::vector<double> exBL(N, kBoxHalfWidth), exBH(N, kBoxHalfWidth);
        std::vector<double> exZ(N, 0.);
        gIBox = std::make_unique<TGraphAsymmErrors>(
            N, etaHi.data(), yArr[r],
            exBL.data(), exBH.data(), eyArrI[r], eyArrI[r]);
        gIBox->SetFillColorAlpha(kBlue + 1, 0.25);
        gIBox->SetFillStyle(1001);
        gIBox->SetLineWidth(0);
        gIBox->Draw("2 SAME");

        gIMark = std::make_unique<TGraphErrors>(
            N, etaHi.data(), yArr[r], exZ.data(), eyArrI[r]);
        gIMark->SetMarkerStyle(24);
        gIMark->SetMarkerSize(1.2);
        gIMark->SetMarkerColor(kBlue + 1);
        gIMark->SetLineColor(kBlue + 1);
        gIMark->SetLineWidth(2);
        gIMark->Draw("P SAME");
      }

      std::unique_ptr<TLegend> leg;
      if (plotIdealErr) {
        
        leg = std::make_unique<TLegend>(0.19, 0.5-0.05*2, 0.65, 0.5);
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->SetTextSize(0.033);
        leg->AddEntry(&gR, "Expected statistical uncertainty", "lp");
        leg->AddEntry(gIBox.get(), "Uncertainty with the simulated data", "f");
        leg->Draw();
      }

      TLatex latex;
      latex.SetNDC();
      latex.SetTextSize(0.038);
      latex.SetTextFont(42);
      double legOffset = plotIdealErr ? 0.14 : 0.;
      latex.DrawLatex(0.19, 0.47 - legOffset, "NA60+/DiCE Performance");
      latex.DrawLatex(0.19, 0.47 - 0.05 - legOffset,
          Form("Pb#font[122]{-}Pb,  #sqrt{s_{NN}} = %0.1f GeV, 0-5%%", energy));
      latex.DrawLatex(0.19, 0.47 - 0.10 - legOffset,
          Form("%s #leq #it{#eta} #leq #it{#eta}_{max}", fmt(etaLo)));
      latex.DrawLatex(0.19, 0.47 - 0.15 - legOffset,
          Form("%s #leq #it{p}_{T} #leq %s GeV/#it{c}", fmt(ptLo), fmt(ptHi)));

      c.SaveAs(Form("%s/vsEta_%s_%s.pdf", dir.Data(), tag.c_str(), ratioNames[r]));
      c.SaveAs(Form("%s/vsEta_%s_%s.png", dir.Data(), tag.c_str(), ratioNames[r]));

      fOut.cd();
      gR.Write(Form("gVsEta_%s_%s_real", tag.c_str(), ratioNames[r]));
      if (gIBox)  gIBox->Write(Form("gVsEta_%s_%s_idealBox",  tag.c_str(), ratioNames[r]));
      if (gIMark) gIMark->Write(Form("gVsEta_%s_%s_idealMark", tag.c_str(), ratioNames[r]));
      c.Write();
    }
  }
}

// ─── ki/kj vs eta_max overlaid for all pt selections ────────────────────────
void analyseVsEtaOverlay(TFile& fIn, const std::vector<std::string>& labels,
                         TFile& fOut, double energy = 8.8,
                         int pdg = 3122, bool net = false,
                         double targetEvents = 4.5e9,
                         bool plotIdealErr = false, TString dir = "./")
{
  auto etaRange = [](const std::string& lbl) -> std::pair<double,double> {
    auto pos = lbl.find("_eta_");
    if (pos == std::string::npos) return {0,0};
    std::string s = lbl.substr(pos + 5);
    auto us = s.find('_');
    return {std::stod(s.substr(0, us)), std::stod(s.substr(us + 1))};
  };
  auto ptRange = [](const std::string& tag) -> std::pair<double,double> {
    std::string s = tag.substr(2);
    auto us = s.find('_');
    return {std::stod(s.substr(0, us)), std::stod(s.substr(us + 1))};
  };
  auto fmt = [](double v) -> const char* {
    if (std::fabs(v - std::round(v)) < 1e-6)        return Form("%.0f", v);
    if (std::fabs(v*10 - std::round(v*10)) < 1e-6)  return Form("%.1f", v);
    return Form("%.2f", v);
  };

  auto species = getSpeciesLabel(pdg);
  auto partSpecies = getSpeciesLabel(std::abs(pdg));
  auto antiSpecies = getSpeciesLabel(-std::abs(pdg));
  if (pdg == 0) {
    partSpecies = "h^{+}";
    antiSpecies = "h^{-}";
  }
  const std::string speciesStr = species.Data();
  const std::string partStr = partSpecies.Data();
  const std::string antiStr = antiSpecies.Data();
  const char* ratioNames[3]  = {"k2k1", "k3k1", "k4k2"};
  std::string ratioTitles[3];
  if (net) {
    ratioTitles[0] = "#kappa_{2}(" + partStr + "-" + antiStr + ")/<" + partStr + "+" + antiStr + ">";
    ratioTitles[1] = "#kappa_{3}(" + partStr + "-" + antiStr + ")/#kappa_{1}(" + partStr + "-" + antiStr + ")";
    ratioTitles[2] = "#kappa_{4}(" + partStr + "-" + antiStr + ")/#kappa_{2}(" + partStr + "-" + antiStr + ")";
  } else {
    ratioTitles[0] = "#kappa_{2}(" + speciesStr + ")/<" + speciesStr + ">";
    ratioTitles[1] = "#kappa_{3}(" + speciesStr + ")/<" + speciesStr + ">";
    ratioTitles[2] = "#kappa_{4}(" + speciesStr + ")/#kappa_{2}(" + speciesStr + ")";
  }

  const int kColors[] = {kRed+1, kBlue+1, kGreen+2, kOrange+1,
                          kMagenta+1, kCyan+2, kViolet+1, kTeal+2};
  const int nColors = (int)(sizeof(kColors)/sizeof(kColors[0]));
  const int kMarkers[]     = {20, 21, 22, 23, 29, 33, 34, 47}; // filled
  const int kMarkersOpen[] = {24, 25, 26, 32, 30, 27, 28, 46}; // open counterparts

  std::vector<std::string> ptTags;
  for (const auto& lbl : labels) {
    auto pos = lbl.find("_eta_");
    if (pos == std::string::npos) continue;
    std::string tag = lbl.substr(0, pos);
    if (std::find(ptTags.begin(), ptTags.end(), tag) == ptTags.end())
      ptTags.push_back(tag);
  }

  std::vector<double> etaHiAll;
  for (const auto& lbl : labels) {
    double hi = etaRange(lbl).second;
    if (std::find(etaHiAll.begin(), etaHiAll.end(), hi) == etaHiAll.end())
      etaHiAll.push_back(hi);
  }
  std::sort(etaHiAll.begin(), etaHiAll.end());
  const int Neta = (int)etaHiAll.size();

  double etaLo = 0.;
  for (const auto& lbl : labels) { etaLo = etaRange(lbl).first; break; }

  const int Npt = (int)ptTags.size();
  // [pt][ratio][eta] for rescaled-real and ideal errors
  std::vector<std::vector<std::vector<double>>> yData(Npt,
      std::vector<std::vector<double>>(3, std::vector<double>(Neta, 0.)));
  std::vector<std::vector<std::vector<double>>> eyDataR(Npt,
      std::vector<std::vector<double>>(3, std::vector<double>(Neta, 0.)));
  std::vector<std::vector<std::vector<double>>> eyDataI(Npt,
      std::vector<std::vector<double>>(3, std::vector<double>(Neta, 0.)));

  double ymax = 0;
  for (int p = 0; p < Npt; ++p) {
    for (int e = 0; e < Neta; ++e) {
      std::string lbl;
      for (const auto& l : labels) {
        if (l.find(ptTags[p] + "_eta_") == 0 &&
            std::fabs(etaRange(l).second - etaHiAll[e]) < 1e-9) {
          lbl = l; break;
        }
      }
      if (lbl.empty()) continue;
      const CumulantResults real  = extractResults(fIn, lbl, targetEvents, net);
      const CumulantResults ideal = extractResults(fIn, lbl + "_ideal", targetEvents, net);
      if (!real.valid || !ideal.valid) continue;
      yData[p][0][e]  = ideal.k2k1;  eyDataR[p][0][e] = real.errK2K1R;  eyDataI[p][0][e] = ideal.errK2K1;
      yData[p][1][e]  = ideal.k3k1;  eyDataR[p][1][e] = real.errK3K1R;  eyDataI[p][1][e] = ideal.errK3K1;
      yData[p][2][e]  = ideal.k4k2;  eyDataR[p][2][e] = real.errK4K2R;  eyDataI[p][2][e] = ideal.errK4K2;
      ymax = std::max<double>(ymax, yData[p][0][e] + eyDataI[p][0][e]);
      ymax = std::max<double>(ymax, yData[p][1][e] + eyDataI[p][1][e]);
      ymax = std::max<double>(ymax, yData[p][2][e] + eyDataI[p][2][e]);
    }
  }

  std::vector<double> ex(Neta, 0.);

  for (int r = 0; r < 3; ++r) {
    double yminErr = 1e9;
    for (int p = 0; p < Npt; ++p)
      for (int e = 0; e < Neta; ++e) {
        yminErr = std::min(yminErr, yData[p][r][e] - eyDataR[p][r][e]);
        if (plotIdealErr)
          yminErr = std::min(yminErr, yData[p][r][e] - eyDataI[p][r][e]);
      }

    std::string cName = Form("cVsEtaOverlay_%s", ratioNames[r]);
    TCanvas c(cName.c_str(), cName.c_str(), 800, 700);
    c.SetLeftMargin(0.16);
    c.SetBottomMargin(0.13);
    c.SetTopMargin(0.05);
    c.SetRightMargin(0.04);

    TH1D hF(Form("hFrameOverlay_%s", ratioNames[r]),
          Form(";#eta_{max};%s", ratioTitles[r].c_str()),
            100,
            etaHiAll.front() - 0.1,
            etaHiAll.back()  + 0.1);
    hF.SetMinimum(yminErr - 0.2);
    hF.SetMaximum(1.005);
    hF.GetXaxis()->SetTitleSize(0.05);
    hF.GetYaxis()->SetTitleSize(0.05);
    hF.GetXaxis()->SetLabelSize(0.04);
    hF.GetYaxis()->SetLabelSize(0.04);
    hF.GetYaxis()->SetRangeUser(0, ymax + 0.01);
    gStyle->SetOptStat(0);
    hF.Draw("AXIS");

    std::vector<std::unique_ptr<TGraphErrors>>      graphsR(Npt);
    std::vector<std::unique_ptr<TGraphAsymmErrors>> graphsI(Npt);     // ideal error boxes
    std::vector<std::unique_ptr<TGraphErrors>>      graphsIMark(Npt); // ideal markers on top
    for (int p = 0; p < Npt; ++p) {
      std::vector<double> xs(Neta);
      for (int e = 0; e < Neta; ++e) xs[e] = etaHiAll[e];
      int col = kColors[p % nColors];

      // Rescaled real errors — classic error bars
      graphsR[p] = std::make_unique<TGraphErrors>(
          Neta, xs.data(), yData[p][r].data(),
          ex.data(), eyDataR[p][r].data());
      graphsR[p]->SetMarkerStyle(kMarkers[p % nColors]);
      graphsR[p]->SetMarkerSize(1.2);
      graphsR[p]->SetMarkerColor(col);
      graphsR[p]->SetLineColor(col);
      graphsR[p]->SetLineWidth(2);
      graphsR[p]->Draw("P SAME");

      // Ideal errors — box (TGraphAsymmErrors) + open markers on top
      if (plotIdealErr) {
        std::vector<double> exBL(Neta, kBoxHalfWidth), exBH(Neta, kBoxHalfWidth);
        std::vector<double> exZ(Neta, 0.);
        // box
        auto gIBox = std::make_unique<TGraphAsymmErrors>(
            Neta, xs.data(), yData[p][r].data(),
            exBL.data(), exBH.data(),
            eyDataI[p][r].data(), eyDataI[p][r].data());
        gIBox->SetFillColorAlpha(col, 0.20);
        gIBox->SetFillStyle(1001);
        gIBox->SetLineWidth(0);
        gIBox->Draw("2 SAME");
        // store box so it stays alive until canvas is saved
        graphsI[p] = std::move(gIBox);
      }
    }

    auto tmpGraphR = std::unique_ptr<TGraphErrors>(
        static_cast<TGraphErrors*>(graphsR[0]->Clone("tmpGraphR"))
    );

    tmpGraphR->SetMarkerColor(kGray + 1);
    tmpGraphR->SetLineColor(kGray + 1);
    auto tmpGraphI = std::unique_ptr<TGraphAsymmErrors>(
        static_cast<TGraphAsymmErrors*>(graphsI[0]->Clone("tmpGraphI"))
    );

    tmpGraphI->SetFillColorAlpha(kGray + 1, 0.25);
    tmpGraphI->SetFillStyle(1001);
    tmpGraphI->SetLineWidth(0);
    // Legend: one row per pt selection, filled+open if plotIdealErr
    double legY2 = 0.50;
    int nLegRows = plotIdealErr ? Npt + 2 : Npt; // +2 for error-type header entries
    TLegend leg(0.2, legY2 - 0.05 * nLegRows, 0.65, legY2);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.033);
    for (int p = 0; p < Npt; ++p) {
      auto [ptLo, ptHi] = ptRange(ptTags[p]);
      TString entryLabel = Form("%s #leq #it{p}_{T} #leq %s GeV/#it{c}", fmt(ptLo), fmt(ptHi));
      leg.AddEntry(graphsR[p].get(), entryLabel, "lp");
    }
    if (plotIdealErr) {
      // dummy entries explaining the two error styles
      leg.AddEntry(tmpGraphR.get(), "Expected statistical uncertainty", "lp");
      leg.AddEntry(tmpGraphI.get(), "Uncertainty with the simulated data", "f");
    }
    leg.Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.038);
    latex.SetTextFont(42);
    latex.DrawLatex(0.2+0.015, legY2 - 0.05*nLegRows - 0.03, "NA60+/DiCE Performance");
    latex.DrawLatex(0.2+0.015, legY2 - 0.05*nLegRows - 0.08,
        Form("Pb#font[122]{-}Pb,  #sqrt{s_{NN}} = %0.1f GeV, 0-5%%", energy));
    latex.DrawLatex(0.2+0.015, legY2 - 0.05*nLegRows - 0.13,
        Form("%s #leq #it{#eta} #leq #it{#eta}_{max}", fmt(etaLo)));

    c.SaveAs(Form("%s/vsEtaOverlay_%s.pdf", dir.Data(), ratioNames[r]));
    c.SaveAs(Form("%s/vsEtaOverlay_%s.png", dir.Data(), ratioNames[r]));

    fOut.cd();
    for (int p = 0; p < Npt; ++p) {
      graphsR[p]->Write(Form("gVsEtaOverlay_%s_%s_real",       ptTags[p].c_str(), ratioNames[r]));
      if (graphsI[p])
        graphsI[p]->Write(Form("gVsEtaOverlay_%s_%s_idealBox",   ptTags[p].c_str(), ratioNames[r]));
      if (graphsIMark[p])
        graphsIMark[p]->Write(Form("gVsEtaOverlay_%s_%s_idealMark", ptTags[p].c_str(), ratioNames[r]));
    }
    c.Write();
  }
}

// ─── Main entry point ─────────────────────────────────────────────────────────
void analyzeAndPlotCumulantRatios(std::vector<std::string> labels = {
                                          "pt0.2_2.5_eta_2_3.5",
                                          "pt0.2_2.5_eta_2_3",
                                          "pt0.2_2_eta_2_3.5",
                                          "pt0.2_2_eta_2_3",
                                        },
                                        double energy = 7.5,
                                        int pdg = 0,
                                        bool net = true,
                                        double targetEvents = 4.5e9,
                                        TString inputFile = "netCharge/Accumulators.root",
                                        TString outputFile = "netCharge/Cumulants.root",
                                        bool plotIdealErr = true
                                      )
{
  TFile fIn(inputFile);
  if (fIn.IsZombie()) {
    std::cout << "Error: cannot open " << inputFile << std::endl;
    return;
  }

  std::filesystem::path p(outputFile.Data());
  TString dir = "./";
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
    dir = p.parent_path().string();
  }
  TFile fOut(outputFile, "RECREATE");
  printf("dir = %s\n", dir.Data());

  for (const std::string& lbl : labels)
    analyseLabel(fIn, lbl, fOut, energy, pdg, net, targetEvents, plotIdealErr, dir);

  analyseVsEtaCuts(fIn, labels, fOut, energy, pdg, net, targetEvents, plotIdealErr, dir);
  analyseVsEtaOverlay(fIn, labels, fOut, energy, pdg, net, targetEvents, plotIdealErr, dir);

  fOut.Close();
}