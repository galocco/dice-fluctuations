// ============================================================
// Cumulant post-processing and plotting pipeline
//
// Most credits go to Mario Ciacco.
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
gErrorIgnoreLevel = kWarning;
// ─── POD holding the cumulant ratios and errors extracted from one hQ ─────────
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

// ─── Extract results from hQ_<label> ─────────────────────────────────────────
CumulantResults extractResults(TFile& f, const std::string& label,
                               double targetEvents = 4.5e9,
                               bool net = false)
{
  CumulantResults res;
  TH2D *h = (TH2D *)f.Get(Form("hQ_%s", label.c_str()));
  if (!h) {
    std::cout << "Warning: hQ_" << label << " not found, skipping." << std::endl;
    return res;
  }

  double k1{0}, k2{0}, k3{0}, k4{0};
  double k2k1{0}, k3k1{0}, k4k2{0};
  double k1sq{0}, k2sq{0}, k3sq{0}, k4sq{0};
  double k2k1sq{0}, k3k1sq{0}, k4k2sq{0};

  const int nSamples = h->GetNbinsX();
  int nUsed = 0;

  for (int i = 0; i < nSamples; ++i) {
    double e = h->GetBinContent(i + 1, 1);
    if (e <= 0.) continue;

    double k1_tmp = h->GetBinContent(i + 1, 2) / e;
    double k2_tmp = h->GetBinContent(i + 1, 3) / e;
    double k3_tmp = h->GetBinContent(i + 1, 4) / e;
    double k4_tmp = h->GetBinContent(i + 1, 5) / e;
    double ntot_tmp = h->GetBinContent(i + 1, 6) / e;

    if (k1_tmp == 0.) continue;
    if (k2_tmp == 0.) continue;
    if (net && ntot_tmp == 0.) continue;

    const double firstRatio = net ? (k2_tmp / ntot_tmp) : (k2_tmp / k1_tmp);

    res.nev += e;
    ++nUsed;
    k1   += k1_tmp;    k2   += k2_tmp;
    k3   += k3_tmp;    k4   += k4_tmp;
    k2k1 += firstRatio;
    k3k1 += k3_tmp / k1_tmp;
    k4k2 += k4_tmp / k2_tmp;

    k1sq   += k1_tmp * k1_tmp;   k2sq   += k2_tmp * k2_tmp;
    k3sq   += k3_tmp * k3_tmp;   k4sq   += k4_tmp * k4_tmp;
    k2k1sq += firstRatio * firstRatio;
    k3k1sq += (k3_tmp / k1_tmp) * (k3_tmp / k1_tmp);
    k4k2sq += (k4_tmp / k2_tmp) * (k4_tmp / k2_tmp);
  }

  if (nUsed == 0) {
    std::cout << "Warning: no valid subsamples for hQ_" << label << ", skipping." << std::endl;
    return res;
  }

  k1   /= nUsed;   k2   /= nUsed;
  k3   /= nUsed;   k4   /= nUsed;
  k2k1 /= nUsed;   k3k1 /= nUsed;   k4k2 /= nUsed;

  k1sq   /= nUsed;   k2sq   /= nUsed;
  k3sq   /= nUsed;   k4sq   /= nUsed;
  k2k1sq /= nUsed;   k3k1sq /= nUsed;   k4k2sq /= nUsed;

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
bool analyseLabel(TFile& f, const std::string& label, TFile& fOut, double energy = 8.8, int pdg = 3122, bool net = false, double targetEvents = 4.5e9)
{
  const CumulantResults real  = extractResults(f, label, targetEvents, net);
  const CumulantResults ideal = extractResults(f, label + "_ideal", targetEvents, net);

  if (!real.valid || !ideal.valid) return false;

  // ── Console summary: ideal values with real rescaled errors ─────────────────
  const char* firstRatioLabel = net ? "k2/<Ntot>" : "k2/k1";
  std::cout << "\n=== " << label << " (ideal values, real errors) ===" << std::endl;
  std::cout << "N_ev  = " << ideal.nev << std::endl;
  std::cout << firstRatioLabel << " = " << ideal.k2k1 << " +/- " << ideal.errK2K1
            << "  (rescaled: " << real.errK2K1R << ")" << std::endl;
  std::cout << "k3/k1 = " << ideal.k3k1 << " +/- " << ideal.errK3K1
            << "  (rescaled: " << real.errK3K1R << ")" << std::endl;
  std::cout << "k4/k2 = " << ideal.k4k2 << " +/- " << ideal.errK4K2
            << "  (rescaled: " << real.errK4K2R << ")" << std::endl;

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
  gStyle->SetOptStat(0);
  hFrame.Draw("AXIS");

  // Points: ideal efficiency values (unbiased by detector)
  // Error bars: real efficiency subsample errors rescaled to target statistics
  double x[3]  = {1.0, 2.0, 3.0};
  double ex[3] = {0., 0., 0.};
  double y[3]  = {ideal.k2k1,    ideal.k3k1,    ideal.k4k2};
  double ey[3] = {real.errK2K1R, real.errK3K1R, real.errK4K2R};

  TGraphErrors gR(3, x, y, ex, ey);
  gR.SetMarkerStyle(20);
  gR.SetMarkerSize(1.0);
  gR.SetMarkerColor(kRed + 1);
  gR.SetLineColor(kRed + 1);
  gR.SetLineWidth(2);
  gR.Draw("P SAME");

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

  for (; it != end; ++it) {
      numbers.push_back(std::stod(it->str()));
  }

  auto fmt = [](double x) {
      // se è praticamente intero → niente decimali
      if (std::fabs(x - std::round(x)) < 1e-6)
          return Form("%.0f", x);
      // se ha una sola cifra significativa → 1 decimale
      if (std::fabs(x * 10 - std::round(x * 10)) < 1e-6)
          return Form("%.1f", x);
      // altrimenti → 2 decimali
      return Form("%.2f", x);
  };
  latex.DrawLatex(0.19, 0.47-0.10,
      Form("%s #leq #it{#eta} #leq %s",
          fmt(numbers[2]), fmt(numbers[3])));

  latex.DrawLatex(0.19, 0.47-0.15,
      Form("%s #leq #it{p}_{T} #leq %s GeV/#it{c}",
          fmt(numbers[0]), fmt(numbers[1])));

  cR.SaveAs(Form("cumulants_rescaled_%s.pdf", label.c_str()));
  cR.SaveAs(Form("cumulants_rescaled_%s.png", label.c_str()));

  fOut.cd();
  hCR.Write();
  cR.Write();

  return true;
}

// ─── Plot cumulant ratios vs eta upper cut, one canvas per pt range ───────────
void analyseVsEtaCuts(TFile& fIn, const std::vector<std::string>& labels,
                      TFile& fOut, double energy = 8.8,
                      int pdg = 3122, bool net = false,
                      double targetEvents = 4.5e9)
{
  // Collect unique pt-range tags (everything before "_eta_")
  std::vector<std::string> ptTags;
  for (const auto& lbl : labels) {
    auto pos = lbl.find("_eta_");
    if (pos == std::string::npos) continue;
    std::string tag = lbl.substr(0, pos);
    if (std::find(ptTags.begin(), ptTags.end(), tag) == ptTags.end())
      ptTags.push_back(tag);
  }

  // Helper: parse the two eta numbers from a label like "pt0.2_2_eta_2_3.5"
  auto etaRange = [](const std::string& lbl) -> std::pair<double,double> {
    auto pos = lbl.find("_eta_");
    if (pos == std::string::npos) return {0,0};
    std::string etaPart = lbl.substr(pos + 5); // e.g. "2_3.5"
    auto us = etaPart.find('_');
    double lo = std::stod(etaPart.substr(0, us));
    double hi = std::stod(etaPart.substr(us + 1));
    return {lo, hi};
  };

  // Helper: parse the two pt numbers from the tag like "pt0.2_2"
  auto ptRange = [](const std::string& tag) -> std::pair<double,double> {
    // tag = "pt<lo>_<hi>"
    std::string s = tag.substr(2); // remove "pt"
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

  // Ratio names and titles
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
    // Collect all labels belonging to this pt tag, sorted by eta upper cut
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

    // Build arrays
    std::vector<double> etaHi(N), ex(N, 0.);
    std::vector<double> yK2K1(N), eyK2K1(N);
    std::vector<double> yK3K1(N), eyK3K1(N);
    std::vector<double> yK4K2(N), eyK4K2(N);

    double etaLo = 0.;
    for (int i = 0; i < N; ++i) {
      etaHi[i] = etaLabels[i].first;
      const auto& lbl = etaLabels[i].second;
      auto [lo, hi]   = etaRange(lbl);
      etaLo = lo;

      const CumulantResults real  = extractResults(fIn, lbl, targetEvents, net);
      const CumulantResults ideal = extractResults(fIn, lbl + "_ideal", targetEvents, net);
      if (!real.valid || !ideal.valid) {
        yK2K1[i] = yK3K1[i] = yK4K2[i] = 0.;
        eyK2K1[i] = eyK3K1[i] = eyK4K2[i] = 0.;
        continue;
      }
      yK2K1[i]  = ideal.k2k1;    eyK2K1[i]  = real.errK2K1R;
      yK3K1[i]  = ideal.k3k1;    eyK3K1[i]  = real.errK3K1R;
      yK4K2[i]  = ideal.k4k2;    eyK4K2[i]  = real.errK4K2R;
    }

    auto [ptLo, ptHi] = ptRange(tag);

    // One canvas per ratio
    double* yArr[3]  = {yK2K1.data(),  yK3K1.data(),  yK4K2.data()};
    double* eyArr[3] = {eyK2K1.data(), eyK3K1.data(), eyK4K2.data()};

    for (int r = 0; r < 3; ++r) {
      std::string cName = Form("cVsEta_%s_%s", tag.c_str(), ratioNames[r]);

      TCanvas c(cName.c_str(), cName.c_str(), 800, 700);
      c.SetLeftMargin(0.16);
      c.SetBottomMargin(0.13);
      c.SetTopMargin(0.05);
      c.SetRightMargin(0.04);

      // determine y range: max fixed at 1.02, min = (min y - err) - 0.2
      double yminErr = yArr[r][0] - eyArr[r][0];
      for (int j = 1; j < N; ++j)
        yminErr = std::min(yminErr, yArr[r][j] - eyArr[r][j]);

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
      gStyle->SetOptStat(0);
      hF.Draw();

      TGraphErrors g(N, etaHi.data(), yArr[r], ex.data(), eyArr[r]);
      g.SetMarkerStyle(20);
      g.SetMarkerSize(1.2);
      g.SetMarkerColor(kRed + 1);
      g.SetLineColor(kRed + 1);
      g.SetLineWidth(2);
      g.Draw("P SAME");

      TLatex latex;
      latex.SetNDC();
      latex.SetTextSize(0.038);
      latex.SetTextFont(42);
      latex.DrawLatex(0.38, 0.88, "NA60+/DiCE Performance");
      latex.DrawLatex(0.38, 0.83,
          Form("Pb#font[122]{-}Pb,  #sqrt{s_{NN}} = %0.1f GeV, 0-5%%", energy));
      latex.DrawLatex(0.38, 0.78,
          Form("%s #leq #it{#eta} #leq #it{#eta}_{max}", fmt(etaLo)));
      latex.DrawLatex(0.38, 0.73,
          Form("%s #leq #it{p}_{T} #leq %s GeV/#it{c}", fmt(ptLo), fmt(ptHi)));

      c.SaveAs(Form("vsEta_%s_%s.pdf", tag.c_str(), ratioNames[r]));
      c.SaveAs(Form("vsEta_%s_%s.png", tag.c_str(), ratioNames[r]));

      fOut.cd();
      g.Write(Form("gVsEta_%s_%s", tag.c_str(), ratioNames[r]));
      c.Write();
    }
  }
}

// ─── ki/kj vs eta_max overlaid for all pt selections ────────────────────────
void analyseVsEtaOverlay(TFile& fIn, const std::vector<std::string>& labels,
                         TFile& fOut, double energy = 8.8,
                         int pdg = 3122, bool net = false, double targetEvents = 4.5e9)
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
  const int kMarkers[] = {20, 21, 22, 23, 29, 33, 34, 47};

  // Collect unique pt tags (sorted by pt lo)
  std::vector<std::string> ptTags;
  for (const auto& lbl : labels) {
    auto pos = lbl.find("_eta_");
    if (pos == std::string::npos) continue;
    std::string tag = lbl.substr(0, pos);
    if (std::find(ptTags.begin(), ptTags.end(), tag) == ptTags.end())
      ptTags.push_back(tag);
  }

  // Collect unique eta upper cuts (sorted)
  std::vector<double> etaHiAll;
  for (const auto& lbl : labels) {
    double hi = etaRange(lbl).second;
    if (std::find(etaHiAll.begin(), etaHiAll.end(), hi) == etaHiAll.end())
      etaHiAll.push_back(hi);
  }
  std::sort(etaHiAll.begin(), etaHiAll.end());
  const int Neta = (int)etaHiAll.size();

  // eta lo (assume same for all)
  double etaLo = 0.;
  for (const auto& lbl : labels) { etaLo = etaRange(lbl).first; break; }

  // For each pt tag build arrays over eta upper cuts
  const int Npt = (int)ptTags.size();
  // yData[pt][ratio][etaIdx]
  std::vector<std::vector<std::vector<double>>> yData(Npt,
      std::vector<std::vector<double>>(3, std::vector<double>(Neta, 0.)));
  std::vector<std::vector<std::vector<double>>> eyData(Npt,
      std::vector<std::vector<double>>(3, std::vector<double>(Neta, 0.)));

  for (int p = 0; p < Npt; ++p) {
    for (int e = 0; e < Neta; ++e) {
      // find matching label
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
      yData[p][0][e]  = ideal.k2k1;  eyData[p][0][e]  = real.errK2K1R;
      yData[p][1][e]  = ideal.k3k1;  eyData[p][1][e]  = real.errK3K1R;
      yData[p][2][e]  = ideal.k4k2;  eyData[p][2][e]  = real.errK4K2R;
    }
  }

  std::vector<double> ex(Neta, 0.);

  for (int r = 0; r < 3; ++r) {
    // y range across all pt tags
    double yminErr = 1e9;
    for (int p = 0; p < Npt; ++p)
      for (int e = 0; e < Neta; ++e)
        yminErr = std::min(yminErr, yData[p][r][e] - eyData[p][r][e]);

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
    gStyle->SetOptStat(0);
    hF.Draw("AXIS");

    // keep points exactly at eta_{max}
    std::vector<std::unique_ptr<TGraphErrors>> graphs(Npt);
    for (int p = 0; p < Npt; ++p) {
      std::vector<double> xs(Neta);
      for (int e = 0; e < Neta; ++e) xs[e] = etaHiAll[e];
      graphs[p] = std::make_unique<TGraphErrors>(
          Neta, xs.data(), yData[p][r].data(),
          ex.data(), eyData[p][r].data());
      int col = kColors[p % nColors];
      graphs[p]->SetMarkerStyle(kMarkers[p % nColors]);
      graphs[p]->SetMarkerSize(1.2);
      graphs[p]->SetMarkerColor(col);
      graphs[p]->SetLineColor(col);
      graphs[p]->SetLineWidth(2);
      graphs[p]->Draw("P SAME");
    }

    // Legend
    double legY2 = 0.50;
    TLegend leg(0.2, legY2 - 0.05*Npt, 0.55, legY2);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.033);
    for (int p = 0; p < Npt; ++p) {
      auto [ptLo, ptHi] = ptRange(ptTags[p]);
      leg.AddEntry(graphs[p].get(),
          Form("%s #leq #it{p}_{T} #leq %s GeV/#it{c}", fmt(ptLo), fmt(ptHi)),
          "lp");
    }
    leg.Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.038);
    latex.SetTextFont(42);
    latex.DrawLatex(0.2 +0.015, legY2 - 0.05*Npt - 0.03, "NA60+/DiCE Performance");
    latex.DrawLatex(0.2 +0.015, legY2 - 0.05*Npt - 0.08,
        Form("Pb#font[122]{-}Pb,  #sqrt{s_{NN}} = %0.1f GeV, 0-5%%", energy));
    latex.DrawLatex(0.2 +0.015, legY2 - 0.05*Npt - 0.13,
        Form("%s #leq #it{#eta} #leq #it{#eta}_{max}", fmt(etaLo)));

    c.SaveAs(Form("vsEtaOverlay_%s.pdf", ratioNames[r]));
    c.SaveAs(Form("vsEtaOverlay_%s.png", ratioNames[r]));

    fOut.cd();
    for (int p = 0; p < Npt; ++p)
      graphs[p]->Write(Form("gVsEtaOverlay_%s_%s",
          ptTags[p].c_str(), ratioNames[r]));
    c.Write();
  }
}

// ─── Main entry point ─────────────────────────────────────────────────────────
/*std::vector<std::string> labels = {
                                          "pt0.2_2_eta_2_4", "pt0.5_2_eta_2_4", "pt0.75_2_eta_2_4", "pt1_2_eta_2_4",
                                          "pt0.2_2_eta_2_3", "pt0.5_2_eta_2_3",  "pt0.75_2_eta_2_3", "pt1_2_eta_2_3",
                                          "pt0.2_2_eta_2_3.5", "pt0.5_2_eta_2_3.5",  "pt0.75_2_eta_2_3.5", "pt1_2_eta_2_3.5",
                                          "pt0.2_2_eta_2_2.5", "pt0.5_2_eta_2_2.5",  "pt0.75_2_eta_2_2.5", "pt1_2_eta_2_2.5"
                                        },*/
void analyzeAndPlotCumulantRatios(std::vector<std::string> labels = {
                                          "pt0.2_2.5_eta_2_3.5"                                    
                                        },
                                        double energy = 7.5,
                                        int pdg = 0,
                                        bool net = true,
                                        double targetEvents = 4.5e9
                                      )
{
  TFile fIn("fOutAll.root");
  if (fIn.IsZombie()) {
    std::cout << "Error: cannot open fOutAll.root" << std::endl;
    return;
  }

  TFile fOut("fC.root", "RECREATE");

  for (const std::string& lbl : labels)
    analyseLabel(fIn, lbl, fOut, energy, pdg, net, targetEvents);

  analyseVsEtaCuts(fIn, labels, fOut, energy, pdg, net, targetEvents);
  analyseVsEtaOverlay(fIn, labels, fOut, energy, pdg, net, targetEvents);

  fOut.Close();
}