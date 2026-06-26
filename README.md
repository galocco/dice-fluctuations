This repository contains scripts to compute net-charge cumulants following the formalism presented in:

> Nonaka, T., Kitazawa, M., & Esumi, S. *More efficient formulas for efficiency correction of cumulants and effect of using averaged efficiency*. **Physical Review C**, 95(6), 064912 (2017).

## Overview

The analysis pipeline consists of four main stages:

1. **Event generation** – Generate particle events in the HepMC3 text format.
2. **Efficiency calculation with NA6PRoot** – Determine the reconstruction efficiency for particles and antiparticles as a function of transverse momentum ($p_{\mathrm{T}}$), pseudorapidity ($\eta$), and target.
3. **Accumulator computation** – The ROOT macro `computeAccumulators.C` computes the raw cumulant estimators. Storing the intermediate accumulators enables parallel processing, after which the outputs from multiple jobs can be merged using `hadd`.
4. **Merging and analysis** – The merged accumulators are processed by `analyzeAndPlotCumulantRatios.C`, which computes the efficiency-corrected cumulants and produces the corresponding cumulant-ratio plots.