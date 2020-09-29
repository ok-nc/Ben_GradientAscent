# Machine Learning with Metamaterials 
## Author: Omar Khatib

## This is a machine learning network designed to aid in the modeling and simulation of metamaterials (MMs). 
### The original forward network model was developed by Christian Nadell and Bohao Huang in tensoflow, and further optimized and ported to pytorch by Ben Ren. 

### The main aim of this project is to incorporate known physics relations to improve deep learning networks applied to MM systems. 
### The fundamental relationships involve using Lorentz oscillator equations as the basis for mapping an input geometry to output transmission, reflection, or absorption spectra. 

# Developer Log:

### Roadmap for this work:
1. Incorporate Lorentz oscillator equations into neural network
2. Train a toy model to learn prescribed relationship of lorentz osc parameters to input geometry
3. Train network on CST simulations for 2x2 unit cell MMs using a Lorentz layer. 
4. Incorporate auxilliary network(s) for capturing 2nd order effects (e.g. scattering, spatial dispersion, etc)

## Stages for realizing the Lorentz param facilitated training
- [x] Test out the validity of the concept using Omar's Toy problem of 4 Oscillators
- [x] swipe through the facilitation ratio and the loss using random selection
- [x] Maximize utility of faciliated labels, change the code structure  to  alternating training 
- [x] Test out the hypothesis of unmatching labels would harm training instead of helping, add the permuated Lorentz labels as the training
- [x] Simulate on toy data set if the Lorentzian correspondence is built according to the frequency rank
- [ ] Debug the code on DCC for the new modes of gt_match_style == 'random' and 'peak'
## To-do list for Ben
- [x] Add the module for multi-loss training