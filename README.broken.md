(This is in relation to commit `07911bcc1c66ba5121a67e419af32190075f0970`, dated `Fri May 3
17:32:29 2019`.)

# Doesn't work
The code runs, but I cannot reproduce the results of Torlai and Melko in the 1D Ising model
with 6 spins. The authors don't give the parameters of the RBM nor the Ising model for this
case, nor do they give any indication of how long training took or how long to expect. I was
able to determine by guessing that they used a temperature of `kB*T = 0.4` with coupling `J =
1.0`, and assumed the learning rate and initialization are the same as their 2D case.  I've
tried various numbers of hidden node between `2` and `2*num_input_nodes`. For me, the KL
divergence is instantly constant; it can be made to artificially start out high by giving bad
initial values to the weights and biases, but it extremely quickly comes back down to the same
constant value. This would seem to indicate that the RBM has converged, but the doing the same
comparison between the RBM probability distribution and the exact Ising prob. dist. as Torlai
and Melko shows it to be quite terrible, and not at all like the perfect match they had after
many training steps. It would also seem that I can't get the approximation to the KL
divergence and the exact KL divergence to agree (defined as `kldiv()` in `src/RBM.jl`), and
the approximation appears to have a random value compared to the exact. The plot of the KL
divergence for the case of
```
# Ising
J=1.0; kB*T=0.4; stepsize=1; skip=0

# RBM
ising_samplesize=10^3; ising_batchsize=50; learning_rate=0.01; cd_num=5
```
can be found in `kldiv_1D.pdf`, and the prob. dist. comparison can be found in `pdf_1D.pdf`.

I found what appears to be some version of their code at
[https://github.com/GTorlai/IsingRBM](https://github.com/GTorlai/IsingRBM), but it just has
`import Restricted_Boltzmann_Machine` at the top of the main file, and there is no such file
in the repository. There is also no script for producing the 1D results.

I decided to look at the properties of the Ising model Metropolis Markov chain, to see perhaps
if that was a problem. The relevant files are

- `metroising_kldiv.pdf`: The KL divergence between the exact Ising prob. dist. and a sample
  of size `n_burn_samples` after various amounts of "burn-in" up tp `max_burn`. The results
  appear to be random fluctuations around a constant value.
- `metroising_energy.pdf`: Part of the same burn-in experiment, but this time with the
  relative difference between the exact averge energy and the approximated average energy. We
  see the difference fluctuates around 0. The experiments following this are after `max_burn`
  burn-in.
- `metroising_avgmag_autocorr.pdf`: Auto-correlation over `n_autocorr_samples` samples of the
  average magnetization vs. number of samples skipped up to `max_skip`. It would appear that
  about `700` samples should be skipped between each actual sample.
- `metroising_accept_prob.pdf`: The Metropolis acceptance probability for
  `n_accept_prob_samples` samples. It seems to be pretty high most of the time, and I
  believe there is no problem with that.
- `metroising_hist.pdf`: Histogram comparison to the exact distribution vs. number of samples
  taken to make the histogram up to `max_samples`. We see that the ideal is about `10^5`
  samples.

It should be noted that this with only at most one spin being flipped at each step. I also
tried flipping a random number of at most all 6 spins each step, and the resulting histograms
were terrible. I also attempted to look at these with the skip of `700` mentioned, but got
strange results and I need to confirm I didn't make a mistake before concluding anything. My
analysis would be that (for the seed of 1830375382140242722) burn-in doesn't accomplish
anything, and that I should use `10^5` samples. I had already tested these conditions on the
the RBM, so the problem does not appear to be the Ising model Markov chain.
