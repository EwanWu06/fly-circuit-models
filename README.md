# Fly Circuit Models

Two spiking-neuron models of *Drosophila* circuits. Both use the leaky
integrate-and-fire (LIF) equations and parameters of Shiu et al. (2024).

| Folder | What it is |
|---|---|
| [`escape-circuit/`](escape-circuit) | The giant fiber escape circuit (364 neurons) built from the MaleCNS v1.0 connectome. Does it reproduce published physiology, and does that depend on its specific wiring? |
| [`minimal-fly/`](minimal-fly) | A teaching model: 8- and 20-neuron circuits that steer a simulated fly in an odour field. |

An illustrated explainer of the escape-circuit project, in English and
Chinese: <https://ewanwu06.github.io/fly-escape-switch/>

## Escape circuit

The real circuit is checked against four published observations (B1-B4), then
rewired with degree-preserving double-edge swaps. Each neuron keeps its in- and
out-degree, and its outgoing synapse counts and signs. The pass criteria were
fixed before any benchmark was run.

- **One of four observations is reproduced.** The giant fiber spikes when the
  looming object subtends 33.7-37.2° at every approach speed (B1, band
  30-50°). The width of the input tuning was calibrated against B1.
- **B2-B4 are not reproduced.** Two coded checks were flawed. B3's check passes
  trivially when the giant fiber does not respond. The measure behind B2 and
  B4 reads how close together the two giant fibers reach threshold, not how
  large the response is. A post hoc check that reads each giant fiber
  separately finds that the two eyes add linearly, so the published
  sublinearity (B4) is not reproduced.
- **B1 survives rewiring.** None of the 24 tests (6 doses x 4 measures,
  r/v = 40 ms) is significant after Benjamini-Hochberg correction; the
  smallest uncorrected p is 0.287. At the largest dose, 40.6% of the original
  edges remain. The mean spike angle shifts by at most 0.11°, and every
  rewired circuit still spikes inside B1's band. For B1's measure, the degree
  distribution is enough. This holds at one approach speed, one input draw and
  one input gain.

Degree-preserving shuffles of the whole fly brain have been run before
([Kisame76/drosophila-brain-mlx](https://github.com/Kisame76/drosophila-brain-mlx)).
This one is graded, circuit-specific and checked against published
physiology.

## Minimal digital fly

A teaching model, not a research finding. The fly compares odour at its two
antennae, and the comparison sets the speed of its left and right legs.

- **Version 1, 8 neurons, designed wiring.** The circuit is wired to turn
  towards the stronger odour (Braitenberg's vehicle 2b), so finding food is
  built in: 185 of 200 trials reach it. Of all 964 degree-preserving rewirings,
  424 steer towards the odour, 426 away and 114 neither. One registered
  prediction failed: raising every threshold slowed the fly but did not lower
  its success rate.
- **Version 2, 20 neurons, random wiring.** Every connection and every
  interneuron's sign is drawn at random, with no new tuning. 11 of 1,000 random
  circuits (1.1%, 95% CI 0.6-2.0%) reach food more often than they do in
  uniform odour.

The circuit is fly-inspired, not anatomical. The odour field is smooth and
static, and the legs are a differential drive.

## Reproduce

```bash
mamba env create -f environment.yml
conda activate fly-circuits

cd escape-circuit
python run_all.py    # about 30 min; downloads about 1.1 GB of connectome data
pytest tests -q

cd ../minimal-fly
python run_all.py    # about 1 hour with 10 worker processes
pytest tests -q
```

Every random step is seeded. Results are written to each project's
`results/` folder.

## Data

Connectome data from the MaleCNS v1.0 dataset, HHMI Janelia Research Campus
FlyEM project, in collaboration with the University of Cambridge, MRC
Laboratory of Molecular Biology, and Google Research. Licensed under
CC-BY 4.0. The raw files are downloaded from the public MaleCNS bulk-data
bucket by `escape-circuit/src/download.py` and checked against SHA-256 hashes.

## References

- Ache, J. M., et al. (2019). Neural basis for looming size and velocity
  encoding in the *Drosophila* giant fiber escape pathway. *Current Biology*,
  29(6), 1073-1081.
- Jang, H., Goodman, D. P., Ausborn, J., & von Reyn, C. R. (2023). Azimuthal
  invariance to looming stimuli in the *Drosophila* giant fiber escape
  circuit. *Journal of Experimental Biology*, 226(8), jeb244790.
  https://doi.org/10.1242/jeb.244790
- Shiu, P. K., et al. (2024). A *Drosophila* computational brain model reveals
  sensorimotor processing. *Nature*, 634, 210-219.
- Gaudry, Q., et al. (2013). Asymmetric neurotransmitter release enables rapid
  odour lateralization in *Drosophila*. *Nature*, 493, 424-428.
- Braitenberg, V. (1984). *Vehicles: Experiments in Synthetic Psychology*.
  MIT Press.

## Authorship and license

Code was written with AI coding tools. The research direction, decisions and
review of all results are by Ewan Wu.

Code: MIT License (see [`LICENSE`](LICENSE)). Connectome data: CC-BY 4.0.
