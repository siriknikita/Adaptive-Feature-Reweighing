
# Experiment #6 — Real-Data Recoverability on a T4

This is the first **real-image** experiment in the Recoverable Feature Routing series.

The default backend is the public Hugging Face dataset:

`ThinothW/Deepfake-Identity-Isolated-Dataset-PreP`

Why this backend for the first T4 run:

- 142,837 real/fake face crops.
- 260×260 images.
- train / validation / test splits are already provided.
- the dataset card reports identity-isolated graph-based splitting.
- it contains real images and deepfakes derived from DF40/FF++ plus additional synthesis sources.
- total dataset size is only about 2.87 GB, which is sane for Colab.
- label convention is `0 = fake`, `1 = real`.

This is intentionally a **pilot mechanism experiment**, not yet the final official-DF40 paper benchmark.

## Scientific question

Do the synthetic results survive when the underlying images are real deepfake benchmark images?

We train four separate real-image experts:

1. `rgb`
2. `highpass`
3. `chroma`
4. `structure`

They are trained on clean images only.

Then we apply a deterministic but randomized **procedural glasses intervention** to held-out real benchmark crops. The underlying sample and fake/real label do not change.

We test whether:

- one or more experts lose reliability under eyewear;
- backup experts remain useful;
- a fixed/global router fails to exploit the change;
- a slow router becomes difficult to adapt;
- an independent fast route recovers backup evidence;
- a memory switch invokes the fast route selectively while preserving clean performance.

## T4 design

This project is deliberately T4-friendly:

- ResNet-18 experts, not giant ViTs.
- 224×224 inputs.
- AMP / FP16 on CUDA.
- experts are trained **sequentially**.
- experts are frozen before routing experiments.
- expert logits and RGB context embeddings are cached.
- all routing / few-shot experiments run on cached tensors.
- one representation seed; five routing/adaptation seeds.
- checkpoints and caches are written to Google Drive.
- every expensive stage is resumable.

Default pilot size:

- train: 8,000 balanced samples
- validation: 2,000 balanced samples
- test: 3,000 balanced samples
- expert epochs: 2
- routing seeds: 11, 22, 33, 44, 55

## Google Colab

Open:

`Experiment_6_Real_Data_Recoverability_T4.ipynb`

Set the runtime to **T4 GPU**, then run top-to-bottom.

The notebook:

1. installs dependencies with `uv`;
2. mounts Google Drive;
3. writes the experiment code to `/content`;
4. downloads the public dataset;
5. trains/resumes experts;
6. builds/resumes caches;
7. runs the routing experiments;
8. writes a final ZIP into Drive.

The output directory is:

`MyDrive/RFR_Exp6_Real/`

The final archive is:

`MyDrive/RFR_Exp6_Real/results_exp6.zip`

Send that archive back to ChatGPT.

## Real-data caveats

The default Hugging Face pool is flattened and does not expose a manipulation-method field, so this first run cannot make generator-specific claims.

The next benchmark step, if the mechanism survives, should use the official categorized DF40 data to test:

- generator-disjoint evaluation;
- FF++ → Celeb-DF domain shift;
- natural eyewear;
- multiple nuisance conditions;
- per-generator feature reliability.

## Data/licensing

The dataset card says the derivative dataset is for academic/research use and that source datasets remain subject to their original licenses. DF40 itself is distributed under CC BY-NC 4.0. Review the source terms before publication or redistribution.
