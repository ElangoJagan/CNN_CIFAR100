# Project 7: CIFAR-100 CNN Classification (Advanced)

Part of a 15-project ML/AI learning roadmap. This project builds on Project 6 (CIFAR-10 CNN)
by tackling a genuinely harder dataset — CIFAR-100's 100 fine-grained classes — using the
same production-style modular pipeline (logging, custom exceptions, OOP patterns) rather
than notebook-style code.

## Goal

Build a from-scratch CNN for CIFAR-100 classification, then honestly evaluate it against
data augmentation and transfer learning (MobileNetV2) — no cherry-picking results, reporting
what actually happened including where the model struggles.

## Project Structure
Project7_CIFAR100_CNN/
├── src/
│   ├── logger.py                      Singleton logger
│   ├── exception.py                   Custom exception handling
│   ├── configfile.py                  DataLoaderConfig (dataclass)
│   ├── data_loader.py                 Loads CIFAR-100, saves to disk
│   ├── preprocessor.py                Normalize + stratified train/val split
│   ├── model.py                       CNN architecture (3 Conv blocks)
│   ├── trainer.py                     Training + EarlyStopping/ModelCheckpoint callbacks
│   ├── evaluator.py                   Confusion matrix, classification report, plots
│   ├── augmentor.py                   Data augmentation (method chaining pattern)
│   ├── experiment_tracker.py          Logs every run to CSV for comparison
│   ├── misclassification_analyzer.py  Which classes get confused, and why
│   ├── interpretability.py            Grad-CAM heatmaps
│   ├── transfer_learning.py           MobileNetV2 fine-tuning (inherits CNNModel)
│   ├── pipeline.py                    Orchestrates the full flow end-to-end
│   └── app.py                         Flask app for real-image predictions
├── artifacts/                         Models, plots, experiment logs (gitignored)
├── data/raw/                          Cached CIFAR-100 data (gitignored)
├── templates/index.html               Flask UI
└── requirements.txt

## Setup

```powershell
# from the Git_Clone_repo level, with ml_env already created
.\ml_env\Scripts\activate
cd Project7_CIFAR100_CNN
pip install -r requirements.txt
```

## Running the pipeline

```powershell
python -m src.pipeline
```

Runs the full flow: load data -> preprocess -> train CNN -> evaluate -> log experiment ->
misclassification analysis. Takes 15-25 minutes on CPU (no GPU support on native Windows
for TensorFlow >= 2.11 -- WSL2 or DirectML needed for GPU acceleration).

## Running the Flask app

```powershell
python -m src.app
```

Open `http://127.0.0.1:5000`, upload an image, get top-5 predictions.

## Results

| Experiment                      | Test Accuracy |
|----------------------------------|---------------|
| Baseline CNN (no augmentation)   | 42.34%        |
| CNN + data augmentation          | **45.67%**    |
| Transfer learning (MobileNetV2, frozen base) | 45.02% |

**Best model:** CNN with data augmentation, 45.67% test accuracy on CIFAR-100 (100 classes).

For context: random guessing on 100 classes = 1%. A basic from-scratch CNN typically lands
35-50% on CIFAR-100, so this result is in the expected, healthy range for this model scale.

## Key findings

**Data augmentation helped, honestly.** +3.3 percentage points over baseline, and reduced
the train/val accuracy gap (52.8% / 41.0% baseline -> smaller gap with augmentation),
consistent with reduced overfitting.

**Transfer learning did not clearly beat the from-scratch CNN in this run.** Three likely
reasons, not treated as a failure but as a real, reportable finding:
- MobileNetV2's base layers were kept frozen — only a small custom head was trained
- Training ran for just 10 epochs, and validation accuracy was still climbing at the end
- Input images were upscaled from 32x32 to 96x96, which stretches pixels rather than adding
  real detail, working against a model pretrained on genuinely high-resolution images

**Real-world photos generalize poorly.** Testing the deployed app with an actual photo (a
cat) produced a low-confidence, incorrect top prediction (wolf, 16.5%). This reflects a
genuine domain gap between CIFAR-100's small, compressed training images and real-world
photography — a well-known limitation of small benchmark datasets, demonstrated firsthand
rather than assumed.

**Misclassification analysis showed sensible errors, not random noise.** 8 of the top 10
confused class pairs were within the same CIFAR-100 superclass (e.g. visually similar
animals), suggesting the model is learning genuine, reasonable visual features rather than
guessing randomly — even on the classes it gets wrong.

## What would change at true production scale

- Data augmentation and hyperparameters would be tuned via `experiment_tracker.py` across
  many more runs rather than 2-3 manual comparisons
- Transfer learning would use unfrozen fine-tuning with a proper learning-rate schedule,
  not a single frozen-base run
- Real-world deployment would need a preprocessing pipeline matched to the actual input
  distribution (e.g. training with augmented/degraded images to bridge the domain gap),
  not just resizing
- Model monitoring for input drift (planned for Project 13) would catch exactly this kind
  of domain mismatch automatically in production

## Tech stack

Python, TensorFlow/Keras, scikit-learn, Flask, OpenCV, matplotlib/seaborn, pandas