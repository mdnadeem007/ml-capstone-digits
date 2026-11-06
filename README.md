# Capstone: Handwritten Digit Recognition, from Raw Data to Prediction

An end-to-end machine learning pipeline: raw CSV in, checked data, honest model comparison, a saved model, a prediction script, and a written error analysis of the cases the model gets wrong.

**Result:** an RBF support vector machine reaches **99.2% accuracy** on a held-out test set (3 mistakes in 360 images). Almost every mistake comes with a low confidence score, so the model can tell you when not to trust it. It also has a serious limit: it only works on digits that are centred, as in its training data (see [Assumptions and limits](#assumptions-and-limits)).

## The problem

Given an 8 x 8 greyscale image of a handwritten digit, predict which digit (0 to 9) it shows.

- **Data:** the UCI "optical recognition of handwritten digits" set bundled with scikit-learn, 1,797 images
- **Raw format:** `data/raw/digits.csv`, one row per image, 64 pixel columns (`pixel_0_0` ... `pixel_7_7`, values 0 to 16) and a `target` column
- **Classes:** 10, roughly balanced, so accuracy is a fair headline metric (macro F1 is reported too)

## Pipeline

| Step | Script | What happens |
|---|---|---|
| 1. Load and validate | `common.py` | Reads the raw CSV and fails loudly on missing columns, missing values, pixels outside 0 to 16, or labels outside 0 to 9 |
| 2. Split | `common.py` | 80/20 stratified split, fixed seed. The test set is not touched again until step 4 |
| 3. Compare models | `train.py` | Five candidates plus a baseline, scored with 5-fold cross-validation on the training data only |
| 4. Final score | `train.py` | The winner is trained on all training data and scored on the test set once |
| 5. Predict | `predict.py` | New pixel CSV in, predicted digit and confidence out |
| 6. Error analysis | `error_analysis.py` | Studies the mistakes (details below) |

## Results

**Model comparison** (5-fold cross-validation, training data only)

| Model | CV accuracy |
|---|---|
| Baseline (always the most common digit) | 0.101 |
| Logistic Regression | 0.965 |
| Random Forest | 0.975 |
| SVM (RBF) with StandardScaler | 0.981 |
| KNN (k=5) | 0.988 |
| **SVM (RBF), fixed scaling (chosen)** | **0.988** |

KNN and the SVM are effectively tied (0.9875 against 0.9882, with a fold-to-fold spread of about 0.004 to 0.008). I chose the SVM because it scored marginally higher and gives usable confidence scores.

**Final test set (360 images, used once):** accuracy **0.9917**, macro F1 **0.9916**, **3 mistakes**. One mistake moves accuracy by about 0.3 points, so treat the exact figure as approximate.

## Error analysis

The mistakes were studied on the **training** data using out-of-fold predictions (each image predicted by a model that never saw it). That gives 17 mistakes to study instead of 3, and keeps the test set clean. Outputs are in `reports/`.

**1. The mistakes are scattered, not systematic.** There is no single bad pair of digits. The most common confusion is 8 predicted as 1 (3 times). 5 and 9 are mixed up in both directions (2 times each). The weakest digit by recall is 9 (0.965). With only 17 mistakes I would not read much into any one pair. `reports/oof_confusion_matrix.png` shows the full picture.

**2. Many mistakes are hard to read.** `reports/misclassified_examples.png` shows the mistakes at 8 x 8. Several are badly formed or ambiguous even to a human eye at that resolution, so some of the remaining error probably cannot be removed with a better model.

**3. The model knows when it is unsure. This is the most useful finding.**
- Median confidence when the model is right: **0.97**. When wrong: **0.45**.
- 116 of 1,437 training images (8%) fall below 0.8 confidence, and those contain **16 of the 17 mistakes**.
- On the remaining confident images, out-of-fold accuracy is **99.9%**.
- Check on the untouched test set: 24 of 360 images (7%) fall below 0.8, and they contain **all 3** test mistakes.

In practice that suggests a simple rule: accept predictions at 0.8 confidence or above automatically and send the rest to a person. The 0.8 cut-off was picked by looking at the training analysis, and the test set agreed with it, but 3 test mistakes is a very small sample.

**4. One idea that did not hold up.** I checked whether mistakes tend to be unusually faint or unusually heavy images. Average ink is almost identical (315 for mistakes, 313 for correct), so there is no evidence for that.

**5. Stress test: where it breaks.** Test images were slightly altered and scored again (`reports/stress_test.csv`):

| Change to the test images | Accuracy |
|---|---|
| None | 0.992 |
| Random pixel noise (sd 2) | 0.983 |
| Rotated 15 degrees | 0.747 |
| Shifted 1 pixel down | 0.533 |
| Shifted 1 pixel right | 0.519 |

The model is robust to noise but very fragile to position. A one-pixel shift roughly halves the accuracy.

## A decision the analysis changed

My first version used `StandardScaler`, the usual default. It scored fine on the clean test set (97.5%), but my noise stress test dropped it to about 12%. The cause: some corner pixels are almost always empty in training, so standardising them makes any stray ink there look extreme. Pixel values already have a known range (0 to 16), so I replaced it with a fixed divide-by-16 that learns nothing from the data. Clean accuracy rose and noise robustness came back. The cross-validation table above still includes the StandardScaler version so the comparison is visible (0.981 against 0.988).

One honest caveat: I found this problem using a stress test on the test images, so the test set influenced a preprocessing choice. The cross-validation numbers (training data only) point the same way, but the 99.2% test figure is slightly less pristine than it would otherwise be.

## Assumptions and limits

**Assumptions the model relies on**
- Images are 8 x 8, with pixel values from 0 to 16, and the digit is **centred** in the frame
- New data comes from the same kind of source as the training data (same capture process, similar writers)

**Limits**
- **Position sensitivity.** As the stress test shows, an off-centre or rotated digit will often be misread. It would need re-centring or augmentation before use on real images.
- **Possible optimism from the split.** The data has no writer ID, so the split is random. If the same writer appears in both train and test, scores are probably higher than for a completely new writer.
- **Small, clean data.** 1,797 images of tidy digits. Results do not transfer to real handwriting, other resolutions or scanned forms.
- **Confidence is a ranking aid.** The SVM's probabilities are an estimate, not a guarantee. A 0.9 does not mean "right 90% of the time".
- **Small error counts.** 3 test mistakes and 17 out-of-fold mistakes are too few to rank KNN against the SVM or to claim a specific weak digit.
- **No hyperparameter tuning.** All models use near-default settings. A tuned model might gain a little.

## How to run

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt

python train.py                              # compare models, save model and metrics
python predict.py data/sample_input.csv      # predict on the 10-row sample
python error_analysis.py                     # recreate the error analysis in reports/
```

Run `train.py` first: it creates the saved model (`models/model.joblib`, not stored in the repo, because saved models depend on the library version). Everything runs in under two minutes and needs no internet connection.

To predict on your own images, supply a CSV with the 64 columns `pixel_0_0` ... `pixel_7_7` (values 0 to 16) and run `python predict.py your_file.csv`. Results go to `reports/predictions.csv`.

**In Google Colab:** upload the project folder (or zip), then run `!pip install -r requirements.txt`, `!python train.py`, `!python predict.py data/sample_input.csv` and `!python error_analysis.py` from inside it.

## Files

```
common.py            shared settings, data loading and checks
prepare_data.py      recreates data/raw/digits.csv (already included)
train.py             model comparison and final model
predict.py           predictions on new data
error_analysis.py    error analysis and stress test
data/raw/digits.csv  raw data
data/sample_input.csv  10 test images without labels, for trying predict.py
reports/             metrics, tables and plots produced by the scripts
```
