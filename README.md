# BoltGuard AI — Screw Defect Detection

A computer vision system that classifies screws as **good** or **defective** from images, with visual explainability (Grad-CAM) and a deployed two-page inspection dashboard. Built as an end-to-end ML engineering exercise: dataset curation, iterative model development, a real data bug found and fixed mid-project, evaluation beyond accuracy, and a working live demo — not just a notebook.

**[Live demo →](https://digiryte-challenge2-screw-defect-qc-mxb6t65jnghfannhkwduvt.streamlit.app/)** &nbsp;|&nbsp; **[Notebook →](https://github.com/harinipalanisamy-krishna/digiryte-challenge2-screw-defect-qc/blob/main/Digiryte_Challenge2_ScrewQC_v2%20%282%29.ipynb)** &nbsp;|&nbsp; ResNet18 · PyTorch · Streamlit · Grad-CAM

![Grad-CAM example](gradcam_example.png)
*A defective screw with its Grad-CAM attention overlay, as shown on the Inspect page.*

---

## Why This Project

Visual quality control is one of the most common real-world applications of computer vision, and one of the most unforgiving: a missed defect can mean a faulty product reaches a customer, while too many false alarms erode trust in the tool and get it switched off. This project treats that tradeoff as a first-class design decision rather than an afterthought — the detection threshold, precision/recall reporting, and evaluation methodology are all built around the question *"what does a QC operator actually need from this?"*

## Problem Statement

Given an image of a screw, predict whether it is defective, and — for defective screws — highlight *where* the model believes the defect is, so a human operator can quickly verify the flag rather than treat the model as a black box.

## Dataset

**[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)** (screw category) — the field-standard academic benchmark for industrial visual anomaly detection, chosen because it's the closest publicly available proxy for a real factory QC dataset, and because strong performance here demonstrates transferability to any client with a visual inspection use case (not just this specific product).

- **~380 images** total, combining the "good" images from both the train and test splits with 5 defect-type folders (`manipulated_front`, `scratch_head`, `scratch_neck`, `thread_side`, `thread_top`), then re-split 80/20 with class balance preserved.

## The App

The deployed Streamlit app is a two-page dashboard, not a single upload form:

- **🔍 Inspect** — a long-form, one-screw-at-a-time view. Upload a batch or take a live photo via the browser camera; each result renders as its own card: full-size photo → verdict badge → (for defects) the Grad-CAM attention map with an explicit "approximate region, not a confirmed boundary" disclaimer.
- **📊 Dashboard** — session-wide analytics, computed live from every image inspected so far (not simulated numbers): Inspected / Passed / Defective / Defect Rate, filterable by status (All / Passed / Defective), a compact result grid, and two export buttons (`.txt` report, `.csv` defect log).

## Key Design Decisions

| Decision | Rationale | Tradeoff acknowledged |
|---|---|---|
| **Supervised classification**, not unsupervised anomaly detection | MVTec AD is primarily used for *unsupervised* anomaly detection (train on "good" images only), since real factories often have abundant good parts but scarce labeled defects. This project uses supervised learning because labeled defects were available, and supervised methods are simpler and typically more accurate when labels exist. | In a deployment with few/no labeled defects, an unsupervised method (PatchCore, PaDiM, EfficientAD) would be more appropriate — see Future Work. |
| **ResNet18** backbone | Small dataset (~380 images) + binary task → a smaller pretrained model reduces overfitting risk without sacrificing accuracy, and trains fast enough for rapid iteration. | Not benchmarked against alternative architectures (e.g. EfficientNet-B0) — a natural next experiment. |
| **Threshold tuned toward recall** | For QC, a missed defect is costlier than a false alarm, so the deployed default favors catching more defects at the cost of more false positives. | Precision on the defective class is 0.68 — explicitly tracked and reported, not hidden. |
| **Grad-CAM for explainability** | Gives the operator a visual reason to trust (or override) the model's flag, rather than a bare label. | Localization is weakly-supervised (heatmap thresholding), not pixel-precise — see Known Limitations. |
| **Two-page app instead of one long page** | Separates "look at this one result closely" (Inspect) from "how is the whole session going" (Dashboard) — mirrors how real QC operators actually use inspection tools: verify individual flags, then check shift-level stats. | Slightly more code to maintain (`common.py` shares logic between pages to avoid duplication). |

## Development Process

The model wasn't built in one pass — each stage was driven by a specific question raised by the previous result:

1. **Baseline** (no augmentation, plain split): 81.4% test accuracy vs. 100% training accuracy — a clear overfitting gap on this small dataset.
2. **Data augmentation** (flips, rotation, color jitter): improved to 84.9% test accuracy with a narrower train/test gap, confirming augmentation was addressing overfitting as intended.
3. **Hyperparameter tuning** via a single validation split initially pointed to one clear winning config — but test accuracy on that config came back similar to the baseline, which didn't match the validation signal.
4. **K-fold cross-validation** was introduced specifically to resolve that discrepancy, since a single small validation split is an unreliable way to compare configs. This gave a trustworthy final configuration.
5. **Data quality bug found and fixed**: a routine check of the training folder revealed that several MVTec defect-type folders shared identical filenames (e.g. `000.png`), which silently overwrote images when combined into one folder without renaming. Fixing this (prefixing filenames by defect type) and retraining with the *same proven approach* confirmed the data bug — not the architecture or hyperparameters — had been the real bottleneck the whole time.
6. **Evaluation beyond accuracy**: confusion matrix, precision/recall/F1, ROC-AUC and PR-AUC, and multi-threshold analysis to make the precision/recall tradeoff explicit and tunable.
7. **Grad-CAM localization investigation**: after noticing the attention heatmap sometimes didn't land on the visually obvious defect, tested both a finer feature layer (layer3 vs layer4) and a sharper CAM variant (GradCAM++ vs GradCAM) — neither moved the highlighted region. Concluded this points to the model's own learned attention (likely a small-dataset shortcut feature) rather than a fixable localization-algorithm problem; documented as a known limitation instead of over-engineering further algorithmic patches. An adaptive percentile-based threshold was still added to the bounding-box logic, since that *does* measurably reduce oversized, misleading boxes on diffuse heatmaps.
8. **Deployment**: a two-page Streamlit app (Inspect + Dashboard) with batch upload, live camera capture, Grad-CAM overlays, session-wide analytics, and downloadable TXT/CSV reports.

This debugging arc — catching a data bug that a purely metrics-driven approach would have missed, using k-fold to catch an unreliable validation signal, and knowing when a localization discrepancy is a model-attention issue rather than an algorithm bug — is arguably the most representative part of the project: it reflects the kind of diagnostic judgment that separates a model that trains from a model that's trustworthy.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **~87%** |
| Recall (defective class) | **~88%** (21/24 defects caught) |
| Precision (defective class) | **0.68** |

Full precision/recall breakdown across thresholds, confusion matrix, and threshold-tuning analysis are in the notebook (`confusion_matrix.png`, `gradcam_example.png`).

## Architecture

```
Image upload / live camera capture (Streamlit)
        │
        ▼
 Preprocessing (resize 224×224, normalize)
        │
        ▼
 ResNet18 (ImageNet-pretrained, fine-tuned FC head)
        │
        ├──► Softmax probability ──► Threshold ──► Good / Defective label
        │
        └──► Grad-CAM (layer4, vanilla GradCAM) ──► Heatmap
                  │
                  └──► Adaptive-threshold bounding box ──► Overlay shown to operator
```

```
app.py (navigation)
  ├── views/inspect.py    → single-image detail cards
  └── views/dashboard.py  → session-wide analytics + exports
         both import shared logic from common.py
                                │
                                ▼
                          predict.py (model-only, no UI code)
```

`predict.py` holds all model/inference logic and has zero UI code; `common.py` holds shared styling and session-state helpers used by both pages — kept separate so inference logic could be reused in a different frontend (e.g. a FastAPI service) without touching it, and so the two pages never drift out of visual sync.

## Real-World Deployment Considerations

This app uses a single top-down camera image per screw — sufficient for a demo and for many visible defect types, but worth being explicit about where that falls short of an actual production line:

- **Camera blind spots**: a single top-down view cannot see the underside of the screw head or the very tip of the threads. Any defect located there would simply never appear in the image the model receives, regardless of how accurate the model itself is.
- **How real industrial systems solve this** (multiple camera/optical setups, not a software fix):
  - **Glass rotary dial table** — screws travel across a clear, hardened glass conveyor; one camera shoots from above, a second shoots from directly underneath through the glass, capturing both the top and underside in one pass.
  - **Mirrored prism enclosures** — the screw sits in a V-shaped slot surrounded by angled mirrors, letting a single overhead camera capture the top view plus reflected side/bottom views in one frame.
  - **Rotational fixtures** — a mechanical gripper spins the screw 360° in front of a stationary camera, building a full panoramic wrap of the fastener for high-precision inspection.
- **What this means for this project**: the current model and app are correctly scoped to what a single-image, single-camera setup can support. Extending to full-coverage inspection would be a hardware/imaging-rig decision made alongside the ML pipeline, not a model or code change — worth calling out explicitly as future scope rather than an oversight.

## Known Limitations

Stated explicitly, because a QC tool that hides its own failure modes is more dangerous than one that's upfront about them:

- **Small test set** (24 defective examples): individual metrics have real variance — a couple of examples flipping outcome shifts recall by several points. K-fold results from tuning should be weighted more heavily than any single held-out split.
- **Precision/recall tradeoff is deliberate, not accidental**: the default threshold accepts more false alarms in exchange for catching more real defects, which is the right prioritization for QC-assist — but it's a design choice a deployer should be able to tune, hence the sensitivity slider.
- **Trained on studio-condition images**: MVTec's plain background and controlled lighting may not represent a real factory floor's camera setup; performance under different lighting/background/angle conditions is untested.
- **Single product category**: evaluated only on "screw"; generalization to other MVTec categories or other product types entirely is unverified.
- **No out-of-distribution detection**: the model only knows "good screw" vs. "defective screw" — it has no way to recognize an input that isn't a screw at all. Feeding it an unrelated object still forces a prediction into one of the two known classes. This is expected behavior for a binary classifier, not a bug, but it means the app currently assumes the operator is only ever submitting screw images.
- **Approximate localization**: Grad-CAM bounding boxes are a weakly-supervised heuristic, not a precise segmentation. Both a finer feature layer and a sharper CAM variant (GradCAM++) were tested and neither changed the highlighted region, indicating the model's own learned attention — not the localization algorithm — is the limiting factor. MVTec's own pixel-level ground-truth masks (unused here) would give a stronger signal; see Future Work.

## Future Work

- **Unsupervised anomaly detection baseline** (PatchCore / PaDiM / EfficientAD) trained only on "good" images, as a direct comparison to the supervised approach — most relevant if labeled defects are scarce in a real deployment.
- **Multi-class defect typing**, using MVTec's 5 labeled defect types instead of collapsing them into one "defective" class.
- **Pixel-level defect segmentation** using MVTec's ground-truth masks, replacing the Grad-CAM bounding-box approximation — this is the most direct fix to the localization limitation above, since it would require retraining rather than just changing which layer/CAM variant reads out an already-frozen model.
- **Out-of-distribution / input validation check** so the app can flag "this doesn't look like a screw" instead of forcing a good/defective label on unrelated images.
- **Cross-category generalization testing** against other MVTec object categories.
- **FastAPI inference service** separated from the Streamlit UI, for production-style integration.
- **CI pipeline** (unit tests + lint) on every push.
- **Experiment tracking** (Weights & Biases / MLflow) to replace the manual/printed tracking used across the baseline → augmentation → tuning → k-fold experiments.

## Tech Stack

`PyTorch` · `torchvision` (ResNet18) · `pytorch-grad-cam` · `scikit-learn` (metrics, k-fold) · `Streamlit` (multipage app + `st.camera_input`) · `PIL` · `gdown`

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained weights are downloaded automatically on first run via `gdown` (see `predict.py`).

## Project Structure

```
.
├── app.py                          # Entry point: page config + navigation between the two pages
├── common.py                       # Shared styling (dark theme) + session-state logic used by both pages
├── predict.py                      # Inference logic (model load, predict, Grad-CAM) - no UI code
├── views/
│   ├── inspect.py                  # "Inspect" page - long-form single-image workflow
│   └── dashboard.py                # "Dashboard" page - session analytics, filters, exports
├── requirements.txt
├── sample_images/                  # Example good/defective images for quick demoing
├── confusion_matrix.png
├── gradcam_example.png
├── Digiryte_Challenge2_ScrewQC_v2.ipynb   # Full training & evaluation pipeline
└── README.md
```

---

*Dataset: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), Bergmann et al., CVPR 2019 — used under its research/non-commercial license terms.*
