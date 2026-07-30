# Task 4: Identity Overfitting Mitigation (GRL & Baseline Normalization)

## Problem Addressed: Identity Memorization
In previous training runs (Task 3), the models achieved high accuracy on the training set but performed poorly on validation and test sets containing unseen drivers. Our analysis concluded that the CNN backbones were memorizing **driver-specific traits** (e.g., skin color, glasses, natural eye shape, and lighting conditions) instead of learning generalized signs of drowsiness. 

Because the frames were extracted sparsely using VLC without strict temporal continuity, we could not rely on sequential models (like LSTMs or TCNs) to capture temporal dynamics. Instead, we addressed the identity memorization problem directly on static images using two complementary techniques.

---

## 1. Phase 2: Driver Baseline Normalization
### Why we did this
Geometric features like EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) vary wildly between individuals. For example, a person with naturally droopy eyelids might have a "normal" EAR that looks exactly like another person's "drowsy" EAR. Feeding raw geometric features directly to the model causes confusion when evaluating unseen drivers.

### What we did
- **Subject-Specific Baselines**: In `ml_pipeline/dataset.py`, we updated the `DrowsinessDataset` to group the geometric data by `subject_id` (extracted from the filename).
- **Max Baseline Extraction**: For each subject, we compute the 95th percentile (to ignore outliers) of their EAR, MAR, eyebrow distance, and head tilt across all their images in the dataset. This serves as an approximation of their "wide-open/alert" baseline.
- **Dynamic Normalization**: During training and evaluation, the raw features for an image are normalized by dividing them by the subject's baseline maximum (`norm = val / baseline_max`). This transforms the features into a standardized 0.0 – 1.0 scale relative to the specific driver's natural state.

---

## 2. Phase 1: Adversarial Domain Generalization (GRL)
### Why we did this
While baseline normalization fixes the geometric inputs, the EfficientNet CNN backbone processing the raw images was still prone to overfitting on visual identity markers (e.g., glasses, facial hair). We needed a way to force the CNN to extract features that are informative for predicting drowsiness but completely uninformative about *who* the driver is.

### What we did
- **Gradient Reversal Layer (GRL)**: Added a custom `autograd.Function` in `ml_pipeline/models.py`. During the forward pass, it acts as an identity function. During the backward pass, it multiplies the gradients by a negative scalar (`-alpha`).
- **FusionGRLModel**: Created a new architecture extending the Fusion Model. We attached a secondary **Domain Classification Head** that takes the fused features and tries to predict the identity (`domain_label`) of the driver.
- **Adversarial Training**: The Domain Classification Head is placed *after* the GRL.
  - The Domain Head tries to minimize its loss by accurately guessing the driver's identity.
  - Due to the gradient reversal, the CNN backbone receives *inverted* gradients from the Domain Head, forcing it to actively erase identity-specific information from its feature representations.
- **Dynamic Alpha**: In `ml_pipeline/train.py`, we scheduled the GRL strength (`alpha`) to gradually ramp up from 0 to 1 over the course of the training epochs, allowing the primary drowsiness task to stabilize before adversarial confusion kicks in.

## Verification
A successful forward/backward pass verification script confirmed that:
1. The new DataLoaders correctly parse 4 outputs per batch: `(images, geo_features, labels, domain_labels)`.
2. The `FusionGRLModel` correctly predicts both `drowsy_logits` (shape: `[B, 2]`) and `domain_logits` (shape: `[B, 22]`, where 22 is the number of distinct subjects in the training set).
3. Evaluation scripts (`evaluate.py`, `generate_report.py`, `gradcam.py`) were successfully updated to unpack the modified tuple formats and support the new model.
