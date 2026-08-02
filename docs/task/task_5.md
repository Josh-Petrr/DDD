# Task 5: Aggressive Generalization

## The Problem: Identity Memorization & Overfitting
During Task 4, we observed a massive generalization gap in the `FUSION_GRL_3` model:
- **Training Accuracy**: ~99.95%
- **Validation Accuracy**: ~50.89%
- **Test Accuracy**: 71.22%

Because the Driver Drowsiness Dataset (DDD) has highly correlated frames for a small number of unique subjects (~22 in training), the large 4M-parameter EfficientNet-B0 backbone was simply memorizing specific driver identities (skin tone, lighting, glasses, head shape) rather than learning universal drowsiness cues. 

While the GRL (Gradient Reversal Layer) helped improve Test Recall to 81%, the network was still fundamentally overfitting.

---

## The Solution: 4-Pronged Aggressive Generalization
To force the model to focus on structural drowsiness cues (e.g., eye closure, mouth gap) rather than driver identities, we implemented the following aggressive regularization techniques in a new model variant: `FUSION_GRL_V4`.

### 1. Identity-Destroying Augmentations (`dataset.py`)
We added specific transforms designed to destroy the visual shortcuts the CNN was using to identify drivers:
- **`RandomGrayscale (p=0.3)`**: Forces the model to ignore skin tone and vehicle lighting 30% of the time, forcing reliance on structural features.
- **`GaussianBlur`**: Removes fine-grained textures like facial hair, pores, and wrinkles.
- **`RandomPerspective`**: Warps the face to simulate extreme camera angles, preventing the model from memorizing a specific driver's natural resting head pose.
- **Aggressive `RandomErasing`**: Increased the probability (to 40%) and maximum size of blacked-out patches. This prevents the model from relying entirely on a single feature (e.g., just the left eye) and forces it to learn holistic cues.

### 2. Batch-Level CutMix Augmentation (`train.py`)
We implemented a custom **CutMix** function in the training loop.
- **What we did**: With a 50% probability during training, a rectangular patch from one image is physically pasted onto another image. 
- **Why**: The labels for both the drowsiness class and the domain identity are mixed proportionally to the patch size (soft labels). This state-of-the-art technique makes it virtually impossible for the CNN to confidently memorize an entire image, forcing it to find discriminative features anywhere on the face.

### 3. High-Penalty Regularization (`config.py` & `models.py`)
We increased the standard regularization hyperparameters to heavily penalize memorization:
- **Dropout (`DROPOUT_RATE_V4`)**: Increased from `0.3` to `0.5` in the fusion and domain heads. Dropping 50% of the neurons during every forward pass forces the network to build highly redundant, robust feature pathways.
- **Weight Decay (`WEIGHT_DECAY_V4`)**: Increased the L2 penalty from `1e-3` to `5e-3`. This prevents weights from growing too large and complex, forcing a simpler and more generalized decision boundary.

### 4. Partial Backbone Freezing (`models.py` & `train.py`)
- **What we did**: Added an `unfreeze_backbone_partial(num_blocks=3)` method. During Stage 2 (fine-tuning), instead of unfreezing the entire EfficientNet backbone, we now only unfreeze the **last 3 blocks**.
- **Why**: The early blocks of a pre-trained ImageNet model act as excellent generic edge and shape detectors. By keeping them frozen permanently, we prevent the model from destroying its pre-trained weights to memorize dataset-specific identities, severely limiting its capacity to overfit while still allowing the final layers to adapt to facial features.

---

## Expected Outcome
By applying these techniques simultaneously to `FUSION_GRL_V4`, we expect:
1. The Training Accuracy to no longer hit 99-100%.
2. A significant reduction in the gap between Training and Validation accuracy.
3. An overall improvement in cross-subject Test Accuracy and Test Recall.
