# ResNet18-CBAM-Thyroid-Ultrasound-Classification

This repository presents a deep learning framework for benign–malignant classification of thyroid nodules in ultrasound images based on ResNet18 enhanced with the CBAM attention mechanism.

---

## Dataset
We use the publicly available **TN5000 thyroid ultrasound dataset** introduced by Huan Zhang et al. in  
[TN5000: An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification](https://www.nature.com/articles/s41597-025-05757-4).

---

## Framework
To enhance the classification capability of the baseline ResNet18, we incorporate the Convolutional Block Attention Module (CBAM), which jointly models channel and spatial attention.

<p align="center">
  <img src="figure/fig1.png" width="600"><br>
  <b>Figure 1.</b> Overview of the proposed ResNet18-CBAM framework.
</p>

---

## Training Strategy
We adopt **5-fold cross-validation** to improve model generalization under limited data. The **Adam optimizer** is used with an initial learning rate of **1 × 10⁻⁴**, which is reduced by a factor of **0.5** if the validation performance does not improve for two consecutive epochs.  
To mitigate overconfident predictions and enhance generalization, **cross-entropy loss with label smoothing** (ε = 0.1) is applied during training.

---

## Performance Evaluation
After obtaining the optimal weights from the five folds, all trained models are used to predict each test sample. The final prediction is obtained by averaging the output probabilities from different folds, with a threshold of **0.5** for benign–malignant classification.  
The proposed ResNet18-CBAM model demonstrates strong performance on the test set, achieving **high sensitivity** to malignant nodules and an **AUC of 0.942**, indicating robust overall classification ability.

**Confusion Matrix**
<p align="center">
  <img src="figure/fig2.png" width="420"><br>
  <b>Figure 2.</b> Confusion matrix of the ResNet18-CBAM model on the test set.
</p>

**ROC Curve**
<p align="center">
  <img src="figure/fig3.png" width="420"><br>
  <b>Figure 3.</b> ROC curve of the ResNet18-CBAM model with an AUC of 0.942.
</p>

---

## Model Comparison
The proposed method is compared with the baseline **ResNet18** and a modified ResNet18 incorporating **Squeeze-and-Excitation Blocks (SEB)** under identical training conditions.  
Results show that introducing attention mechanisms leads to consistent improvements across all evaluation metrics. In particular, the **CBAM-based model outperforms the SEB variant**, demonstrating that jointly modeling channel and spatial attention is more effective for thyroid nodule classification.

<p align="center">
  <img src="figure/fig4.png" width="520"><br>
  <b>Figure 4.</b> Performance comparison among ResNet18, SE-ResNet18, and ResNet18-CBAM.
</p>
