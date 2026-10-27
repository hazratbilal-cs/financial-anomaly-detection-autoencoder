# Hybrid Autoencoder + DBSCAN for Financial Anomaly Detection

Bachelor's thesis project (BS Computer Science, 2022-2026), Govt. Degree College Lal Qilla, affiliated with the University of Malakand, Pakistan.
Supervisor: Dr. Hanif Ur Rahman

**Team:** Hazrat Bilal, Muhammad Ayub, Tariq Ullah

## Overview
Fraud is rare and constantly changing, so labeled fraud data is scarce. This project studies unsupervised and deep learning methods that learn what *normal* transactions look like and flag deviations. We compare 8 models and propose a hybrid that combines Autoencoder reconstruction error (global anomalies) with DBSCAN noise detection (local density outliers).

## Dataset
- 333,320 financial transactions, [dataset name and Kaggle link]
- Fraud is about 0.057% of records (highly imbalanced)
- Features: step, type, amount, old/new balance (origin and destination), isFraud (used for evaluation only)

## Pipeline
1. Quality checks: missing values, duplicates, invalid entries
2. Dropped ID columns (nameOrig, nameDest); label-encoded `type`
3. Scaling (StandardScaler / MinMaxScaler); outliers capped at 5th-95th percentile
4. Feature engineering: balance-change ratios, transaction velocity indicators
5. Feature analysis: Pearson correlation, chi-square and Cramer's V
6. 80/20 stratified split; hyperparameters tuned with GridSearchCV / RandomizedSearchCV, k-fold CV, early stopping
7. Evaluation on the held-out test set

## Models
Isolation Forest, One-Class SVM, Local Outlier Factor, K-Means, DBSCAN, Autoencoder, LSTM Autoencoder, **Hybrid Autoencoder + DBSCAN**. Random Forest is used as a supervised reference.

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.0013 | 0.0214 | 0.0024 | 0.2659 |
| One-Class SVM | 0.0489 | 0.1905 | 0.0779 | 0.5668 |
| LOF | 0.0267 | 0.1064 | 0.0426 | 0.5634 |
| K-Means | 0.0056 | 0.1383 | 0.0108 | 0.7629 |
| DBSCAN | 0.0103 | 0.1277 | 0.0191 | 0.5555 |
| Autoencoder | 0.0060 | 0.1489 | 0.0116 | 0.7593 |
| LSTM Autoencoder | 0.0039 | 0.0978 | 0.0075 | 0.4606 |
| **Hybrid AE + DBSCAN** | 0.0056 | 0.1489 | 0.0107 | **0.7821** |

**Key findings**
- The hybrid model achieved the **highest ROC-AUC (0.7821)**, followed by K-Means (0.7629) and the Autoencoder (0.7593).
- Accuracy (94-99%) is misleading because of the extreme class imbalance.
- Precision stays low for all unsupervised models (many false positives), and recall is limited (about 15% for the hybrid). Detecting very rare fraud remains hard.
- One-Class SVM had the best F1 (0.0779).
- Most influential features: step, old balance, new balance.

## Limitations and future work
Adaptive thresholding and cost-sensitive learning to reduce false positives, PR-AUC evaluation, richer behavioural and graph features (GNN), explainability with SHAP/LIME, concept-drift testing, real-time deployment.

## How to run
```bash
conda env create -f environment.yml
conda activate fraud-anomaly
jupyter notebook
```
Open `thesis_anomaly_detection.ipynb` (or open in Google Colab) and run all cells. Download the dataset from [link] and place it in the `data/` folder.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Google Colab

## Contact
Hazrat Bilal, hazratbilal.cs.pk@gmail.com
