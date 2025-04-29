# ML Classification and Clustering Tool

A desktop application built in Python using Tkinter that provides an end-to-end workflow for exploratory machine learning on tabular CSV data. With just a few clicks, you can:

- **Upload any CSV dataset** and preview its first 10 rows and dimensions  
- **Select a target column** and configure train/test split for classification  
- **Train a Random Forest classifier**, view accuracy, precision/recall/F1 scores, confusion matrix, and feature importances  
- **Perform K-Means clustering** with a user-specified number of clusters, visualized via PCA  
- **Export results** (classification report, confusion matrix, feature importances) to text files  
- **Save your trained model** as a pickle for later reuse  

---

## Features

- **Data Preview**: Instantly preview your dataset’s rows, columns, and basic metadata.  
- **Automatic Encoding**: Handles categorical features via one-hot encoding (for low-cardinality) or factorization.  
- **Classification Module**:  
  - Configurable train/test split (%)  
  - Random Forest model training with live progress updates  
  - Detailed classification report & confusion matrix  
  - Visual feature-importance bar chart  
- **Clustering Module**:  
  - K-Means clustering on all numeric features  
  - Standard scaling & PCA reduction for visualization  
  - Interactive scatter plot colored by cluster label  
  - Inertia and iteration count displayed  
- **Export & Persistence**:  
  - Save model predictions and metrics to a `.txt` report  
  - Serialize the trained model to `.pkl` for deployment or further analysis  

---

## Tech Stack

- **Python 3.7+**  
- **Tkinter** for GUI  
- **pandas**, **NumPy** for data handling  
- **scikit-learn** (RandomForestClassifier, KMeans, PCA, StandardScaler)  
- **matplotlib** for charts & visualizations  

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/omarfoud/ml-classification-clustering.git
   cd ml-classification-clustering
