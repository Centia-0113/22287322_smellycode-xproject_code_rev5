Cross-project prediction of Java code smells on SmellyCode++
This repository contains the code and outputs for a cross-project study of Java code smells using SmellyCode++.

The project compares Random Forest, Logistic Regression, and XGBoost for detecting multiple smells under grouped project validation. The main workflow includes data cleaning, project-level splitting, model tuning, and evaluation on held-out test projects.

1. Data
Download the dataset from figshare:

multi-smell-dataset-v1_2.csv (618 MB)
Source: https://doi.org/10.6084/m9.figshare.28519385
Put it in data/.
2. Setup
Python 3.13 was used. Install dependencies:

pip install -r requirements.txt
3. Run
bash run_all.sh data/multi-smell-dataset-v1_2.csv
The full run takes about 95 minutes on two CPU cores. Outputs are written to work/, and the reference outputs are in results/.

4. Notes
The pipeline uses project-grouped validation and keeps all data-dependent steps inside training folds.
The repository includes both the primary analysis and a smaller post hoc exploratory analysis.
The notebook can be opened in Google Colab from SmellyCode_cross_project.ipynb.
