# Machine_Learniing.py

Machine learning lab exercises in Python, exported from Jupyter notebooks. They cover text classification, classic supervised classifiers, clustering, concept learning, a genetic algorithm, and a small neural network.

## Contents

| File | Description |
| --- | --- |
| `EXP_1.py` | Text classification on the 20 Newsgroups dataset (fetched through scikit-learn) with a TF-IDF and Multinomial Naive Bayes pipeline; prints a classification report, plots a confusion matrix heatmap, and predicts categories for custom sentences. |
| `EXP_5.py` | Diabetes prediction (`diabetes.csv`) with Linear Discriminant Analysis and Logistic Regression; reports classification metrics, confusion matrices, and accuracy for both models, and predicts a new record. |
| `EXP_6.py` | Diabetes prediction with a Decision Tree classifier, including a plot of the fitted tree. |
| `EXP_7.py` | Diabetes prediction with a Random Forest classifier (100 trees, max depth 2). |
| `EXP_8.py` | Customer segmentation with K-Means on `Mall_Customers.csv`: one-hot encoding, scatter plots, the Elbow method for choosing k, clustering into 5 groups, and SSE output. |
| `EXP_9.py` | Genetic algorithm with PyGAD that searches for weights so that a weighted sum of six inputs approaches a target value of 44. |
| `EXP_12A.py` | Find-S concept learning algorithm on the `ENJOYSPORT.csv` dataset. |
| `EXP_12B.py` | Candidate Elimination algorithm on the `ENJOYSPORT.csv` dataset, printing the specific and general hypotheses at each step. |
| `LAB TEST 3.py` | Sentiment classification of short statements (`Finite Words.csv`) using a bag-of-words `CountVectorizer` and an `MLPClassifier`, followed by prediction for a user-entered statement. |

## Datasets

The CSV datasets are not included in this repository; they are public datasets such as those available on [Kaggle](https://www.kaggle.com/). Place each CSV (for example `diabetes.csv`) in the folder you run the scripts from, or change the `pd.read_csv(...)` path in the script. `EXP_1.py` downloads its dataset automatically through scikit-learn.

## Tech stack

- Python 3
- NumPy, pandas
- scikit-learn
- Matplotlib, seaborn
- PyGAD (`EXP_9.py` only)

## Getting started

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pygad
python EXP_6.py
```

Notes:

- The files are notebook exports and keep the `# In[ ]:` cell markers. Some cells rely on notebook display behaviour (a bare expression as the last line), so certain intermediate outputs are only shown when run in Jupyter.
- `EXP_9.py` contains a notebook-only `pip install pygad` line, which must be removed (after installing PyGAD) before running it as a plain Python script.

## License

No license has been specified for this project.
