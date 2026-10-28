# Machine Learning Lab Experiments 5, 8 and 11

Python implementations based on the lab manuals provided.

- **Experiment 5:** Multiclass Classification using Logistic Regression
- **Experiment 8:** Decision Tree Classifier
- **Experiment 11:** Random Forest Classifier, compared with a single Decision Tree

## Requirements
Python 3.x, scikit-learn, and matplotlib.

Install:
```bash
pip install -r requirements.txt
```

## Dataset
All programs use the Iris dataset built into scikit-learn, so a separate CSV is not needed. It has four flower measurements and three target classes.

## Run
```bash
python experiment5_multiclass_classification.py
python experiment8_decision_tree.py
python experiment11_random_forest.py
```
If needed, use `python3` instead of `python`.

## Upload to GitHub
1. Create a repository named `ml-lab-experiments-5-8-11`.
2. Select **Add file → Upload files**.
3. Upload the three Python files, `requirements.txt`, and `README.md`.
4. Click **Commit changes**.

## What each script does
- Experiment 5: trains Logistic Regression and prints accuracy, confusion matrix, and classification report.
- Experiment 8: trains and visualizes a Decision Tree, and prints classification metrics.
- Experiment 11: trains Random Forest, reports metrics, compares accuracy with one Decision Tree, and plots feature importance.

Run the scripts to obtain the actual results on your computer.
