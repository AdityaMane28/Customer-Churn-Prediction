# Customer Churn Prediction

This project explores whether a bank customer is likely to leave the bank. The analysis and model are in **Customer Churn Prediction.ipynb**, which uses Logistic Regression to predict the `Exited` target.

## Dataset

The notebook loads `Churn_Modelling.csv` from the current working directory. Keep the CSV in the same folder from which you launch Jupyter (typically the notebook's folder).

The dataset contains 10,000 customer records. `Exited` is the target: `1` means the customer left and `0` means the customer stayed. The notebook's exploratory analysis finds that about 20.4% of customers in the dataset left, so the target classes are imbalanced.

The notebook removes the row number, customer ID, and surname, handles missing numeric values with median imputation, removes duplicate rows, creates a `HasBalance` flag, and label-encodes `Gender`. It drops `Geography` before model training. The model uses customer attributes such as credit score, age, tenure, balance, product count, account activity, salary, and the balance flag.

## Requirements

- Python 3
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install the Python packages with:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

## Run the notebook

1. Place `Customer Churn Prediction.ipynb` and `Churn_Modelling.csv` in the same directory.
2. Open a terminal in that directory and start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open the notebook and run its cells.

## Modeling workflow

1. Explore the dataset, missing values, target balance, and feature relationships.
2. Prepare features and target (`Exited`).
3. Split into training and test sets with an 80/20 split, stratified by the target (`random_state=42`).
4. Fit `StandardScaler` on the training set and apply it to both sets.
5. Train `LogisticRegression(max_iter=1000, random_state=42)`.
6. Evaluate predictions with accuracy, ROC-AUC, a confusion matrix, and a classification report.

## Results in the saved notebook

The notebook's recorded test-set outputs report:

| Metric | Result |
|---|---:|
| Accuracy | 0.8080 |
| ROC-AUC | 0.7698 |
| Churn precision (class 1) | 0.61 |
| Churn recall (class 1) | 0.16 |
| Churn F1-score (class 1) | 0.25 |

The confusion matrix, with rows as actual classes and columns as predicted classes in the order `[0, 1]`, is:

```text
[[1552,   41],
 [ 343,   64]]
```

The model identifies most customers who stay, but misses many customers who leave. Accuracy alone therefore gives an incomplete picture; consider the low churn recall when interpreting this baseline. Threshold tuning, class weighting, and comparing other models could be useful next steps if catching more potential churners is a priority.

## Reproducibility note

In the notebook source, the cleaning section drops `AgeGroup` and `CreditTier`, but the preceding cells do not create those columns. A fresh run from the top may stop with a missing-column error at those statements. Remove the two drop statements or add the intended feature-creation steps before running all cells. Some saved notebook outputs show those columns, so the saved outputs may not reflect the currently visible source code.

This is a notebook-based analysis and baseline model; it does not include a saved model artifact, a prediction API, or deployment code.
## Author

Aditya Mane  
[GitHub profile](https://github.com/AdityaMane28)
