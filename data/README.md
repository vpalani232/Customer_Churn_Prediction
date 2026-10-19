# Data

Raw data is **not stored in GitHub**. It lives in S3 and is documented here.

**Source:** Kaggle "Customer Churn Dataset" (muhammadshahidazeem), a training file and a testing file.

| Item | Location |
|---|---|
| Training file | `s3://<bucket>/churn/raw/train/customer_churn_dataset-training-master.csv` |
| Testing file (held-out "production" data) | `s3://<bucket>/churn/raw/test/customer_churn_dataset-testing-master.csv` |
| Cleaned data and splits | `s3://<bucket>/churn/processed/` |
| Athena database / tables | `customer_churn_db.churn_train`, `customer_churn_db.churn_test` |
| Checksums | `s3://<bucket>/churn/raw/checksums.json` |

**Columns:** CustomerID, Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay,
Subscription Type, Contract Length, Total Spend, Last Interaction, Churn (target).

To reproduce, download the two CSV files from Kaggle into `data/local/` (git-ignored) and run
`notebooks/01_data_ingestion_athena.ipynb`.
