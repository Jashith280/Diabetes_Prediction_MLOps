import pandas as pd
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "X_train.csv"
)

LIVE_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "live_predictions.csv"
)


# ============================================================
# REPORT PATH
# ============================================================

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

REPORT_FILE = os.path.join(
    REPORT_DIR,
    "drift_report.html"
)

# File used by Flask/Prometheus
DRIFT_STATUS_FILE = os.path.join(
    REPORT_DIR,
    "drift_status.txt"
)


# ============================================================
# CHECK LIVE DATA
# ============================================================

if not os.path.exists(LIVE_DATA):

    print("ERROR: live_predictions.csv not found!")
    print("Make at least one prediction using Flask.")
    exit()


# ============================================================
# LOAD DATA
# ============================================================

reference_data = pd.read_csv(
    TRAIN_DATA
)

current_data = pd.read_csv(
    LIVE_DATA
)


print(
    "Reference data loaded:",
    reference_data.shape
)

print(
    "Live prediction data loaded:",
    current_data.shape
)


# ============================================================
# CREATE EVIDENTLY REPORT
# ============================================================

data_drift_report = Report(
    metrics=[
        DataDriftPreset()
    ]
)


# ============================================================
# RUN DRIFT DETECTION
# ============================================================

data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data
)


# ============================================================
# GET EVIDENTLY RESULTS
# ============================================================

results = data_drift_report.as_dict()

dataset_drift = False
drifted_columns = 0
total_columns = len(current_data.columns)


# Search through Evidently metrics
for metric in results.get("metrics", []):

    result = metric.get("result", {})

    if "dataset_drift" in result:

        dataset_drift = result["dataset_drift"]

    if "number_of_drifted_columns" in result:

        drifted_columns = result[
            "number_of_drifted_columns"
        ]


# ============================================================
# SAVE DRIFT STATUS
# ============================================================

if dataset_drift:

    drift_value = 1

else:

    drift_value = 0


with open(
    DRIFT_STATUS_FILE,
    "w"
) as file:

    file.write(str(drift_value))


# ============================================================
# SAVE HTML REPORT
# ============================================================

data_drift_report.save_html(
    REPORT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print(
    "Evidently live data drift analysis completed!"
)

print(
    "Drifted columns:",
    drifted_columns,
    "/",
    total_columns
)

print(
    "Dataset drift:",
    dataset_drift
)

print(
    "Prometheus drift value:",
    drift_value
)

print(
    "Drift report saved successfully!"
)

print(
    REPORT_FILE
)

print(
    "Drift status saved successfully!"
)

print(
    DRIFT_STATUS_FILE
)