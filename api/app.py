from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import pandas as pd

from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "diabetes_model.pkl"
)


# ============================================================
# LIVE DATA PATH
# ============================================================

LIVE_DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


LIVE_DATA_FILE = os.path.join(
    LIVE_DATA_DIR,
    "live_predictions.csv"
)


# ============================================================
# DRIFT STATUS PATH
# ============================================================

DRIFT_STATUS_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "drift_status.txt"
)


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(
    LIVE_DATA_DIR,
    exist_ok=True
)


os.makedirs(
    os.path.join(
        BASE_DIR,
        "reports"
    ),
    exist_ok=True
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

prediction_counter = Counter(
    "diabetes_prediction_total",
    "Total number of diabetes predictions"
)


diabetes_counter = Counter(
    "diabetes_risk_total",
    "Total number of diabetes risk predictions"
)


no_diabetes_counter = Counter(
    "no_diabetes_risk_total",
    "Total number of no diabetes risk predictions"
)


# ============================================================
# DRIFT GAUGE
# ============================================================

drift_gauge = Gauge(
    "diabetes_drift_detected",
    "Whether Evidently detected dataset drift"
)


# ============================================================
# LOAD DRIFT STATUS
# ============================================================

def update_drift_metric():

    try:

        if os.path.exists(
            DRIFT_STATUS_FILE
        ):

            with open(
                DRIFT_STATUS_FILE,
                "r"
            ) as file:

                value = file.read().strip()

            if value == "1":

                drift_gauge.set(1)

            else:

                drift_gauge.set(0)

        else:

            drift_gauge.set(0)

    except Exception:

        drift_gauge.set(0)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# CSS
# ============================================================

@app.route("/style.css")
def style():

    return send_from_directory(
        FRONTEND_DIR,
        "style.css"
    )


# ============================================================
# JAVASCRIPT
# ============================================================

@app.route("/script.js")
def script():

    return send_from_directory(
        FRONTEND_DIR,
        "script.js"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json()

    try:

        # ----------------------------------------------------
        # GET INPUT FEATURES
        # ----------------------------------------------------

        features = [

            data["Pregnancies"],

            data["Glucose"],

            data["BloodPressure"],

            data["SkinThickness"],

            data["Insulin"],

            data["BMI"],

            data["DiabetesPedigreeFunction"],

            data["Age"]

        ]


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            [features]
        )[0]


        # ----------------------------------------------------
        # COLUMNS
        # ----------------------------------------------------

        columns = [

            "Pregnancies",

            "Glucose",

            "BloodPressure",

            "SkinThickness",

            "Insulin",

            "BMI",

            "DiabetesPedigreeFunction",

            "Age"

        ]


        # ----------------------------------------------------
        # CREATE LIVE DATA
        # ----------------------------------------------------

        live_data = pd.DataFrame(
            [features],
            columns=columns
        )


        # ----------------------------------------------------
        # SAVE / APPEND LIVE DATA
        # ----------------------------------------------------

        if os.path.exists(
            LIVE_DATA_FILE
        ):

            live_data.to_csv(
                LIVE_DATA_FILE,
                mode="a",
                header=False,
                index=False
            )

        else:

            live_data.to_csv(
                LIVE_DATA_FILE,
                mode="w",
                header=True,
                index=False
            )


        # ----------------------------------------------------
        # TOTAL PREDICTION COUNTER
        # ----------------------------------------------------

        prediction_counter.inc()


        # ----------------------------------------------------
        # PREDICTION RESULT
        # ----------------------------------------------------

        if prediction == 1:

            result = "Diabetes Risk"

            diabetes_counter.inc()

        else:

            result = "No Diabetes Risk"

            no_diabetes_counter.inc()


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return jsonify({

            "prediction": int(prediction),

            "result": result

        })


    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400


# ============================================================
# PROMETHEUS METRICS
# ============================================================

@app.route("/metrics")
def metrics():

    update_drift_metric()

    return (
        generate_latest(),
        200,
        {
            "Content-Type":
                CONTENT_TYPE_LATEST
        }
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )