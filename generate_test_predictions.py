import requests
import random
import time

API_URL = "http://127.0.0.1:5000/predict"

for i in range(30):

    data = {
        "Pregnancies": random.randint(0, 10),
        "Glucose": random.randint(80, 180),
        "BloodPressure": random.randint(50, 100),
        "SkinThickness": random.randint(10, 50),
        "Insulin": random.randint(0, 250),
        "BMI": round(random.uniform(18, 40), 1),
        "DiabetesPedigreeFunction": round(
            random.uniform(0.1, 1.5), 3
        ),
        "Age": random.randint(20, 70)
    }

    response = requests.post(
        API_URL,
        json=data
    )

    print(
        f"Prediction {i + 1}:",
        response.json()
    )

    time.sleep(0.5)