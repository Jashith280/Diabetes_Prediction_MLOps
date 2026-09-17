import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Pregnancies": 4,
    "Glucose": 129,
    "BloodPressure": 75,
    "SkinThickness": 30,
    "Insulin": 120,
    "BMI": 26.7,
    "DiabetesPedigreeFunction": 0.28,
    "Age": 37
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())