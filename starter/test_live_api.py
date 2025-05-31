import requests
import json

def test_api_live():
    # Replace with your deployed API URL
    API_URL = "https://your-app-name.onrender.com"
    
    # Test GET endpoint
    response = requests.get(f"{API_URL}/")
    print("GET Response:", response.json())
    assert response.status_code == 200
    
    # Test POST endpoint
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print("\nPOST Response:", response.json())
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]

if __name__ == "__main__":
    test_api_live()
