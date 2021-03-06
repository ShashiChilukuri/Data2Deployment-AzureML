import requests
import json

scoring_uri = 'http://7d0ae494-b0b6-4b06-ade9-7fad47ff694e.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = ''

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 61,
            "anaemia": 0,
            "creatinine_phosphokinase": 582,
            "diabetes": 1,
            "ejection_fraction": 38,
            "high_blood_pressure": 0,
            "platelets": 147000.0,
            "serum_creatinine": 1.2,
            "serum_sodium": 141,
            "sex": 1,
            "smoking": 0,
            "time": 237
          },
          {
            "age": 49.0,
            "anaemia": 0,
            "creatinine_phosphokinase": 972,
            "diabetes": 1,
            "ejection_fraction": 35,
            "high_blood_pressure": 1,
            "platelets": 268000,
            "serum_creatinine": 0.8,
            "serum_sodium": 130,
            "sex": 1,
            "smoking": 0,
            "time": 187
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())