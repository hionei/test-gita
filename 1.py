import requests

url="http://95.217.102.100/get_sigma"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data["miner_239"])
else:
    print(f"Error: {response.status_code}")