import json
import requests
# from datetime import date
# from datetime import timedelta

# dateToday = date.today() - timedelta(days=1)
# dateToday = str(dateToday)
# print(dateToday)
# response = requests.get("https://api.covid19india.org/v4/data-"+dateToday+".json")
# data = response.json
# print(data)

response = requests.get("https://api.covidindiatracker.com/state_data.json").json()
data = response

cases = {}

for res in data:
    state_code = res['id']
    state_name = res['state']
    active = res['active']
    cases[state_code] = state_name , active

print(cases['IN-MH'])


    


