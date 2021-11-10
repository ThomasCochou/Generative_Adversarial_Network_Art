import requests, os
import pandas as pd

col_list = ["AUTHOR","BORN-DIED","TITLE","DATE","TECHNIQUE","LOCATION","URL","FORM","TYPE","SCHOOL","TIMEFRAME"]
df = pd.read_csv("catalog.csv", usecols=col_list)

ID_URL = 6
ID_FORM = 7
ID_TYPE = 8
ID_TIMEFRAME = 10

DATASET_MAXSIZE = 4000	# /!\ MAX = 51459
output_path = "wga"

form_feature = "painting"
type_feature = "landscape"

type_list = []
form_list = []

for type_string in df["TYPE"] :
	if type_string not in type_list :
		type_list.append(type_string)

print("TYPE = "+str(type_list))

for form_string in df["FORM"] :
	if form_string not in form_list :
		form_list.append(form_string)

print("FORM = "+str(form_list))

i = 0

if not os.path.exists(output_path):
	os.makedirs(output_path)

for data in df.iterrows():
	print("DATASET = "+str(i)+"/"+str(DATASET_MAXSIZE), end="\r")
	if i < DATASET_MAXSIZE: 
		if  data[1][ID_FORM] == form_feature:
			if data[1][ID_TYPE] == type_feature :
				url = data[1][ID_URL]
				url = url.split("html")[0] + "art" + url.split("html")[1] + "jpg"
				filename = url.split('/')[-1]
				r = requests.get(url, allow_redirects=True)
				open(output_path+"/"+filename, 'wb').write(r.content)
				i = i + 1
	else :
		pass