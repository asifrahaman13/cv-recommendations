import json
data={}
with open("cv_data.json", "r") as file:
    data=json.load(file)

# print(data)


company_and_job_desctiptions={}

for idx, item in enumerate(data):
    # print("*************************************************************************************************************8")
    # for i in range(5):
    #     print()
    # print(item['company_name'])
    # print(item['job_description'])
    company_and_job_desctiptions[item['company_name']]=item['job_description']
# print(company_and_job_desctiptions)



for key, value in company_and_job_desctiptions.items():
    print(value)