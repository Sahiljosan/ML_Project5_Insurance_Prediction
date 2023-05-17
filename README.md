# ML_Project5_Insurance_Prediction
- [Day 3](#day-3)

## Insert Data into MongoDB
```
import pymongo 
import pandas as pd
import json

client = pymongo.MongoClient("mongodb+srv://sahil_josan:samongodbhil5@cluster0.sptya9h.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE_PATH = (r"G:\Udemy\DATA SCIENCE ineuron\VS Code\ML_Project5_Insurance_Prediction\insurance.csv")

DATABASE_NAME = "INSURANCE"
														
COLLECTION_NAME = "INSURANCE_PROJECT"					


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    df.reset_index(drop = True, inplace = True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

```

## Day 3
After logger and Exception handling , we will fetch data from mongoDB <br>
`step1` go to utils and write code
`Step2` Create one file .env and store mongoDB client.
`step3` go to Insurance_Prediction.config.py file <br>
In order to read data from mongoDB, we will define our client in mongoDB from where we are going to fetch the data.
`step4` Write code in main.py 
`step5` install all libraries using pip install -r requirements.txt
`step6` Run python main.py file <br>
After this all data will be ingested from mongoDB server
