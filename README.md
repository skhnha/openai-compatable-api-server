# openai-compatable-api-server
```
***Important*** 
Python 3.11 is required to support stream output.
```


## 1. install requirements
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install requirements.txt
```

## 2. run api server
python main.py

## 3. test
```
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(base_url="http://127.0.0.1:8000/v1", model="gpt-40-mini",api_key="aaa")

# NON-STREAM TEST
llm.invoke('hi')

# STREAM TEST
for x in llm.stream("hi"):
    print(x)
```



