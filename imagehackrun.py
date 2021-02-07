from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request
from imagehack import *
import time
import requests
import json

class Item(BaseModel):
    id: str
    url: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/items/")
async def create_item(item: Item):
    print(item)
    cap = ""
    logf = open("logs/image_caption_{0}.log".format(time.time()), "w")
    try:
        logf.write("Input Data: {0}\n".format(str(item)))
        url = item.url.replace(" ","%20")
        print("ID:",item.id)
        logf.write("updated Url: {0}\n".format(url))
    except:
        url = item.url
        logf.write("Exception occured at checking url format: {0} \n".format(item.url))

    try:
        logf.write("image downloading: {0} \n".format(url))
        urllib.request.urlretrieve(url, "testing.jpg")
        logf.write("image downloaded \n")
        cap =  predict_captions("testing.jpg")
        logf.write("caption predicted:- {0} \n".format(cap))
        print ('Normal Max search:', cap) 

        
    except Exception as e:
        print("Error:",e)
        logf.write("Caption error \n")
        cap = ""
    header = {'Content-Type': 'application/json'}
    response_url = "http://35.154.145.86/process-completion-hook"
    response_data = {
            "id":item.id,
            "caption": cap
        }

    try:
        r = requests.post(response_url, data=json.dumps(response_data),headers=header)
        print(r.status_code, r.reason, r.json())
    except Exception as e:
        print("error in sending post msg to database",e)

    return {"id":item.id, "url":item.url, "output":cap}

@app.get("/ping")
async def create_item():

    return {"ping":"pong"}

import uvicorn

uvicorn.run(app, port=8000)