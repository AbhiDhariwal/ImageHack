from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request
from imagehack import *



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
    try:
        url = item.url.replace(" ","%20")
    except:
        url = item.url
    try:
        urllib.request.urlretrieve(url, "testing.jpg")
        cap =  predict_captions("testing.jpg")
        print ('Normal Max search:', cap) 
    except Exception as e:
        print("Error:",e)
        cap = ""

    return {"id":item.id, "url":item.url, "output":cap}

@app.get("/ping")
async def create_item():

    return {"ping":"pong"}

import uvicorn

uvicorn.run(app, port=8000)