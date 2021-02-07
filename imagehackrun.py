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

    urllib.request.urlretrieve(item.url, "testing.jpg")
    cap =  predict_captions("testing.jpg")
    print ('Normal Max search:', cap) 


    return {"id":item.id, "url":item.url, "output":cap}

@app.get("/ping")
async def create_item():

    return {"ping":"pong"}

import uvicorn

uvicorn.run(app, port=8000)