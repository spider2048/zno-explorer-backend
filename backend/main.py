from detectron_predictor import DetectronPredictor

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles
import secrets

logging.basicConfig(
    level=logging.INFO,
    filename="server.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

predictor = DetectronPredictor()

app = FastAPI()
# app.mount("/frontend", StaticFiles(directory="../frontend"))
# app.mount("/assets", StaticFiles(directory="../frontend/assets"))
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_float(text):
    try:
        return float(text)
    except:
        return 1.0

@app.post("/upload")
async def create_upload_file(request: Request):
    request = await request.form()
    image = request["image_file"]
    brigntness = parse_float(request["brightness"])
    contrast = parse_float(request["contrast"])
    scale = parse_float(request.get("scale", 1))
    unit = parse_float(request.get("unit", "nm"))

    # txt_file = request["text_file"]

    logging.info("received mime type %s", image.content_type)

    image_bytes = await image.read()
    with open(f"{secrets.token_hex(64)}.png", "wb+") as fp:
        fp.write(image_bytes)
    # txt_contents = await txt_file.read()

    results = predictor.work("", image_bytes, {
        "brightness": brigntness,
        "contrast": contrast,
        "scale": scale,
        "unit": unit
    })
    return ORJSONResponse(results)
