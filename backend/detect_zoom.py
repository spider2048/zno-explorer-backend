import logging
import os
import re
from typing import List

import pytesseract
from PIL import Image
from PIL.Image import Image as TyImage

pytesseract.pytesseract.tesseract_cmd = '../standalone/tesseract/tesseract.exe'

logger = logging.getLogger(__name__)

os.environ["TESSDATA_PREFIX"] = '../standalone/tesseract/tessdata'

def split_images(image: TyImage) -> List[TyImage]:
    cropped_image = image.crop((0, 0, image.width, image.height - 64))
    text_image = image.crop((0, image.height - 64, image.width, image.height))

    image_text = pytesseract.image_to_string(text_image)
    zoom = re.search(r"(\d+),(\d+)", image_text).groups()
    zoom = int("".join(zoom))

    logging.info("detected zoom level %d", zoom)

    if zoom == 200000:
        logging.info("returning %d images", 1)
        return [cropped_image]
    factor = 200000 // zoom

    w, h = cropped_image.width, cropped_image.height

    segw = int(w // factor)
    segh = int(h // factor)

    cropped_images = []
    
    for x in range(0, w, segw):
        for y in range(0, h, segh):
            crop = (x, y, x+segw, y+segh)
            cropped_images.append(
                cropped_image
                .crop(crop)
                .resize((1280, 960))
            )
    
    logging.info("returning %d images", len(cropped_images))
    return cropped_images


if __name__ == "__main__":
    from PIL import Image
    cropped_images = split_images(Image.open(r"D:\Projects\RodDetection\Pictures\Growth100W_B_2_1_100000.bmp"))

    print("total =", len(cropped_images))

