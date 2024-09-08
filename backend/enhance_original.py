import numpy as np
import skimage as sk
from PIL import Image

# def enhance_original(image: Image.Image):
#     image = np.asarray(image)
#     e1 = equalize_adapthist(image, clip_limit=20)
#     blur = gaussian(e1, sigma=3)
#     fI = denoise_bilateral(blur, win_size=5, sigma_spatial=5, mode='reflect')   
#     fI = np.round(rescale_intensity(fI, out_range=(0, 255))).astype(np.uint8)
#     return Image.fromarray(fI).convert("RGB")

def enhance_original(image: Image.Image):
    image = np.array(image)
    """
    image = sk.exposure.equalize_adapthist(image, clip_limit=40)
    image = sk.exposure.adjust_sigmoid(image, 1, 1)
    image = sk.filters.gaussian(image, sigma=4)
    image = sk.restoration.denoise_bilateral(image, win_size=5)
    """
    
    image = sk.exposure.adjust_sigmoid(image, cutoff=0.5, gain=2)
    image = sk.filters.gaussian(image, sigma=2)
    image = sk.morphology.dilation(image, footprint=sk.morphology.disk(10))
    
    image = sk.exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    return Image.fromarray(image).convert('RGB')
