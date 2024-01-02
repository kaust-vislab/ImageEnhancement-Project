from PIL import Image
import numpy as np
from torchvision import transforms


def convert_to_pil(path):
    image = Image.open(path)
    return image

def normalize(image):
    image = (image-(31.555462*1000)) /(14.908989*1000)
    return image

def to_numpy(image):
    return np.asarray(image).astype(np.float32)


def wrapper_function(image):
    low = np.random.randint(200,300,1)[0]
    high =  np.random.randint(200,300,1)[0]
    resize_transform = transforms.Resize((low,high))
    return resize_transform(image)