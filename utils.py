from PIL import Image
from torchvision import transforms

def pad_to_multiple(img: Image.Image, factor: int):
    w,h = img.size
    nw = ((w + factor - 1)//factor)*factor
    nh = ((h + factor - 1)//factor)*factor
    canvas = Image.new("RGB", (nw, nh))
    canvas.paste(img, (0,0))
    return canvas, (w,h)

def tensor_to_pil(tensor):
    return transforms.ToPILImage()(tensor.clamp(0,1))
