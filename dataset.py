import normalizePhoto
import normalizeText
import os
import normalizePhoto
from PIL import Image
def create_dataset():
    path = "dataset/other"
    for i in os.listdir(path):
        if "jpg" in i:
            img = normalizePhoto.normalizePhoto(f"{path}/{i}")
            text = normalizeText.get_text(img)
            f = open(f"text_{path}/{i.replace('.jpg','.txt')}","w",encoding="utf-8")
            f.write(text)
            f.close()
            print(i,normalizeText.normalize(text))

    path = "dataset/bill"
    for i in os.listdir(path):
        if "jpg" in i:
            img = normalizePhoto.normalizePhoto(f"{path}/{i}")
            text = normalizeText.get_text(img)
            f = open(f"text_{path}/{i.replace('.jpg','.txt')}","w",encoding="utf-8")
            f.write(text)
            f.close()
            print(i,normalizeText.normalize(text))

    path = "dataset/facture"
    for i in os.listdir(path):
        if "jpg" in i:
            img = normalizePhoto.normalizePhoto(f"{path}/{i}")
            text = normalizeText.get_text(img)
            f = open(f"text_{path}/{i.replace('.jpg','.txt')}","w",encoding="utf-8")
            f.write(text)
            f.close()
            print(i,normalizeText.normalize(text))

def get_dataset():
    X = []
    y = []
    for dirname, _, filenames in os.walk('text_dataset'):
        for filename in filenames:
            with open(os.path.join(dirname, filename), "r") as f:

                X.append(f.read())
                if "bill" in os.path.join(dirname, filename):
                    y.append([1,0,0])
                elif "facture" in os.path.join(dirname, filename):
                    y.append([0,1,0])
                elif "other" in os.path.join(dirname, filename):
                    y.append([0,0,1])
    return X, y