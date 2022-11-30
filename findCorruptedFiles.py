from struct import unpack
#from tqdm import tqdm
import os
import shutil

#Iterates through a folder to determine if an image is corrupt. 
#Used Yasoob's script. https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


#JPEG class was copied from https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/
class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            #open the image_data, >H tells struct to treat the data as big-endian and as unsigned short.
            #Read first three big-endian values
            marker, = unpack(">H", data[0:2])
            #Check if marker is the start of the image
            if marker == 0xffd8:
                data = data[2:]
            #Check if marker is the end of the image, end decode
            elif marker == 0xffd9:
                return
            #Check if marker is the start of Scan, skips to end of file.
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        


bads = []

imagesDir=["TrainingDataset/CVDfriendly", "TrainingDataset/CVDunfriendly", "TestImages/CVDfriendly","TestImages/CVDunfriendly"]

#Iterate through all directories in imagesDir and removes the bad images to a Corrupt Image folder.
for images in imagesDir:    
    for img in os.listdir(images):
        image = os.path.join(images,img)
        image = JPEG(image)
        try:
            image.decode()
        except:
            bads.append(img)    

    for name in bads:
        print(name,images)
        file = os.path.join(images,name)
        dst = os.path.join("CorruptImages",name)
        shutil.move(file, dst)
    #os.remove(os.path.join(images,name))