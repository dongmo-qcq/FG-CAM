import json
from importlib_resources import path

class ImageNet():

    def __init__(self):
        self.index_to_class={}
        
        with open("imagenet_class_index.json", 'r') as source:
            data = json.load(source)

        for index, (_, class_name) in data.items():
            class_name = class_name.lower().replace('_', ' ')
            self.index_to_class[int(index)] = class_name
    
    def get_class_name(self,index):
        return self.index_to_class[index]
