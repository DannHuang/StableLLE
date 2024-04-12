import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class LLIEdataset(Dataset):
    def __init__(self,
                 data_root,
                 ):
        self.data_root = data_root
        self.target_paths=self._get_image_paths(self.data_root)

        
    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
     
        target_path = self.target_paths[index]
        target_img=Image.open(target_path).convert('RGB')
        target_img = np.array(target_img).astype(np.uint8)
        
        H,W,_=target_img.shape
        target_img=self.crop_to_target_size(target_img,(H,H))
        H=H//64
        W=W//64
        target_img=np.resize(target_img,(H*64,H*64,3))
        
        return  {'png':target_img,'txt':"prompt"}
    
    def _get_image_paths(self,data_root):
        image_paths = []
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    image_path = os.path.join(root, file)
                    image_paths.append(image_path)
        return image_paths

    def crop_to_target_size(self,image, target_size):
        
        original_height, original_width,_ = image.shape
        top = (original_height - target_size[0]) // 2
        left = (original_width - target_size[1]) // 2
        cropped_image = image[top: top + target_size[0], left: left + target_size[1]]

        return cropped_image
    
