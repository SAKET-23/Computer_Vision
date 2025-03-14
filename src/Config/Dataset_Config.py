import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def class_label_2_id(class_label : str):
    label = {
        "Pedestrian" : 0,
        "Car" : 1,
        "Cyclist": 2,
        "Van" : 3,
        "Truck" : 4,
        "Tram" : 5,
        "Misc" : 6
    }
    if class_label in label.keys(): return label[class_label]
    else: return label["Misc"]
    
    
class Kitti(Dataset):
    def __init__(self , folder : str = "Dataset/Object_Detection/train"):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(project_root , folder)
        self.train_image_folder = os.path.join(path, "image")
        self.label_image_folder = os.path.join(path, "label")
        self.image_files = sorted(os.listdir(self.train_image_folder))  # List of image files
        self.label_files = sorted(os.listdir(self.label_image_folder))  # List of label files

        assert len(self.image_files) == len(self.label_files)
    def __len__(self):
        return len(self.image_files)
     
    def label(self , label_file , width, height):
        label_data = []
        target = []
        with open(label_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            text = line.strip().split()
        
            t = {
                "class_id"      : class_label_2_id(text[0]),
                "truncation"    : float(text[1]), # The truncation parameter describes how much of the object is visible in the image. It is a value between 0 and 1
                "occlusion"     : int(text[2]),   #The occlusion parameter indicates how much of the object is hidden by other objects or objects in the scene. This is represented by an integer value between 0 and 3
                "alpha"         : float(text[3]), #The alpha value represents the observation angle of the object relative to the camera. 
                "x1"            : float(text[4]), # xmin
                "y1"            : float(text[5]), # ymin
                "x2"            : float(text[6]), # xmax
                "y2"            : float(text[7]), # ymax
                "length"        : float(text[8]), # 3D Box Dimensions
                "width"         : float(text[9]),
                "height"        : float(text[10]),
                "x"             : float(text[11]), # 3D position
                "y"             : float(text[12]),
                "z"             : float(text[13]),
                "rotation_y"    : float(text[14]) # The rotation_y value represents the object’s rotation around the y-axis in 3D space. 
            }    
            # label_data.append(t)
            target.append(
                {
                   "class_id" : t["class_id"],
                   "bbox"     : [ ((t["x1"] + t["x2"])/2)/width, (t["y1"] + (t["y2"])/2)/height , abs(t["x2"]-t["x1"])/width , abs(t["y2"] - t["y1"])/height]
                    # x_center, y_center , width , height
                }
            )
            
        return target
    
    def __getitem__(self,idx):
        label_path = os.path.join(self.label_image_folder, self.label_files[idx])
        image_path = os.path.join(self.train_image_folder, self.image_files[idx])   
            # Parse the label file and get a list of dictionaries for each image
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        target = self.label(label_path, w, h)
        # print(f"succes_{idx}")
        return image, target
        
if __name__ == "__main__":
    
    dataset = Kitti()

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Get a few samples
    for images, targets in dataloader:
        print(f"Number of images: {len(images)}")
        
        # Check image properties
        for i, img in enumerate(images):
            print(f"Image {i} - Size: {img.size}, Mode: {img.mode}")

        # Check labels
        for i, target in enumerate(targets):
            print(f"Target {i}: {target}")

        break  # Only check the first batch