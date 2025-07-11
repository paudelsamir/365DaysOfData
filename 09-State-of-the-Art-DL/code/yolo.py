import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# input folder where your images are
input_folder = Path("/mnt/ExtraData/Github/365DaysOfData/09-State-of-the-Art-DL/images")

# output folder to save detection results
output_folder = Path("/mnt/ExtraData/Github/365DaysOfData/09-State-of-the-Art-DL/detected_images")
output_folder.mkdir(parents=True, exist_ok=True)  # create if not exists

# process each png image in input folder
for image_path in input_folder.glob("*.png"):
    print(f"processing {image_path.name}")
    
    # run detection
    results = model(str(image_path))
    results.print()
    results.render()
    
    # convert to PIL image
    img = Image.fromarray(results.ims[0])
    
    # save detected image to output folder with same name
    save_path = output_folder / image_path.name
    img.save(save_path)
    print(f"saved detected image to {save_path}")
    
    # display image
    plt.imshow(img)
    plt.axis('off')
    plt.title(image_path.name)
    plt.show()
