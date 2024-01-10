# SharpifyAI
My model for enhancing low resolution images to high resolution images, built in PyTorch using the the CelebA dataset from Kaggle.

## Results:
### Orignial image 
![alt text](http://url/to/img.png)
### SharpigyAI produced image 
![alt text](http://url/to/img.png)

> [!IMPORTANT]
> If you have any issues even when following the documentation below, please reference the google collab as not all the needed code for rendering the images for example is present within this section, also do feel free to contact me @hashemjaber02@gmail.com for any inquries or issues

Introduction

This project presents a neural network model, the SuperResolutionNetworkNew, which is designed to upscale low-resolution images into high-quality, detailed versions. This model is implemented using PyTorch and has been trained on a variety of datasets to ensure robust performance.

Setup

Prerequisites
Python 3.x
PyTorch
PIL (Python Imaging Library)
Matplotlib (for image display)
Installation
Clone the repository:
bash
```
git clone [My Repository URL] TO DO
```

```
pip install torch torchvision pillow matplotlib
```
Model Download
```
Download the pre-trained model from [Link to my model] and place it in the model_face_gen_1 directory inside My project folder.
```
Usage

To use the model for enhancing images:

Load the model:

```Python
model_path = 'path_to_My_model/super_resolution_model_2.pth'
model = torch.load(model_path)
model.eval()
```
Prepare your images:
Place your low-resolution images in a directory.
Ensure the images are in a compatible format (e.g., JPEG, PNG).

Run the model:
Use the provided load_images_parallel function to load and preprocess the images.
Feed the low-resolution images into the model to get high-resolution outputs.
Display the results:
Use the imshow function to display the enhanced images.


Super Resolution Neural Network

Introduction

This project presents a neural network model, the SuperResolutionNetworkNew, which is designed to upscale low-resolution images into high-quality, detailed versions. This model is implemented using PyTorch and has been trained on a variety of datasets to ensure robust performance.

Setup

Prerequisites
Python 3.x
PyTorch
PIL (Python Imaging Library)
Matplotlib (for image display)
Installation
Clone the repository:
bash
Copy code
git clone [Your Repository URL]
Install the required Python packages:
Copy code
pip install torch torchvision pillow matplotlib
Model Download
Download the pre-trained model from [Link to my model] and place it in the model_face_gen_1 directory inside your project folder.

Usage

To use the model for enhancing images:
The model and relevant methods for using it and interacting with the model:
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision import transforms


class SuperResolutionNetworkNew(nn.Module):
    def __init__(self):
        super(SuperResolutionNetworkNew, self).__init__()

        # Early convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)


        self.conv_mid1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_mid2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)



        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Early convolution layers with ReLU and batch normalization
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))

        # Flatten and pass through fully connected layers
        x3 = x2.view(x2.size(0), -1)

        # Reshape and apply skip connection
        x3 = x3.view(x2.size(0), 128, 64, 64)
        x3 = x3 + x2  # Skip connection

        # Upsampling layers with ReLU and batch normalization
        x4 = self.upsample(x3)
        x4 = F.relu(self.bn3(self.conv3(x4)))
        x4 = torch.sigmoid(self.conv4(x4))

        return x4
# Define transforms
transform_hr = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
transform_lr = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def process_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return transform_hr(img), transform_lr(img)

# Load and transform images in parallel -> This chatGPT generated since it really took a long time to download the images let alone preprocess them and train the model with them so I asked it if there was a way to make this proces quicker and this is what it gave me
def load_images_parallel(path, num_images, num_workers=4):
    img_paths = [os.path.join(path, img_file) for img_file in os.listdir(path)[:num_images]]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_image, img_paths))

    images_hr, images_lr = zip(*results)
    return torch.stack(images_hr), torch.stack(images_lr)

# Example usage
path_to_dataset = '/content/drive/MyDrive/CelebADataSet/img_align_celeba/img_align_celeba'
num_images = 500  # Number of images to load, same as the prevous setup no change
images_hr, images_lr = load_images_parallel(path_to_dataset, num_images)

def imshow(tensor, title='example'):
    image = to_pil_image(tensor)
    plt.imshow(image)

    plt.title(title)
    plt.pause(0.001)  # pause to update plots

def testView():

    example_index = number = random.randint(0, 499)
    print(number)
      # You can change this index to view different images

    # Select the low-resolution and high-resolution images
    example_lr = images_lr[example_index].unsqueeze(0)  # Add batch dimension
    example_hr = images_hr[example_index]
    model.eval()
    plt.figure(figsize=(15, 5))

    # Display low-resolution image
    plt.subplot(1, 3, 1)
    imshow(example_lr.squeeze(0), title='Low-Resolution')  # This is the low resolution image

    # Display super-resolved image
    plt.subplot(1, 3, 2)
    with torch.no_grad():
        example_sr = model(example_lr).squeeze(0)
        imshow(example_sr, title='Super-Resolved') # This is the produced image


    # Display high-resolution image
    plt.subplot(1, 3, 3)
    imshow(example_hr, title='High-Resolution') # This is the original image

    plt.show()

def unitTest(test_size):
  for i in range(test_size):
    testView()
```
Load the model:
```Python
model_path = 'path_to_your_model/super_resolution_model_2.pth'
model = torch.load(model_path)
model.eval()
```
Prepare your images:
Place your low-resolution images in a directory.
Ensure the images are in a compatible format (e.g., JPEG, PNG).
Run the model:
Use the provided load_images_parallel function to load and preprocess the images.
Feed the low-resolution images into the model to get high-resolution outputs.
Display the results:
Use the imshow function to display the enhanced images.
Example

Here is an example of enhancing a batch of images:
```Python
path_to_dataset = 'your_dataset_path'
num_images = 100  # Number of images to enhance
images_hr, images_lr = load_images_parallel(path_to_dataset, num_images)
```
# Enhancing images
```Python
for i in range(num_images):
    enhanced_image = model(images_lr[i].unsqueeze(0))
    imshow(enhanced_image, title='Enhanced Image')
``` 
Additional Notes
Some methods wont work since they might need other dependencies or code to work as well as connections to data files, for references you may want to check the google collab where I did the project on.
