# Deep Convolutional Generative Adversarial Network (DCGAN)

This repository contains the implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the CelebA dataset. The model generates realistic human face images.

---
## **Dataset Preprocessing Steps**

1. **Download the CelebA Dataset**
   - The dataset used is [CelebA (Large-scale CelebFaces Attributes Dataset)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
   - Ensure that the dataset is placed in the correct directory (e.g., `/kaggle/input/celeba-dataset/img_align_celeba`).

2. **Apply Image Transformations**
   - Resize images to **64×64** resolution.
   - Center crop the images to maintain aspect ratio.
   - Convert images to **tensors**.
   - Normalize pixel values to **[-1, 1]** for stable GAN training.

```python
transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

---
## **How to Train the Model**

1. **Install Dependencies**
   Ensure you have the required libraries installed:
   ```bash
   pip install torch torchvision numpy matplotlib
   ```

2. **Run the Training Script**
   - Execute the Python script to start training:
   ```bash
   python train_dcgan.py
   ```
   - The model will train for 50 epochs by default (can be adjusted).
   - Generated images will be saved every 10 epochs in the output directory.

3. **Training Process**
   - The **discriminator** learns to differentiate between real and fake images.
   - The **generator** learns to create realistic human faces from random noise.
   - The loss for both models is monitored and printed after each epoch.

---
## **How to Test the Model**

1. **Generate New Images**
   - Use the trained generator to create new images from random noise:
   ```python
   import torch
   import torchvision.utils as vutils
   from model import Generator
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   generator = Generator().to(device)
   generator.load_state_dict(torch.load("generator.pth"))
   generator.eval()
   
   noise = torch.randn(25, 100, 1, 1, device=device)
   fake_images = generator(noise)
   vutils.save_image(fake_images, "generated_images.png", normalize=True)
   ```
   - This script generates 25 new images and saves them as `generated_images.png`.

2. **Visualize Generated Images**
   - Load and display generated images using Matplotlib:
   ```python
   import matplotlib.pyplot as plt
   import torchvision.transforms as transforms
   from PIL import Image
   
   img = Image.open("generated_images.png")
   plt.imshow(img)
   plt.axis("off")
   plt.show()
   ```

---
## **Expected Outputs**

- **Epoch 1-4**: The generated images may appear blurry and lack meaningful details.
- **Epoch 5-7**: The faces become more structured with recognizable features.
- **Epoch 7-100**: The model should generate realistic human faces with proper textures and details.
- **Final Output**: A set of human-like faces generated from random noise.


---
## **References**
- **DCGAN Paper**: [Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- **CelebA Dataset**: [CelebA Official Page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
