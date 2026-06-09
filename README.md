# NST-VGG-Activations: Visualizing Deep Learning Representations

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Overview
This repository contains a PyTorch implementation focused on **Neural Style Transfer (NST)** using the VGG-19 Convolutional Neural Network. Beyond simply generating stylized images, the primary objective of this project is to extract, compare, and visualize the internal feature maps (activations) from multiple layers of the network. 

By analyzing these activations, this project demonstrates how deep learning models hierarchically encode image content—capturing low-level textures in shallow layers and high-level spatial structures in deeper layers during the style transfer optimization process.

## 🧠 Technical Methodology

The project utilizes a pre-trained VGG-19 network as a static feature extractor. The optimization does not update the network weights; instead, it updates the pixel values of the input image using backpropagation to minimize a custom objective function.

### Mathematical Formulation
The total loss function is a linear combination of the Content Loss ($L_{content}$) and the Style Loss ($L_{style}$):

$$L_{total} = \alpha L_{content}(C, G) + \beta L_{style}(S, G)$$

Where $C$ is the Content Image, $S$ is the Style Image, $G$ is the Generated Image, and $\alpha, \beta$ are the weighting coefficients.

**1. Content Loss:**
Calculated using the Mean Squared Error (MSE) between the feature representations of the content image and the generated image at a specific deep layer (typically `conv4_2`).

**2. Style Loss & The Gram Matrix:**
To capture style (texture, color, and brushstrokes) irrespective of spatial structure, the network relies on the Gram Matrix ($G_{ij}$), which computes the inner product between vectorized feature maps:

$$G_{ij}^l = \sum_k F_{ik}^l F_{jk}^l$$

The Style Loss is the MSE between the Gram matrices of the style image and the generated image across multiple layers (e.g., `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`).

## 📁 Repository Structure

```text
NST-VGG-Activations/
├── NST_Code/             # Core scripts for feature extraction and optimization
├── code.ipynb            # Interactive Jupyter Notebook for visualization and execution
├── requirements.txt      # Python dependencies
├── Procfile.txt          # Deployment configuration
└── README.md             # Project documentation

```

## 🚀 Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/aryan2050/NST-VGG-Activations.git](https://github.com/aryan2050/NST-VGG-Activations.git)
cd NST-VGG-Activations

```


2. **Install dependencies:**
Ensure you have a Python 3.x environment. Install the required packages via:
```bash
pip install -r requirements.txt

```


3. **Run the Notebook:**
Launch Jupyter to explore the code and visualizations interactively:
```bash
jupyter notebook code.ipynb

```



## 📊 Visualizing Activations

The `code.ipynb` notebook includes extensive `matplotlib` routines to plot the tensors extracted from the VGG layers.

* **Shallow Layers:** Visualizations demonstrate the network's focus on basic edges, colors, and simple patterns.
* **Deep Layers:** Visualizations reveal complex geometries, object boundaries, and spatial arrangements, abstracting away the exact pixel values.

## 🛠️ Built With

* **PyTorch:** Model architecture, tensor operations, and backpropagation.
* **Torchvision:** Pre-trained VGG-19 weights and image transformations.
* **Matplotlib:** Plotting feature maps and generation progress.
* **NumPy:** Matrix manipulations.
