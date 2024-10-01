# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7
### Algorithm:

### Step1
Import necessary libraries: OpenCV, NumPy, and Matplotlib.Read an image, convert it to RGB format, define an 11x11 averaging kernel, and apply 2D convolution filtering.Display the original and filtered images side by side using Matplotlib.

### Step2
Define a weighted averaging kernel (kernel2) and apply 2D convolution filtering to the RGB image (image2).Display the resulting filtered image (image4) titled 'Weighted Averaging Filtered' using Matplotlib's imshow function.

### Step3
Apply Gaussian blur with a kernel size of 11x11 and standard deviation of 0 to the RGB image (image2).Display the resulting Gaussian-blurred image (gaussian_blur) titled 'Gaussian Blurring Filtered' using Matplotlib's imshow function.

### Step4
Apply median blur with a kernel size of 11x11 to the RGB image (image2).Display the resulting median-blurred image (median) titled 'Median Blurring Filtered' using Matplotlib's imshow function.

### Step5
Define a Laplacian kernel (kernel3) and perform 2D convolution filtering on the RGB image (image2).Display the resulting filtered image (image5) titled 'Laplacian Kernel' using Matplotlib's imshow function.

## Program:
### Developed By: SARVESH S
### Register Number:212222230135


### 1. Smoothing Filters
### Original Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
image1 = cv2.imread("image.jpeg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

### i) Using Averaging Filter

kernel = np.ones((11, 11), np.float32) / 121
averaging_image = cv2.filter2D(image2, -1, kernel)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(averaging_image)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()

### ii) Using Weighted Averaging Filter

kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

weighted_average_image = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(weighted_average_image)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

### iii) Using gaussian Filter

gaussian_blur = cv2.GaussianBlur(image2, (11, 11), 0)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()

### iv) Using Median Filter

median_blur = cv2.medianBlur(image2, 11)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis("off")
plt.show()

### 2. Sharpening Filters
### i) Using Laplacian Linear Kernal
```
image1 = cv2.imread('image.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel3 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
image5 =cv2.filter2D(image2, -1, kernel3)
plt.imshow(image5)
plt.title('Laplacian Kernel')
```
### ii) Using Laplacian Operator
```
laplacian = cv2.Laplacian(image2, cv2.CV_64F)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()
```

## OUTPUT:
### Original Image

![image](https://github.com/user-attachments/assets/c5577b32-d18d-4436-af82-999c03b8ec20)

### 1. Smoothing Filters

### i) Using Averaging Filter
![image](https://github.com/user-attachments/assets/bf38746c-0325-45a7-adf5-d868727cfde7)

### ii)Using Weighted Averaging Filter
![image](https://github.com/user-attachments/assets/3ade2f6f-dc87-4618-a5b5-b98c5769ad70)


### iii) Using Gaussian Filter
![image](https://github.com/user-attachments/assets/59f780bb-3c35-40ab-bbf6-1e4fb6ff8edd)


### iv) Using median Filter

![image](https://github.com/user-attachments/assets/0b892042-fe2b-4b19-8a56-f43d03e066af)


### 2. Sharpening Filters

### i) Using Laplacian Kernal
![image](https://github.com/user-attachments/assets/3b3f01ad-67c5-499f-ad97-87aaff0e53ed)


### ii) Using Laplacian Operator

![image](https://github.com/user-attachments/assets/7782267e-b3c7-46f4-8b16-67c238f2f8a0)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
