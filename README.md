## 🩺 Breast Cancer Histopathological Image Classification using ResNet50

In this project, we developed a robust deep learning pipeline to classify breast cancer histopathology images into two classes: **Normal** and **Tumor**.

### 🔧 What we did:

- 📂 **Custom Dataset Class:** Loaded images from separate directories for normal and tumor samples, applying one-hot encoding to labels.  
- 🎨 **Image Preprocessing:** Applied resizing, normalization, and data augmentation via `torchvision.transforms` for better model generalization.  
- 🧮 **Data Splitting:** Divided the dataset into training (70%), validation (15%), and test (15%) sets with random shuffling to ensure unbiased evaluation.  
- 🚀 **Model Architecture:** Used a pre-trained ResNet50 model, modifying the final fully connected layer for binary classification.  
- ⚙️ **Training Setup:**  
  - Loss function: Cross-Entropy Loss  
  - Optimizer: Adam with learning rate 0.001  
  - Seed fixed for reproducibility across runs  
- 📈 **Training & Evaluation:**  
  - Trained the model for multiple epochs, saving weights after each epoch  
  - Monitored training and validation loss/accuracy  
  - Evaluated the final model on the test set  
- 🧾 **Performance:** Achieved ~98% accuracy on the test set, with balanced precision and recall for both classes.  
- 📊 **Visualization:** Plotted loss/accuracy curves and confusion matrix to interpret the model’s behavior and performance.  
- 💾 **Saving Artifacts:** Stored image names and training history in Excel files for traceability.

### 🖼️ Sample Code Snippet:

```python
# Load pre-trained ResNet50 and modify last layer
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop, evaluation, and testing handled via defined functions
```


### 📊 Results Summary:

- **Training set size:** 21,299 images  
- **Validation set size:** 4,564 images  
- **Test set size:** 4,565 images  
- **Test Accuracy:** ~98%  
- Confusion Matrix and Classification Report show excellent performance on both classes.
