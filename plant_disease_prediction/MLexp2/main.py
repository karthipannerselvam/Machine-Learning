import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label = 0 if subfolder == 'healthy' else 1  # 0 for healthy, 1 for diseased
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels
def extract_features(images):
    feature_list = []
    for img in images:
        img = cv2.resize(img, (128, 128))  # Resize for consistency
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        feature_list.append(features)
    return np.array(feature_list)
dataset_folder = 'dataset'  # Path to your dataset folder
images, labels = load_images_from_folder(dataset_folder)
features = extract_features(images)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
def plot_sample_predictions(images, X_test, y_test, y_pred, num_samples=5):
    plt.figure(figsize=(15, 5))
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)   
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)       
        img = images[idx]  # Use the original image
        img_resized = cv2.resize(img, (128, 128))       
        plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.axis('off')   
    plt.show()
plot_sample_predictions(images, X_test, y_test, y_pred, num_samples=5)
