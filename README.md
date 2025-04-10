# Spectacles_Detection_Deep_Learning-

## Project Scope

The goal of this project is to build a deep learning model that can **detect whether a person is wearing glasses or not** using image data. The images are first **clustered** into two classes—`Glasses (1)` and `No Glasses (0)`—and then different deep learning classification models are trained and evaluated.

This tool could be potentially integrated into real-time **surveillance systems** or **access control** in safety-critical areas.

---

## Dataset

- **Source:** [Kaggle](https://www.kaggle.com/)  
- **Total Images:**  
  - Before augmentation: 5,000  
  - After augmentation: 9,399  
- **Class Distribution:**  
  - Before: 56% Glasses, 43% No Glasses  
  - After: 51% Glasses, 49% No Glasses  
- **Image Types:** Photos of people with varying facial features and different types of glasses.

---

## Tools & Technologies

| Task                          | Tool/Library         |
|------------------------------|----------------------|
| Data handling                | `pandas`             |
| Preprocessing & Augmentation | `tensorflow`, `keras.preprocessing` |
| Clustering & Classification  | `scikit-learn`, `tensorflow.keras` |
| Visualization                | `matplotlib`, `seaborn`, `visualkeras` |

---

## Approach

1. **Preprocessing**
   - Resize images to a uniform shape.
   - Normalize pixel values.
   - Apply data augmentation to balance classes.

2. **Clustering**
   - Images are grouped into two categories: `Glasses` and `No Glasses`.

3. **Modeling**
   - Several classification models were trained:
     - Convolutional Neural Networks (CNNs)
     - Transfer Learning with pretrained models 
   - The best-performing model is selected based on:
     - **Accuracy**
     - **Overfitting behavior (train vs val loss)**

4. **Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, and F1-score
   - Visualizations of model architecture and performance

---

## Results

- Achieved **0.9957% accuracy**
- Best model: `EfficientNetB1` 

---

![image alt](https://github.com/MantenaMonish/Spectacles_Detection_Deep_Learning-/blob/a4e1cf8e9670fab74bc6a9b33128ae717a6132ff/Images/glasses.png)

![image alt](https://github.com/MantenaMonish/Spectacles_Detection_Deep_Learning-/blob/a4e1cf8e9670fab74bc6a9b33128ae717a6132ff/Images/no-glasses.png)
