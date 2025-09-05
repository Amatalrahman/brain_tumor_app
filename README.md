# Brain Tumor Classification App
## NTI intershipe final project
<img width="960" height="464" alt="Image" src="https://github.com/user-attachments/assets/5de6a6f8-15db-4154-8e5d-998652b90ec7" />


This repository contains a Deep Learning project for classifying brain MRI images into 4 tumor categories. The project covers the full pipeline from data exploration to deployment.

---

## Project Steps

![Image](https://github.com/user-attachments/assets/59acd711-23e1-4a4c-b3c3-5182aa1db722)

### 1. Data Exploration (EDA)
- Analyzed class distribution and image properties.
- Checked for missing or corrupted images.
- Visualized sample images and pixel intensity distributions.

### 2. Data Preprocessing
- Resized all images to a uniform size (224x224).
- Normalized pixel values.
- Split data into **training, validation, and test sets**.
- #### Data Augmentation
  To handle class imbalance, we applied data augmentation on underrepresented classes.
   Techniques included:
   - rotation
   - horizontal/vertical flips
   - zoom
   - width/height shifts
   -  and adding small random noise.
   This increases dataset diversity and helps the model generalize better.

- Use a separate unseen dataset to simulate real-world evaluation.

### 3. Model Training
- Built and trained a mobilenetv2_based model for classification.
- Saved the trained model as `brisc_mobilenetv2_finetuned.keras`.
- Saved class labels in `class_names.json`.

### 4. Deployment
- Created a **simple UI using Streamlit** (`app.py`) for user interaction.
- Hosted the app on **Streamlit Cloud** for global access.
- Uploaded the repository to **GitHub** for version control and collaboration.

---

## Repository Structure
```
  ├── app.py # Streamlit interface
  ├── brisc_mobilenetv2_finetuned.keras # Trained model
  ├── class_names.json # Class mapping
  ├── requirements.txt # Python dependencies
  ├── runtime.txt # Python version for deployment
  ```

---

## Demo
https://github.com/user-attachments/assets/3e2ec3b1-a84f-4347-9433-125dc128e600
# Try the app online:  
![Image](https://github.com/user-attachments/assets/f54d4188-a232-43d9-aef6-5fe3b4503edf)

[Streamlit app link](https://braintumorapp-be6tkpqu4odjcwnfcksrj5.streamlit.app/)


Or run locally:

```bash
git clone <repository-link>
cd <repository-folder>
pip install -r requirements.txt
streamlit run app.py
```
##
<img width="320" height="320" alt="Image" src="https://github.com/user-attachments/assets/2ba6c3b5-3ad2-40a0-883a-dfbc9653837a" />

## Dataset Sources

- Training & Validation dataset: [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)

- Unseen Testing dataset: [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
## Note books
<img width="690" height="362" alt="Image" src="https://github.com/user-attachments/assets/88aee290-92d8-4780-940f-a54673469b7a" />

## All preprocessing, training, and evaluation steps are documented in our notebooks:
- [ EDA & preprocessing approach1 ](https://www.kaggle.com/code/amatalrahmanhasanin/braintumer-classification)
- [preprocessing appraoch 2](https://www.kaggle.com/code/ayamohamednagy12/braintumer-classification)
- [_mobilenetv2 model]()
- [effecient net model]()


# License & Collaboration

This project is licensed under MIT License.
Contributions and collaboration are welcome! Open an issue or pull request if you want to join.


