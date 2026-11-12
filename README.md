\# 🍽️ Omani Cuisine Classification Using Deep Learning



\## Project Overview



Omani Cuisine Classification is a deep learning-based image classification project developed to identify traditional Omani food dishes from images.



The project uses Convolutional Neural Networks (CNN) and TensorFlow to classify food images into 50 different categories, consisting of 25 vegetarian and 25 non-vegetarian Omani dishes.



The system provides a local desktop application where users can upload a food image and receive the predicted dish category.



\## Project Objectives



\- Develop an image classification system for traditional Omani cuisine.

\- Identify 25 vegetarian and 25 non-vegetarian food categories.

\- Prepare and clean the image dataset.

\- Detect and investigate duplicate images.

\- Train and evaluate deep learning models.

\- Develop a user-friendly desktop prediction application.



\## Dataset Description



The dataset consists of 50 Omani cuisine categories.



| Category | Number of Classes |

|---|---:|

| Vegetarian | 25 |

| Non-Vegetarian | 25 |

| Total | 50 |



\### Dataset Preparation



The following steps were performed:



1\. Creation of 50 individual food-class folders.

2\. Collection of Omani cuisine images.

3\. Dataset validation and image verification.

4\. Duplicate image analysis and auditing.

5\. Dataset cleaning and preparation.

6\. Splitting the dataset into training, validation and testing sets.



\### Dataset Split



| Dataset | Number of Images |

|---|---:|

| Training | 670 |

| Validation | 126 |

| Testing | 195 |

| Total | 991 |



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- NumPy

\- Pandas

\- Matplotlib

\- Scikit-learn

\- Pillow

\- Tkinter



\## Project Structure



```text

OMANI\_CUISINE\_PROJECT/

│

├── app.py

├── train\_model.py

├── train\_improved.py

├── evaluate\_models.py

├── split\_dataset.py

├── clean\_dataset.py

├── validate\_dataset.py

│

├── models/

│   ├── best\_model.keras

│   └── class\_names.json

│

├── dataset\_split/

│   ├── train/

│   ├── validation/

│   └── test/

│

├── results/

│   ├── classification reports

│   ├── confusion matrices

│   └── evaluation reports

│

├── scripts/

├── requirements.txt

├── .gitignore

└── README.md

```



\## Model Training and Evaluation



The project includes multiple training and evaluation scripts.



The models are evaluated using classification reports, confusion matrices, category-wise performance analysis and model comparisons.



The evaluation results are stored in the `results` directory.



\## Desktop Application



A Python desktop application has been developed using Tkinter.



\### Application Features



\- Upload a food image from the computer.

\- Process the selected image.

\- Predict the corresponding Omani cuisine category.

\- Display the predicted food class.



The application uses the trained model `best\_model.keras` and the class mapping file `class\_names.json`.



\## Installation and Setup



\### 1. Clone the Repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

```



\### 2. Navigate to the Project Directory



```bash

cd omani-cuisine-classification

```



\### 3. Create a Virtual Environment



```bash

python -m venv venv

```



\### 4. Activate the Environment



Windows:



```bash

venv\\Scripts\\activate

```



\### 5. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 6. Run the Application



```bash

python app.py

```



\*\*Note:\*\* The application requires the trained model and class mapping files to be present inside the `models` directory. The dataset is not required for image prediction.



\## Expected Outcome



The developed system identifies Omani cuisine images across 50 food categories and provides predictions through a local desktop application.



This project demonstrates the application of deep learning and computer vision in food image classification and contributes to the digital documentation and recognition of traditional Omani cuisine.



\## Future Enhancements



\- Improve classification accuracy with larger datasets.

\- Develop a web-based prediction interface.

\- Expand the number of Omani food categories.

\- Integrate multilingual food information.

\- Develop a mobile application.



\## Author



\*\*MANYA HL\*\*



Project: Omani Cuisine Classification Using Deep Learning



\## Acknowledgement



This project is dedicated to the exploration, recognition and digital preservation of traditional Omani cuisine through artificial intelligence and deep learning.



