# OpenCV CNN Classification - Skin Cancer Prediction
## **Link to download the dataset:** *https://www.kaggle.com/datasets/eliocordeiropereira/skin-cancer-the-ham10000-dataset* 

## A. Data Assessment Process   
### A1. Original Dataset Summary:
### -> This is **Computer Vision** based project to classify the ***Skin Cancer*** Type using CNN Technique.
### -> Dataset (Skin Cancer MNIST: HAM10000) International Skin Imaging Collaboration (ISIC)

### -> Publicaly available dataset containing 10,015 dermatoscopic images.

### -> A Metadata file containing the demographic information about each lesion.
### -> Lesions are identified using individual methods:

  - ***histo (histopathology)***
  - ***follow_up (follow up examination)***
  - ***consensus (expert consensus)***
  - ***confocal (in-vivo confocal microscopy)***

### -> There are total 7 class labels in which the skin cancer is classified. (nv, mel, bkl, bcc, akiec, vasc, df)
  - ***Melanocytic nevi (nv)***: Melanocytic nevus - the medical term for a mole (benign).
  - ***Melanoma (mel)***: Melanoma - a type of skin cancer involving the melanin cells.
  - ***Dermatofibroma (df)***: Dermatofibroma - common and benign.
  - ***Actinic keratoses (akiec)***: Actinic keratoses and intraepithelial carcinoma (also called "Bowen's disease") - an early form of skin cancer.
  - ***Basal cell carcinoma (bcc)***: Basal cell carcinoma - the most common type of skin cancer.
  - ***Benign keratosis-like lesions (bkl)***: Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses) - common and benign.
  - ***Vascular lesions (vasc)***: Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage) (benign).

### -> Dataset Details:
  - CSV File: 'ham_meta.csv'
  - Total Records: 10,015 
  - Features: 7 ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization']
  - Total Number of Corresponding Images: 10,015

### A2. Features Description:
| S.NO | Feature              | Description                                                                           |
|:----:| :---                 | :---                                                                                  |
| 1.   | lession_id           | ID of the lesion case                                                                 |
| 2.   | image_id             | ID of an image (also the name of the respective JPG file) associated with that case   |
| 3.   | dx                   | Label of that case                                                                    |
| 4.   | dx_type              | Method used for diagnosing that case                                                  |
| 5.   | age                  | Age of the person associated with that case                                           |
| 6.   | sex                  | Sex of the person associated with that case                                           |
| 7.   | localization         | Location of the lesion in the person body                                             |
|      |                      |                                                                                       |

### A3. Data Issues:
#### a. Dirty Data (Low quality):
- Completeness: Missing values in the 'age' feature.
  
- Validity: No duplicate observations.

- Accuracy: No inaccuracy issues.

- Consistency: No inconsistency issues.

#### b. Messy Data (Untidy / Structural):
- No structure related issues.


## B. Data Pre-Processing Results
#### 1. Total Images: 10,015. 
#### 2. Reading metadata from CSV file about patient's skin cancer class label and the corresponding Image ID.
#### 3. All input images are available on path './data' and label-wise ['nv', 'mel', 'bkl', 'df', 'akiec', 'bcc', 'vasc'] images are copied into the directories on path './folders'.
#### 4. Performing Train, Validation, and Test Splits on path './folders' to create datasets under path as './images/train', './images/val', './images/test'.


## C. Conclusions / Insights of Exploratory Data Analysis
- Feature 'age' has 57 missing values for the patients.
- For feature 'dx', class label 'nv' is the dominating category with 6705 entries in the dataset.
- Feature 'dx_type' has 'histo' as the most frequent method of detecting cancer with 5340 records in the dataset.
- Feature 'sex' has 3 categories as ['male', 'female', 'unknown'], where 'male' has occured 5406 times in the dataset.
- More number of cases (2,192) are present for 'back' category of 'localization' feature.
- Distribution for the age of the patients is slightly left skewed (negatively skewed) with a value of -0.1668.

## D. Feature Engineering
- Reduced Features Dataset from the original dataset.
- Class Labels to be Augmented (using a 1000 as threshold value).
- Image Data Generator Object and Augmentation Utility Function.
- DataFrame Entry of Augmented Images.
- Grouping and Resampling the Sub Datasets to Balance the Dataset.
- Adding Image Path to Dataset.
- Saving the Final Balanced DataFrame into CSV file.
- Copying Aug Images to Source (all data gen) folder

## E. Model Building and Hyper Parameter Tuning
- List of models compared: 5 hyper-tuned ANN models have been implemented in this project.


## F. Production Model
- Model: ANN Model 3.
  ```python
    mdl = Sequential()
  
    mdl.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(128,128,3)))
    mdl.add(MaxPooling2D(pool_size=(2,2), padding='valid', strides=2))
    
    mdl.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
    mdl.add(MaxPooling2D(pool_size=(2,2), padding='valid', strides=2))
    
    mdl.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu'))
    mdl.add(MaxPooling2D(pool_size=(2,2), padding='valid', strides=2))
    
    mdl.add(Flatten())
    
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(32, activation='relu'))
    mdl.add(Dense(nclasses, activation='softmax'))
    
    mdl.summary()
    mdl.compile(optimizer='RMSProp', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    hist = mdl.fit(Xtrain, ytrain, batch_size=16, epochs=18, validation_data=(Xval, yval))
  ```

- Results:
  - Train Set Accuracy: 86.46 %
  - Test Set Accuracy: 49.14 %


## G. Gradio App Development  
#### - Production Model: *'sk_best_mdl.keras'*.
#### - Class Labels = 'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'

#### - Test Image: To be uploaded by the user.
#### - Test image pre-processing steps are applied suitably.
#### - Gradio function and interface.
