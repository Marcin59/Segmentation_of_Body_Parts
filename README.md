# Semantic Segmentation Project

This project focuses on semantic segmentation using various deep learning models. The dataset used is the LIP dataset, which contains images and annotations for semantic segmentation tasks.

## Dataset

- **Dataset URL**: [LIP Dataset](https://paperswithcode.com/dataset/lip)

## Setup
Run in project root

1. **Install required packages**:
    ```python
    pip install -r CV/Project3/requirements.txt
    ```

2. **Download the dataset**:
    - **Setup Credential for DVC**: dvc remote modify --local myremote credentialpath 'path/to/project-XXX.json'
    - **Pull Data from DVC**: dvc pull


## Usage
All of these classes and scripts are implemented and shown in [main.ipynb](main.ipynb)
### Data Loading

The `DataLoader` class is used to load images and annotations for training and validation.

```python
train_gen = DataLoader("train")
val_gen = DataLoader("val")
```

### Model Training

The `Predictor` class is used to compile, train, and save models.

```python
predictor = Predictor(train_gen, val_gen)
predictor.add_model("deeplab", get_deeplab_model(input_shape=(*TARGET_SIZE, 3), num_classes=SEMANTIC_CLASSES))
```

### Model Evaluation

Load a trained model and make predictions on test images.

```python
predictor.load_model("deeplab")
test_img = load_img(os.path.join(test_dir, test_files_names[instance_number]), target_size=TARGET_SIZE)
test_img = img_to_array(test_img)
imshow(test_img)
prediction = predictor.predict("deeplab", test_img)
imshow(prediction, variant="annotation")
```

## Streamlit

To run the Streamlit app, follow these steps:

1. **Navigate to the project directory**:
    ```sh
    cd CV/Project3
    ```

2. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

This will start a local server and open the Streamlit app in your default web browser.

## Points

### Problem

- Semantic segmentation - 1 point

### Model

| Task                 | Points |
|----------------------|--------|
| Unet from scratch    | 1      |
| Segnet from scratch  | 1      |
| Deeplab from scratch | 1      |

### Dataset

| Task                                           | Points |
|------------------------------------------------|--------|
| Evaluation on a set with at least 10000 photos | 1      |

### Training

| Task                                  | Points |
|---------------------------------------|--------|
| Testing a few optimizers (at least 3) | 1      |

### Tools

| Task         | Points |
|--------------|--------|
| Tensorboard  | 1      |
| DVC  | 2      |
| Streamlit  | 1      |
