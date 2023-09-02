# Landmark Image Recognition App

Welcome to the Landmark Image Recognition App project! In this project, I have developed an application that can predict the most likely locations where user-supplied images were taken. The app uses Convolutional Neural Networks (CNNs) to analyze and classify landmarks in images.

## Project Overview

The goal of this project is to create an app that can identify landmarks in user-uploaded images. Landmark recognition can be valuable for photo-sharing and storage services, as it allows them to associate location data with images, improving the user experience. This is especially useful when images lack location metadata, which can happen if the camera does not have GPS or for privacy reasons.

The app takes any user-supplied image as input and suggests the top k most relevant landmarks from a set of 50 possible landmarks from around the world.

## Features
⚡Multi Label Image Classification  
⚡Cutsom CNN  
⚡Transfer Learning CNN  
⚡PyTorch

## Project Instructions

### Development Environment

I have worked on this project in the Udacity Project Workspace, utilizing the provided GPU to speed up computations. The environment was already set up, including the starter code, which allowed me to focus on building the project.

### Project Development

I followed the project instructions step by step:

1. **Building CNN from Scratch**: I started by working on the `cnn_from_scratch.ipynb` notebook. In this notebook, I built a CNN model from scratch, trained it on landmark images, and evaluated its performance.

2. **Transfer Learning**: The next step was to use transfer learning to improve model performance. I worked on the `transfer_learning.ipynb` notebook, where I fine-tuned a pre-trained model (ResNet-50) for landmark recognition.

3. **App Development**: Finally, I created the Landmark Recognition App in the `app.ipynb` notebook. This app accepts user-supplied images and predicts the most likely landmarks based on the trained model.

### Evaluation

To ensure the project met the requirements, I reviewed the CNN project rubric provided by Udacity and self-evaluated my project against it. I made sure that all criteria in the rubric were met to the best of my abilities.

### Dataset Information

The landmark images used in this project are a subset of the Google Landmarks Dataset v2.

<img src="https://github.com/GauravG-20/Landmark-Image-Classifier/blob/main/dataset.png?raw=true">

## Project Results

### Training Graphs

Here are some training graphs showing the model's performance during training.

#### CNN from Scratch Training Loss and Accuracy
![CNN Training Loss and Accuracy](https://raw.githubusercontent.com/GauravG-20/Landmark-Image-Classifier/main/scratch_training_graph.png)

#### Transfer Learning Training Loss and Accuracy
![Transfer Learning Training Loss and Accuracy](https://raw.githubusercontent.com/GauravG-20/Landmark-Image-Classifier/main/transferred_training_graph.png)

### Deployed App

I have deployed the Landmark Recognition App, and it is accessible via the following link: [Landmark Recognition App](https://landmark-classification.streamlit.app/).

<img src="https://raw.githubusercontent.com/GauravG-20/Landmark-Image-Classifier/main/webapp_landmark_classifier.png">

## Conclusion

This project allowed me to gain valuable experience in building a real-world image recognition application using deep learning techniques. I learned how to develop CNN models from scratch, apply transfer learning to improve performance, and create a user-friendly app for landmark recognition. I also improved my skills in data preprocessing, model training, and evaluation.

If you have any questions or would like to explore the project in more detail, please feel free to reach out. Thank you for reviewing my Landmark Image Recognition App project!

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Get in touch
[![email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gauravgupta.092002@gmail.com)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gaurav-gupta-911463210/)
