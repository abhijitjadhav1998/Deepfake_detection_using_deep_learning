# Deep fake detection Django Application
### Please reach out to me on [LinkedIn](https://www.linkedin.com/in/abhijitjadhav1998/) for Step by Step installation YouTube video links.
## Requirements:

**Note :** Nvidia GPU is mandatory to run the application.
- CUDA version >= 10.0 for GPU
- GPU Compute Capability > 3.0 


You can find the list of requirements in [requirements.txt](https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning/blob/master/Django%20Application/requirements.txt). Main requirements are listed below:

```
Python >= v3.6
Django >= v3.0
```

## Directory Structure

- ml_app -> Directory containing code in views.py file
- project_settings -> Contains Django settings and files to run in production
- static -> Contains all css, js and json files (for face-api)
- templates -> Template files for HTML

<b>Note:</b> Before running the project make sure you have created directories namely <strong>models, uploaded_images, uploaded_videos</strong> in the project root and that you have proper permissions to access them.

## Prerequisite
1. Copy your trained model to the models folder.
   - You can download our trained models from the [Google Drive](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing) or you can train your models using the steps mentioned in Model Creation directory.

### Step 1 : Clone the repo and Navigate to Django Application

`git clone https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning.git`

### Step 2: Create virtualenv (optional)

`python -m venv venv`

### Step 3: Activate virtualenv (optional)

`venv\Scripts\activate`

### Step 4: Install requirements

`pip install -r requirements.txt`

### Step 5: Copy Models

`Copy your trained model to the models folder i.e Django Application/models/`

- You can download our trained models from [Google Drive](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing)

**Note :** The model name must be in specified format only i.e *model_84_acc_10_frames_final_data.pt*. Make sure that no of frames must be mentioned after certain 3 underscores `_` , in the above example the model is for 10 frames.


### Step 6: Run project

`python manage.py runserver`

## Demo 
### You can watch the [youtube video](https://www.youtube.com/watch?v=_q16aJTXVRE&t=823s) for demo
<p align="center">
  <img src="https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning/blob/master/github_assets/fakegif.gif" />
</p>

   ***If you need any help regarding the please contact us. We will be happy to help***

