# Model Creation
  - You will be able to preprocess the dataset, train a pytorch model of your own, predict on new unseen data using your model.
  

### Note: We Recommend using [Google Colab](https://colab.research.google.com/)  for running the above code.


 ## Dataset 
Some of the dataset we used are listed below:
  - [FaceForensics++](https://github.com/ondyari/FaceForensics)
  - [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
  - [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)
## Preprocessing
  - Load the dataset
  - Split the video into frames
  - crop the face from each frame
  - save the face cropped video
## Model and train
  - It will load the preprocessed video and labels from a csv file.
  - Create a pytorch model using transfer learning with RestNext50 and LSTM.
  - Split the data into train and test data
  - Train the model
  - Test the model
  - save the model in .pt file
 ## Predict 
  - Load the saved pytorch model
  - Predict the output based in trained weights.
  
## Helpers 
  - Code in the Helpers might be helpful for performing some important task  like :
    - Converting Json label file to csv label
    - Copying files from one directory to another
    - Remove Audio altered files from Deepfake Detection Challenge dataset
## Helpful Link
  - Preprocessed data
    - [Celeb-DF Fake processed videos](https://drive.google.com/drive/folders/1SxCb_Wr7N4Wsc-uvjUl0i-6PpwYmwN65?usp=sharing)
    - [Celeb-DF Real processed videos](https://drive.google.com/drive/folders/1g97v9JoD3pCKA2TxHe8ZLRe4buX2siCQ?usp=sharing)
    - [FaceForensics++ Real and fake processed videos](https://drive.google.com/drive/folders/1VIIWRLs6VBXRYKODgeOU7i6votLPPxT0?usp=sharing)
    - [DFDC Fake processed videos](https://drive.google.com/drive/folders/1yz3DBeFJvZ_QzWsyY7EwBNm7fx4MiOfF?usp=sharing)
    - [DFDC Real processed videos](https://drive.google.com/drive/folders/1wN3ZOd0WihthEeH__Lmj_ENhoXJN6U11?usp=sharing)
    
    **Note:** Labels for all the above preprocessed data is under `/label/Gobal_metadata.csv`

  - Trained Models
    - You can just download our [trained models](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing) and run the predict file for prediction.
    
   ***If you need any help regarding the please contact us. We will be happy to help***
