
# Welcome to Face-Analysis-using-DeepFace

A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python using DeepFace for custom videofile and webcam


## Dependencies
- deepface: A deep learning facial analysis library that provides pre-trained models for facial emotion detection. It relies on TensorFlow for the underlying deep learning operations.
- OpenCV: An open-source computer vision library used for image and video processing.
## Installation

The easiest way to install deepface is to download it from PyPI. It's going to install the library itself and its prerequisites as well.

```bash
  $ pip install deepface
```
Alternatively, you can also install deepface from its source code. Source code may have new features not published in pip release yet.
 ```bash
 $ git clone https://github.com/serengil/deepface.git
$ cd deepface
$ pip install -e .
```
Once you installed the library, then you will be able to import it and use its functionalities.

```bash
from deepface import DeepFace
```

## Run
 Press Run or Ctrl+Alt+N