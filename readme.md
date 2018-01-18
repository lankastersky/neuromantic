# TOC
- [Use Cases](https://github.com/lankastersky/neuromantic#use-cases)
  - [Distributed Neural Networks](https://github.com/lankastersky/neuromantic#distributed-neural-networks)
  - [Go game](https://github.com/lankastersky/neuromantic#go-game)
  - [Gesture Recognition](https://github.com/lankastersky/neuromantic#gesture-recognition)
  - [Image Recognition](https://github.com/lankastersky/neuromantic#image-recognition)
  - [Neural Style Transfer](https://github.com/lankastersky/neuromantic#neural-style-transfer)
  - [Search](https://github.com/lankastersky/neuromantic#search)
  - [Sound recognition](https://github.com/lankastersky/neuromantic#sound-recognition)
- [Tools](https://github.com/lankastersky/neuromantic#tools)
  - [Google Cloud AutoML](https://github.com/lankastersky/neuromantic#google-cloud-automl)
  - [Playgrounds](https://github.com/lankastersky/neuromantic#playgrounds)
- [Books](https://github.com/lankastersky/neuromantic#books)
- [Articles](https://github.com/lankastersky/neuromantic#articles)
- [MOOC](https://github.com/lankastersky/neuromantic#mooc)

# Use Cases

## Distributed Neural Networks
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer by Jeff Dean et al](https://arxiv.org/abs/1701.06538)
  - See article [The VU Sound Corpus by Emiel van Miltenburg et al](http://www.lrec-conf.org/proceedings/lrec2016/pdf/206_Paper.pdf)
- [PathNet: Evolution Channels Gradient Descent in Super Neural Networks by deepmind](https://deepmind.com/research/publications/pathnet-evolution-channels-gradient-descent-super-neural-networks/)

## Go game
- [Mastering the game of Go without human knowledge by David Silver et al, 2017](https://www.gwern.net/docs/rl/2017-silver.pdf)

## Gesture Recognition

### Using wearable sensors (phones, watches etc.)

Articles
- [Physical Human Activity Recognition Using Wearable Sensors by Ferhat Attal et al, 2015](http://www.mdpi.com/1424-8220/15/12/29858)
- [Activity Recognition with Smartphone Sensors by Xing Su et al, 2014](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6838194&tag=1)
- [Run or Walk : Detecting Motion Activity Type with Machine Learning and Core ML](https://towardsdatascience.com/run-or-walk-detecting-user-activity-with-machine-learning-and-core-ml-part-1-9658c0dcdd90)
- Android [DetectedActivity class](https://developers.google.com/android/reference/com/google/android/gms/location/DetectedActivity)
- Android [ActivityRecognitionApi](https://developers.google.com/android/reference/com/google/android/gms/location/ActivityRecognitionApi)

Apps
- [Exercise Tracker: Wear Fitness](https://play.google.com/store/apps/details?id=vimo.co.seven)
- [Google Fit - Fitness Tracking](https://play.google.com/store/apps/details?id=com.google.android.apps.fitness)

Code repositories
- https://github.com/droiddeveloper1/android-wear-gestures-recognition
- https://github.com/drejkim/AndroidWearMotionSensors

## Image Recognition
- [Large-Scale Evolution of Image Classifiers by Esteban Real et al, 2017](https://arxiv.org/pdf/1703.01041.pdf)
- [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
- [Inception in TensorFlow](https://github.com/tensorflow/models/tree/master/research/inception)

## Neural Style Transfer
- [Deep Learning & Art: Neural Style Transfer – An Implementation with Tensorflow in Python](https://www.datasciencecentral.com/profiles/blogs/deep-learning-amp-art-neural-style-transfer-an-implementation)

## Search
- [Neural Architecture Search with Reinforcement Learning by Barret Zoph et al, 2017](https://arxiv.org/abs/1611.01578)

## Sound recognition

Annotated Datasets
- [The VU sound corpus](https://github.com/CrowdTruth/vu-sound-corpus) - based on https://freesound.org/ database
- [AudioSet](https://research.google.com/audioset/) - consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos
- [How do I listen for a sound that matches a pre-recorded sound?](https://arduino.stackexchange.com/questions/8781/how-do-i-listen-for-a-sound-that-matches-a-pre-recorded-sound)
- The Sound Sensor Alert App [sentector](http://sentector.com/)

# Tools

## [Google Cloud AutoML](https://cloud.google.com/automl/)
Pros:
- let users train their own custom machine learning algorithms from scratch, without having to write a single line of code
- uses Transfer Learning (the more data and customers, the better results)
- is fully integrated with other Google Cloud services (Google Cloud Storage to store data, use Cloud ML or Vision API to customize the model etc.)

Cons:
- limited to image recognition (2018-Q1)
- doesn't allow to download a trained model

## Playgrounds
- [Teachable Machine by Google](https://teachablemachine.withgoogle.com/)

# Books

# Articles

# MOOC

