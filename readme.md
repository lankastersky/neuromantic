# TOC
- [Use Cases](#use-cases)
  - [3D recognition](#3d-recognition)
    - [Semantic Segmentation](#semantic-segmentation)
  - [Audio recognition](#audio-recognition)
    - [Speech to Text](#speech-to-text)
  - [Data Agumentation](#data-augmentation)
  - [Design](#design)
  - [Games](#games)
  - [Gesture Recognition](#gesture-recognition)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Image Recognition](#image-recognition)
    - [Face Recognition](#face-recognition)
    - [Food Recognition](#food-recognition)
    - [Image Captioning](#image-captioning)
    - [Person Detection](#person-detection)
    - [Semantic Segmentation](#semantic-segmentation-1)
  - [Interpretability](#interpretability)
  - [Programming and ML](#programming-and-ml)
    - [Predict defects](#predict-defects)
    - [Predict performance](#predict-performance)
    - [Searchinng code](#searching-code)
    - [Writing code](#writing-code)
  - [NLP](#nlp)
    - [Chatbots](#chatbots)
    - [Crossword question answerers](#crossword-question-answerers)
    - [Database queries](#database-queries)
    - [Named entity resolution](#named-entity-resolution)
    - [Reverse dictionaries](#reverse-dictionaries)
    - [Sequence to sequence](#sequence-to-sequence)
    - [Semantic analysis](#semantic-analysis)
    - [Spelling](#spelling)
    - [Summarization](#summarization)
    - [Text classification](#text-classification)
    - [Text to Image](#text-to-image)
    - [Text to Speech](#text-to-speech)
  - [Performance](#performance)
  - [Personality recognition](#personality-recognition)
  - [Search](#search)
  - [Robotics](#robotics)
  - [Transfer Learning](#transfer-learning)
  - [Uber](#uber)
  - [Video recognition](#video-recognition)
    - [Pose Recognition](#pose-recognition)
    - [Object Detection](#object-detection)
    - [Scene Segmentation](#scene-segmentation)
    - [Video Captioning](#video-captioning)
    - [Video Classification](#video-classification)
  - [Visualization](#visualization)
  - [Multiple Modalities](#multiple-modalities)
  - [Open problems](#open-problems)
- [Tools](#tools)
  - [Amazon SageMaker](#amazon-sagemaker)
  - [Apple ARCore](#apple-arcore)
  - [Apple Core ML](#apple-core-ml)
  - [Apple Create ML](#apple-create-ml)
  - [Apple Natural Language Framework](#apple-natural-language-framework)
  - [Firebase ML Kit](#firebase-ml-kit)
  - [Google Cloud AutoML](#google-cloud-automl)
  - [Google Cloud Dataprep](#google-cloud-dataprep)
  - [Google Cloud ML Engine](#google-cloud-ml-engine)
  - [Google Cloud Natural language](#google-cloud-natural-language)
  - [Google Deep Learning Virtual Machine](#google-deep-learning-virtual-machine)
  - [Google Mobile Vision](#google-mobile-vision)
  - [Google Speech API](#google-speech-api)
  - [Google Translation API](#google-translation-api)
  - [Google Video Intelligence](#google-video-intelligence)
  - [Google Vision API](#google-vision-api)
  - [Experiments Frameworks](#experiments-frameworks)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Lobe](#lobe)
  - [Microsoft Azure Bot Service](#microsoft-azure-bot-service)
  - [Microsoft Azure Machine Learning](#microsoft-azure-machine-learning)
  - [Microsoft Cognitive Services](#microsoft-cognitive-services)
  - [Microsoft Cognitive Toolkit](#microsoft-cognitive-toolkit)
  - [Supervisely](#supervisely)
  - [Syn Bot Oscova](#syn-bot-oscova)
  - [TensorFlow](#tensorflow)
  - [Turi Create](#turi-create)
- [Playgrounds](#playgrounds)
  - [Google AIY](#google-aiy)
- [IDEs](#ides)
- [Repositories](#repositories)
- [Models](#models)
  - [Decision trees](#decision-trees)
  - [Distillation](#distillation)
  - [Embedding models](#embedding-models)
  - [Evolutionary Algorithms](#evolutionary-algorithms)
  - [Metrics of dataset quality](#metrics-of-dataset-quality)
  - [Neural Networks](#neural-networks)
    - [Capsule Networks](#capsule-networks)
    - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [Deep Residual Networks](#deep-residual-networks)
    - [Distributed Neural Networks](#distributed-neural-networks)
    - [Feed-Forward Neural Networks](#feed-forward-neural-networks)
    - [Generative Adversarial Networks](#generative-adversarial-networks)
    - [Gated Recurrent Neural Networks](#gated-recurrent-neural-networks)
    - [Long-Short Term Memory Networks](#long-short-term-memory-networks)
    - [Recurrent Neural Networks](#recurrent-neural-networks)
    - [Symmetrically Connected Networks](#symmetrically-connected-networks)
  - [Reinforcement Learning](#reinforcement-learning)
- [Guidelines](#guidelines)
  - [Deep learning](#deep-learning)
- [Interview preparation](#interview-preparation)
- [Books](#books)
  - [NLP](#nlp-1)
  - [Statistics](#statistics)
- [MOOC](#mooc)
  - [Google oriented courses](#google-oriented-courses)
- [Datasets](#datasets)
  - [3D](#3d)
  - [Images](#images)
  - [Videos](#videos)
- [Research groups](#research-groups)
- [Cartoons](#cartoons)

# Use Cases
- [Sample Projects for Data Scientists in Training by V Granville, 2018](https://www.datasciencecentral.com/profiles/blogs/sample-projects-for-data-scientists-in-training)

## 3D Recognition
- https://github.com/IsaacGuan/PointNet-Plane-Detection
  - accuracy around 85% for 100 epochs using TensorFlow
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, 2017](https://github.com/IsaacGuan/PointNet-Plane-Detection)
  - https://github.com/charlesq34/pointnet
  
### Semantic Segmentation
- [SemanticFusion: Dense 3D Semantic Mapping with Convolutional Neural Networks by John McCormac et al, 2016](https://arxiv.org/abs/1609.05130)

## Audio Recognition
- [CNN Architectures for Large-Scale Audio Classification, S. Hershey et al, 2017](https://research.google.com/pubs/pub45611.html)
  - [vggish model used to generate google's AudioSet](https://github.com/tensorflow/models/tree/master/research/audioset)
  - [vggish model adapted for Keras](https://github.com/DTaoo/VGGish)
- [Audio Set: An ontology and human-labeled dataset for audio events, 2017](https://research.google.com/pubs/pub45857.html) 
- [Large-Scale Audio Event Discovery in One Million YouTube Videos, A. Jansen et al, ICASSP 2017](https://research.google.com/pubs/pub45760.html)
- [How do I listen for a sound that matches a pre-recorded sound?](https://arduino.stackexchange.com/questions/8781/how-do-i-listen-for-a-sound-that-matches-a-pre-recorded-sound)
- The Sound Sensor Alert App [sentector](http://sentector.com/)

## Speech to Text
- [Google Duplex: An AI System for Accomplishing Real-World Tasks Over the Phone, 2018](https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html)
- [Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation by Ariel Ephrat et al, 2018](https://arxiv.org/abs/1804.03619)
  - blogpost: [Looking to Listen: Audio-Visual Speech Separation, 2018](https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html)
- https://github.com/facebookresearch/wav2letter
  - [Letter-Based Speech Recognition with Gated ConvNets by Vitaliy Liptchinsky et al, 2017](https://arxiv.org/abs/1712.09444)
  - [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System by Ronan Collobert et al, 2016](https://arxiv.org/abs/1609.03193)

## Data Augmentation
- [Data Augmentation Techniques in CNN using Tensorflow, 2017](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9)
- [How HBO’s Silicon Valley built “Not Hotdog” with mobile TensorFlow, Keras & React Native, 2017](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)
- [My solution for the Galaxy Zoo challenge, 2014](http://benanne.github.io/2014/04/05/galaxy-zoo.html)

## Design
- [Google Design: People + AI Research](https://design.google/library/ai/)
- [PAIR | People+AI Research Initiative](https://ai.google/pair)

## Games
- [Niantic is opening its AR platform so others can make games like Pokémon Go, 2018](https://www.theverge.com/2018/6/28/17511606/niantic-labs-pokemon-go-real-world-platform-ar)
- [Facebook Open Sources ELF OpenGo, 2018](https://research.fb.com/facebook-open-sources-elf-opengo/)
- [Mastering the game of Go without human knowledge by David Silver et al, 2017](https://www.gwern.net/docs/rl/2017-silver.pdf)

## Gesture Recognition

### Using wearable sensors (phones, watches etc.)
- [Physical Human Activity Recognition Using Wearable Sensors by Ferhat Attal et al, 2015](http://www.mdpi.com/1424-8220/15/12/29858)
- [Activity Recognition with Smartphone Sensors by Xing Su et al, 2014](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6838194&tag=1)
- [Motion gesture detection using Tensorflow on Android](http://blog.lemberg.co.uk/motion-gesture-detection-using-tensorflow-android)
- [Run or Walk : Detecting Motion Activity Type with Machine Learning and Core ML](https://towardsdatascience.com/run-or-walk-detecting-user-activity-with-machine-learning-and-core-ml-part-1-9658c0dcdd90)
- Android [DetectedActivity class](https://developers.google.com/android/reference/com/google/android/gms/location/DetectedActivity)
- Android [ActivityRecognitionApi](https://developers.google.com/android/reference/com/google/android/gms/location/ActivityRecognitionApi)

Apps
- [Exercise Tracker: Wear Fitness](https://play.google.com/store/apps/details?id=vimo.co.seven)
- [Google Fit - Fitness Tracking](https://play.google.com/store/apps/details?id=com.google.android.apps.fitness)

Code repositories
- https://github.com/droiddeveloper1/android-wear-gestures-recognition
- https://github.com/drejkim/AndroidWearMotionSensors

## Hyperparameter Tuning
- [Hyperparameter tuning on Google Cloud Platform is now faster and smarter](https://cloud.google.com/blog/big-data/2018/03/hyperparameter-tuning-on-google-cloud-platform-is-now-faster-and-smarter)
- [Hyperparameter tuning in Cloud Machine Learning Engine using Bayesian Optimization, 2017](https://cloud.google.com/blog/big-data/2017/08/hyperparameter-tuning-in-cloud-machine-learning-engine-using-bayesian-optimization)

## Image Recognition
- [MobileNetV2: The Next Generation of On-Device Computer Vision Networks, 2018](https://research.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- [Large-Scale Evolution of Image Classifiers by Esteban Real et al, 2017](https://arxiv.org/pdf/1703.01041.pdf)
- [Rethinking the Inception Architecture for Computer Vision by Christian Szegedy et al, 2015](https://arxiv.org/abs/1512.00567)
  - [Inception in TensorFlow](https://github.com/tensorflow/models/tree/master/research/inception) - 1.4M images and 1000 classes
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications by Andrew G. Howard et al, 2017](https://arxiv.org/abs/1704.04861)
  - [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
  - Similar approach on practice: [How HBO’s Silicon Valley built “Not Hotdog” with mobile TensorFlow, Keras & React Native](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)
- [Deep Residual Learning for Image Recognition by Kaiming He et al, 2015](https://arxiv.org/abs/1512.03385)
- [Going Deeper with Convolutions by C. Szegedy et al, 2014](https://arxiv.org/abs/1409.4842)
- [ImageNet Classification with Deep Convolutional Neural Networks by Alex Krizhevsky et al, 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  - [ImageNet](http://image-net.org/)
  - the model is based on CNN
- [Xception: Deep Learning with Depthwise Separable Convolutions by François Chollet, 2017](https://arxiv.org/abs/1610.02357)
- [ImageNet Classification with Deep Convolutional Neural Networks by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### Face Recognition
- [Вы и Брэд Питт похожи на 99%](https://habr.com/company/rambler-co/blog/417329/)
  - telegram bot telling you which celebrity your face is similar to
  - dlib + resnet + [nmslib](https://github.com/nmslib/nmslib)
- [Умные фотографии ВКонтакте, 2018](https://vk.com/@td-highload-face-recognition) (Smart photos in Vkontakte)
- [FaceNet: A Unified Embedding for Face Recognition and Clustering by Florian Schroff et al, 2015](https://arxiv.org/abs/1503.03832)
  - the model: FaceNet

### Food Recognition
- [NutriNet: A Deep Learning Food and Drink Image Recognition System for Dietary Assessment by Simon Mezgec et al, 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5537777/)
  - uses 520 food and drink items (in Slovene) and the [Google Custom Search API to search for these images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5537777/#app1-nutrients-09-00657title)
- [Food Classification with Deep Learning in Keras / Tensorflow, 2017](http://blog.stratospark.com/deep-learning-applied-food-classification-deep-learning-keras.html)
  - [Creating a Deep Learning iOS App with Keras and Tensorflow](http://blog.stratospark.com/creating-a-deep-learning-ios-app-with-keras-and-tensorflow.html)
  - https://github.com/stratospark/food-101-keras
- [Im2Calories: towards an automated mobile vision food diary by Austin Myers et al, 2015](http://www.cs.ubc.ca/~murphyk/Papers/im2calories_iccv15.pdf)
- [Food 101 Dataset, 2014](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- [Calories nutrition dataset](http://www.fatsecret.com/calories-nutrition/)

### Image Captioning
- [Building an image caption generator with Deep Learning in Tensorflow, 2018](https://medium.freecodecamp.org/building-an-image-caption-generator-with-deep-learning-in-tensorflow-a142722e9b1f)
  - https://github.com/ColeMurray/medium-show-and-tell-caption-generator with docker
  - [Show and Tell: A Neural Image Caption Generator by Oriol Vinyals et al, 2014](https://arxiv.org/abs/1411.4555)
- [Exploring the Limits of Weakly Supervised Pretraining by Dhruv Mahajan et al, 2018](https://arxiv.org/abs/1805.00932)
  - blogpost: [Advancing state-of-the-art image recognition with deep learning on hashtags](https://code.facebook.com/posts/1700437286678763/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/)
- [https://github.com/neural-nuts/Cam2Caption](https://github.com/neural-nuts/Cam2Caption)
  - An Android application which converts camera feed to natural language captions in real time
  - tested: low accuracy, slow (big .pb file is used)
- [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge by Oriol Vinyals et al, 2016](https://arxiv.org/abs/1609.06647)
  - https://github.com/tensorflow/models/tree/master/research/im2txt 
    - training python scripts
    - requires a pretrained Inception v3 checkpoint
  - https://github.com/KranthiGV/Pretrained-Show-and-Tell-model with checkpoints
    - [TensorFlow Deep Learning Machine ezDIY](https://jeffxtang.github.io/deep/learning,/hardware,/gpu,/performance/2017/02/14/deep-learning-machine.html)
  - https://github.com/LitleCarl/ShowAndTell - swift app and training scripts using Keras
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention by Kelvin Xu et al, 2016](https://arxiv.org/abs/1502.03044)

### Performance
- [Quantizing deep convolutional networks for efficient inference: A whitepaper by Raghuraman Krishnamoorthi, 2018](https://arxiv.org/abs/1806.08342)
  - Model sizes can be reduced by a factor of 4 by quantizing weights to 8-bits
  - speedup of 2x-3x for quantized implementations compared to floating point on CPUs
  - [Fixed Point Quantization with tensorflow](https://www.tensorflow.org/performance/quantization)
- [Graph transform](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md) with TensorFlow
  - [Removing training-only nodes](https://www.tensorflow.org/mobile/prepare_models#removing_training_only_nodes) with Tensorflow
- [Optimize for inference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py)  with TensorFlow
  - See example in [TensorFlow for Poets 2: TFMobile](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#3) codelab

### Person Detection
- [Automatic Portrait Segmentation for Image Stylization by Xiaoyong Shen1 et al, 2016](http://www.cse.cuhk.edu.hk/leojia/papers/portrait_eg16.pdf)

### Semantic Segmentation
- [What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)? 2018](https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9)
- [What do we learn from single shot object detectors (SSD, YOLOv3), FPN & Focal loss (RetinaNet)? 2018](https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d)
- [Design choices, lessons learned and trends for object detections?](https://medium.com/@jonathan_hui/design-choices-lessons-learned-and-trends-for-object-detections-4f48b59ec5ff)
- [Semantic Image Segmentation with DeepLab in Tensorflow, 2018](https://research.googleblog.com/2018/03/semantic-image-segmentation-with.html)
  - model DeepLab-v3+ built on top of CNN
  - https://github.com/tensorflow/models/tree/master/research/deeplab
  - has [Checkpoints and frozen inference graphs](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)
  - [Deeplab demo on python](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)
  -  support adopting MobileNetv2 for mobile devices and Xception for server-side deployment
  - evaluates results in terms of mIOU (mean intersection-over-union)
  - use PASCAL VOC 2012 and Cityscapes semantic segmentation benchmarks as an example in the code
  - https://github.com/lankastersky/deeplab_background_segmentation (not working android app)
- [Rethinking Atrous Convolution for Semantic Image Segmentation by Liang-Chieh Chen et al, 2017](https://arxiv.org/pdf/1706.05587.pdf)
MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features by Liang-Chieh Chen et al, 2017](https://arxiv.org/abs/1712.04837)
  - present a model, called MaskLab, which produces three outputs: box detection, semantic segmentation, and direction prediction	
  - built on top of the Faster-RCNN object detector
  - evaluated on the COCO instance segmentation benchmark and shows comparable performance with other state-of-art models
- [Mask R-CNN by Kaiming He et al, 2017](https://arxiv.org/abs/1703.06870)
  - https://github.com/facebookresearch/Detectron
  - see links to articles at the end of the page
  - extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition
  - simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps
  - easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework
  - outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners
  - uses [the area under the precision recall curve (AP)](https://github.com/cocodataset/cocoapi/issues/56) metrics
- [A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN, 2017](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)

## Interpretability
- [The Building Blocks of Interpretability, 2018](https://distill.pub/2018/building-blocks/)
 - GoogleNet for image classification is used as an example
- [Attributing a deep network’s prediction to its input features by MUKUND SUNDARARAJAN, 2017](http://www.unofficialgoogledatascience.com/2017/03/attributing-deep-networks-prediction-to.html)
  - Integrated Gradients method
- [A unified approach to interpreting model predictions by Scott M Lundberg et al, 2017](https://nips.cc/Conferences/2017/Schedule?showEvent=10008)
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier by Marco Tulio Ribeiro et al, 2016](https://arxiv.org/abs/1602.04938)
  - [Lime Framework: Explaining the predictions of any machine learning classifier](https://github.com/marcotcr/lime)
- [Monotonic Calibrated Interpolated Look-Up Tables by Maya Gupta et al, 2016](http://jmlr.org/papers/v17/15-243.html)
- see [Decision trees](#decision-trees)
- see [Distillation](#distillation)

## Programming and ML
- [TREE-TO-TREE NEURAL NETWORKS FOR PROGRAM TRANSLATION by Xinyun Chen et al, 2018](https://openreview.net/pdf?id=rkxY-sl0W)
- [Software	is	eating	the	world,	but	ML	is	going	to	eat	software by Erik Meijer, Facebook, 2018](https://pps2018.soic.indiana.edu/files/2017/12/PPS2018Meijer.pdf)
- [A Survey of Machine Learning for Big Code and Naturalness by Miltiadis Allamanis et al, 2017](https://arxiv.org/abs/1709.06182)

### Predict defects
- [To type or not to type: quantifying detectable bugs in JavaScript by Gao et al, 2017](https://blog.acolyer.org/2017/09/19/to-type-or-not-to-type-quantifying-detectable-bugs-in-javascript/)
- [Predicting Defects for Eclipse by T Zimmermann at al, 2007](http://thomas-zimmermann.com/publications/files/zimmermann-promise-2007.pdf)
  - used code complexity metrics as features and logistic regression for classification (if file/module has defects) and linear regression for ranking (how many defects)
- [Predicting Component Failures at Design Time by Adrian Schroter et al, 2006](http://thomas-zimmermann.com/publications/files/schroeter-isese-2006.pdf)
  - showed that design data such as import relationships can predict failures
  - used the number of failures in a component as dependent variable and the imported resources used from this component as input features
- [Mining Version Histories to Guide Software Changes by T Zimmermann at al, 2004](http://www.ics.uci.edu/~andre/ics228s2006/zimmermanweissgerberdiehlzeller.pdf)
  - used [apriory algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm) to predict likely changes in files/modules

### Predict performance
- https://codescene.io
- [3 ways AI will change project management for the better, 2017](https://www.atlassian.com/blog/software-teams/3-ways-ai-will-change-project-management-better)
- [A deep learning model for estimating story points by Morakot Choetkiertikul et al, 2016](https://arxiv.org/pdf/1609.00489.pdf)
  - estimating story points based on long short-term memory and recurrent highway network

### Searching code
- [Deep code search by Xiaodong Gu1 et al, 2018](https://guxd.github.io/papers/deepcs.pdf)
  - [blog post](https://blog.acolyer.org/2018/06/26/deep-code-search/)
- [How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning, 2018](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)
  - https://github.com/hamelsmu/code_search
  - you can use similar techniques to search video, audio, and other objects

### Writing code
- https://github.com/capergroup/bayou
  - [NEURAL SKETCH LEARNING FOR CONDITIONAL PROGRAM GENERATION by Vijayaraghavan Murali et al, 2018](https://arxiv.org/pdf/1703.05698.pdf)
- [Program Synthesis in 2017-18](https://alexpolozov.com/blog/program-synthesis-2018/)
- https://www.deepcode.ai

## NLP
- [Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/) - OpenAI
- [SentEval: An Evaluation Toolkit for Universal Sentence Representations by A. Conneau et al, 2018](https://arxiv.org/abs/1803.05449)
  - https://github.com/facebookresearch/SentEval
  - the benchmarks may not be appropriate for domain-specific problems
- [Text Embedding Models Contain Bias. Here's Why That Matters, 2018](https://developers.googleblog.com/2018/04/text-embedding-models-contain-bias.html)
- [How to Clean Text for Machine Learning with Python](https://machinelearningmastery.com/clean-text-machine-learning-python/)

### Chatbots
- https://ipavlov.ai/ - open-source conversational AI framework built on TensorFlow and Keras (En, Ru)
  - https://github.com/deepmipt/DeepPavlov
- [Behind the Chat: How E-commerce Robot Assistant AliMe Works, 2018](https://medium.com/mlreview/behind-the-chat-how-e-commerce-bot-alime-works-1b352391172a)
- [How I Used Deep Learning To Train A Chatbot To Talk Like Me (Sorta), 2017](https://adeshpande3.github.io/How-I-Used-Deep-Learning-to-Train-a-Chatbot-to-Talk-Like-Me)
  - Short-Text Conversations generative model based on Tensorflow’s embedding_rnn_seq2seq() with custom dataset. Deployed as a Facebook chatbot using heroku (hosting)+express(frontend)+flask(backend)
- [Deep Learning for Chatbots, Part 1 – Introduction, 2016](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)
- [Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow, 2016](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/)
- https://github.com/gunthercox/ChatterBot
  - Retrieval-based model based on [naive Bayesian classification and search algorithms](http://chatterbot.readthedocs.io/en/stable/faq.html#what-kinds-of-machine-learning-does-chatterbot-use) 
  - see [Sequence to sequence](#sequence-to-sequence)
- [A Persona-Based Neural Conversation Model by Jiwei Li et al, 2016](https://arxiv.org/abs/1603.06155)
- Smart reply
  - [Smart Reply: Automated Response Suggestion for Emai by Anjuli Kannan et al, 2016](https://arxiv.org/abs/1606.04870)
  - [Computer, respond to this email, 2015](https://research.googleblog.com/2015/11/computer-respond-to-this-email.html)
- Chatbot projects: https://github.com/fendouai/Awesome-Chatbot
- see [Chatbot platforms](#chatbot-platforms)

### Crossword question answerers
- see [Reverse dictionaries](#reverse-dictionaries)

### Database queries
- [LEARNING A NATURAL LANGUAGE INTERFACE WITH NEURAL PROGRAMMER by Arvind Neelakantan et al, 2017](https://arxiv.org/pdf/1611.08945.pdf)
  - weakly supervised, end-to-end neural network model mapping natural language queries to logical forms or programs that
provide the desired response when executed on the database

### Named entity resolution
Also known as deduplication and record linkage (but [not entity recognition](https://stackoverflow.com/questions/8589005/difference-between-named-entity-recognition-and-resolution) which is picking up the names and classifying them in running text)
- [Collective Entity Resolution in Familial Networks by Pigi Kouki et al, 2017](https://linqspub.soe.ucsc.edu/basilic/web/Publications/2017/kouki:icdm17/kouki-icdm17.pdf)
  - combines machine learning (although not NNs) with collective inference
- [Entity Resolution Using Convolutional Neural Network by Ram DeepakGottapu et al, 2016](https://www.sciencedirect.com/science/article/pii/S1877050916324796)
- [Adaptive Blocking: Learning to Scale Up Record Linkage by Mikhail Bilenko et al, 2006](http://www.cs.utexas.edu/~ml/papers/blocking-icdm-06.pdf)
  - extremely high recall but low precision
- https://stats.stackexchange.com/questions/136755/popular-named-entity-resolution-software

### Reverse dictionaries
Other name is concept finders
Return the name of a concept given a definition or description:
- [Learning to Understand Phrases by Embedding the Dictionary by Felix Hill et al, 2016](http://www.aclweb.org/anthology/Q16-1002)
  - used models: Bag-of-Words NLMs and LSTM
- comparing definitions in a database to the input query, and returning the word whose definitionis ‘closest’ to that query
- see RNNs (with LSTMs)
- see bag-of-word

### Sequence to sequence
- [Smart Compose: Using Neural Networks to Help Write Emails, 2018](https://ai.googleblog.com/2018/05/smart-compose-using-neural-networks-to.html)
- [Introducing Semantic Experiences with Talk to Books and Semantris by Rey Kurzweil et al, 2018](https://research.googleblog.com/2018/04/introducing-semantic-experiences-with.html)
  - [Universal Sentence Encoder Daniel Cer et al, 2018](https://arxiv.org/abs/1803.11175)
  - [Pretrained semantic TensorFlow module](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1)
- [Keras LSTM tutorial – How to easily build a powerful deep learning language model by Andy, 2018](http://adventuresinmachinelearning.com/keras-lstm-tutorial/)
- [Generating High-Quality and Informative Conversation Responses with Sequence-to-Sequence Models by Louis Shao et al, 2017](https://research.google.com/pubs/pub45936.html)
  - trained on a combined data set of over 2.3B conversation messages mined from the web
  - The model: LSTM on tensorflow
- [Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features by Matteo Pagliardini et al, 2017](https://arxiv.org/pdf/1703.02507.pdf)
  - the model: Sent2Vec based on vec2vec
- [Skip-Thought Vectors by Ryan Kiros et al, 2015](https://arxiv.org/abs/1506.06726)
  - based on RNN encoder-decoder models
- [Sequence to Sequence Learning with Neural Networks by Ilya Sutskever et al, 2014](https://arxiv.org/abs/1409.3215)
  - the model: seq2seq based on LSTM
- [Distributed Representations of Sentences and Documents by Quoc V. Le, Mikolov, 2014](https://arxiv.org/abs/1405.4053)
  - [gensim's doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)
  - [python example to train doc2vec model (with or without pre-trained word embeddings)](https://github.com/jhlau/doc2vec)
- [Distributed Representations of Words and Phrases and their Compositionality by Tomas Mikolov et al, 2013](https://arxiv.org/abs/1310.4546)
  - word2vec based on Mikolov's Skip-gram model
- [Learning Continuous Phrase Representations and Syntactic Parsing with Recursive Neural Networks by Richard Socher et al, 2010](http://ai.stanford.edu/~ang/papers/nipsdlufl10-LearningContinuousPhraseRepresentations.pdf)
  - based on context-sensitive recursive neural networks (CRNN)
- see [Reverse dictionaries](#reverse-dictionaries)
- [How to calculate the sentence similarity using word2vec model](https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt)
  - Doc2Vec
  - Average w2v vectors
  - Weighted average w2v vectors (e.g. tf-idf)
  - RNN-based embeddings (e.g. deep LSTM networks)
  - [Document Similarity With Word Movers Distance](http://jxieeducation.com/2016-06-13/Document-Similarity-With-Word-Movers-Distance/)
    - [From Word Embeddings To Document Distances by Matt J. Kusner et al, 2015](http://proceedings.mlr.press/v37/kusnerb15.pdf)
  - [A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS by Sanjeev Arora et al, 2017](https://openreview.net/pdf?id=SyK00v5xx)
    - uses smooth inverse frequency
    - computing the weighted average of word vectors in the sentence and then remove the projections of the average vectors on their first principal component
    - [example](http://sujitpal.blogspot.com/2017/05/evaluating-simple-but-tough-to-beat.html)
    - https://github.com/peter3125/sentence2vec - requires writing the get_word_frequency() method which can be easily accomplished by using Python's Counter() and returning a dict with keys: unique words w, values: #w/#total doc len

### Semantic analysis
- [Advances in Semantic Textual Similarity, 2018](https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html)
  - [Learning Semantic Textual Similarity from Conversations by Yinfei Yang et al, 2018](https://arxiv.org/abs/1804.07754)
  - [Universal Sentence Encoder by Daniel Cer et al, 2018](https://arxiv.org/abs/1803.11175)
- [Semantic Textual Similarity Wiki, 2017](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page)
- [A Deeper Look into Sarcastic Tweets Using Deep Convolutional Neural Networks by Soujanya Poria et al, 2017](https://arxiv.org/abs/1610.08815)
  - Blog post: [Detecting Sarcasm with Deep Convolutional Neural Networks](https://medium.com/dair-ai/detecting-sarcasm-with-deep-convolutional-neural-networks-4a0657f79e80)
- [Twitter Sentiment Analysis Using Combined LSTM-CNN Models by SOSAVPM, 2018](http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/)
  - https://github.com/pmsosa/CS291K
  - used pre-trained embeddings with LSTM-CNN model with dropouts
  - 75.2% accuracy for binary classification (positive-negative tweet)
- [doc2vec example, 2015](http://linanqiu.github.io/2015/10/07/word2vec-sentiment/)

### Spelling
- [How to Write a Spelling Corrector](http://norvig.com/spell-correct.html)

### Summarization
- [How To Create Data Products That Are Magical Using Sequence-to-Sequence Models](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8)
  - A tutorial on how to summarize text and generate features from Github Issues using deep learning with Keras and TensorFlow
  - https://github.com/hamelsmu/Seq2Seq_Tutorial
- [Generating Wikipedia by Summarizing Long Sequences by Peter J. Liu et al, 2018](https://arxiv.org/abs/1801.10198)

### Text classification
- [Universal Language Model Fine-tuning for Text Classification by Jeremy Howard et al, 2018](https://arxiv.org/pdf/1801.06146.pdf)
  - outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets.
  - with only 100 labeled examples, it matches the performance of training from scratch on 100× more data
  - http://nlp.fast.ai/ulmfit

### Text to Image
- [ChatPainter: Improving text-to-image generation by using dialogue](https://www.microsoft.com/en-us/research/blog/chatpainter-improving-text-image-generation-using-dialogue/)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks by Tao Xu et al, 2017](https://arxiv.org/abs/1711.10485)
  - [Microsoft researchers build a bot that draws what you tell it to, 2018](https://blogs.microsoft.com/ai/drawing-ai/)

### Text to Speech
- [Efficient Neural Audio Synthesis by Nal Kalchbrenner et al, 2018](https://arxiv.org/abs/1802.08435)
  - [Pytorch implementation of Deepmind's WaveRNN model](https://github.com/fatchord/WaveRNN) on github
- [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention by Hideyuki Tachibana et al, 2017](https://arxiv.org/abs/1710.08969)
  - https://github.com/Kyubyong/
- https://github.com/r9y9/ (Ryuichi Yamamoto)
- https://github.com/keithito/
- [WaveNet: A Generative Model for Raw Audio, 2016](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

## Personality recognition
- Mining Facebook Data for Predictive Personality Modeling (Dejan Markovikj,Sonja Gievska, Michal Kosinski, David Stillwell)
- Personality Traits Recognition on Social Network — Facebook (Firoj Alam, Evgeny A. Stepanov, Giuseppe Riccardi)
- The Relationship Between Dimensions of Love, Personality, and Relationship Length (Gorkan Ahmetoglu, Viren Swami, Tomas Chamorro-Premuzic)

## Robotics
- [Grasp2Vec: Learning Object Representations from Self-Supervised Grasping](https://sites.google.com/site/grasp2vec/)
  - Achieved a success rate of 80 percent on objects seen during data collection and 59% on novel objects the robot hasn’t encountered before

## Search
- [Can word2vec be used for search?](https://www.reddit.com/r/MachineLearning/comments/4mw927/can_word2vec_be_used_for_search/)
  - alternative search queries can be built using approximate nearest neighbors in embedding vectors space of terms (using https://github.com/spotify/annoy e.g.)
  - [Improving Document Ranking with Dual Word Embeddings by Eric Nalisnick et al, 2016](https://www.microsoft.com/en-us/research/publication/improving-document-ranking-with-dual-word-embeddings/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2Fdefault.aspx%3Fid%3D260867)

## Transfer Learning
- [Life-Long Disentangled Representation Learning with Cross-Domain Latent Homologies by Alessandro Achille et al, 2018](https://arxiv.org/abs/1808.06508)
  - possible solution of catastrophic forgetting
- [Deep Learning & Art: Neural Style Transfer – An Implementation with Tensorflow in Python, 2018](https://www.datasciencecentral.com/profiles/blogs/deep-learning-amp-art-neural-style-transfer-an-implementation)
- [Image Classification using Flowers dataset on Cloud ML Enginge, 2018](https://cloud.google.com/ml-engine/docs/flowers-tutorial)
- [Android & TensorFlow: Artistic Style Transfer, 2018](https://codelabs.developers.google.com/codelabs/tensorflow-style-transfer-android/index.html#0) codelab
- [The TensorFlow Poet tutorial](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) shows how to retrain a tensorflow graph to classify images of flowers.

## Uber
- [Engineering More Reliable Transportation with Machine Learning and AI at Uber, 2017](https://eng.uber.com/machine-learning/)

## Video recognition

### Pose recognition
- [Everybody Dance Now by CAROLINE CHAN et al, 2018](https://arxiv.org/pdf/1808.07371.pdf)
  - [Motion retargeting video](https://www.youtube.com/watch?v=PCBTZh41Ris&feature=youtu.be)
- [Real-time Human Pose Estimation in the Browser with TensorFlow.js, 2018](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) (Medium post)
  - [Pose Detection in the Browser: PoseNet Model](https://github.com/tensorflow/tfjs-models/tree/master/posenet) (github)
- [Enabling full body AR with Mask R-CNN2Go by Fei Yang et al, 2018](https://research.fb.com/enabling-full-body-ar-with-mask-r-cnn2go/)
- [PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model by George Papandreou et al, 2018](https://arxiv.org/abs/1803.08225)
- [Towards Accurate Multi-person Pose Estimation in the Wild by George Papandreou et al, 2017](https://arxiv.org/abs/1701.01779)
- [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields by Zhe Cao et al, 2017](https://arxiv.org/abs/1611.08050)
  - https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

### Object detection
Here are video-specific methods. See also [Semantic Segmentation](#semantic-segmentation).

- [Training and serving a realtime mobile object detector in 30 minutes with Cloud TPUs, 2018](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193)
  - includes checkpoints
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
  - [YOLOv3: An Incremental Improvement by Joseph Redmon, Ali Farhadi, 2018](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [Mobile Real-time Video Segmentation, 2018](https://research.googleblog.com/2018/03/mobile-real-time-video-segmentation.html)
  - integrated into Youtube stories
- [The Instant Motion Tracking Behind Motion Stills AR, 2018](https://ai.googleblog.com/2018/02/the-instant-motion-tracking-behind.html)
- [Behind the Motion Photos Technology in Pixel 2, 2018](https://ai.googleblog.com/2018/03/behind-motion-photos-technology-in.html)
- [Supercharge your Computer Vision models with the TensorFlow Object Detection API, 2017](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html)
  - https://github.com/tensorflow/models/tree/master/research/object_detection
- [Ridiculously Fast Shot Boundary Detection with Fully Convolutional Neural Networks by Michael Gygli, 2017](https://arxiv.org/abs/1705.08214)
- [Video Shot Boundary Detection based on Color Histogram by J. Mas and G. Fernandez, 2003](http://www-nlpir.nist.gov/projects/tvpubs/tvpapers03/ramonlull.paper.pdf)

### Scene Segmentation
Detects when one video (shot/scene/chapter) ends and another begins
- [Recurrent Switching Linear Dynamical Systems by Scott W. Linderman et al, 2016](https://arxiv.org/abs/1610.08466)
- [Video Scene Segmentation Using Markov Chain Monte Carlo by Yun Zha et al, 2006](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1658031)
- [Automatic Video Scene Segmentation based on Spatial-Temporal Clues and Rhythm by Walid Mahdi et al, 2000](https://arxiv.org/abs/1412.4470)

### Video Captioning
- [Temporal Relational Reasoning in Videos by Bolei Zhou et al, 2018](https://arxiv.org/abs/1711.08496) - Recognizing and forecasting activities by a few frames
  - https://www.youtube.com/watch?time_continue=54&v=JBwSk6nJOyM
  - https://github.com/metalbubble/TRN-pytorch
  - http://relation.csail.mit.edu/
  - http://news.mit.edu/2018/machine-learning-video-activity-recognition-0914
  
- [DeepStory: Video Story QA by Deep Embedded Memory Networks by Kyung-Min Kim et al, 2017](https://arxiv.org/abs/1707.00836)
  - https://github.com/Kyung-Min/Deep-Embedded-Memory-Networks
- [Video Understanding: From Video Classification to Captioning by Jiajun Sun et al, 2017](http://cs231n.stanford.edu/reports/2017/pdfs/709.pdf)
- [Unsupervised Learning from Narrated Instruction Videos by Jean-Baptiste Alayrac et al, 2015](https://arxiv.org/abs/1506.09215)

### Video Classification
- [Learnable pooling with Context Gating for video classification by Antoine Miech et al, 2018](https://arxiv.org/abs/1706.06905)
  - Rank #1 at [Google Cloud & YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m)
  - Slow for inference/training
  - NOT a sequential problem
  - Needs lots of data for training
  - not clear about very long videos
- [The Monkeytyping Solution to the YouTube-8M Video Understanding Challenge, 2017](https://static.googleusercontent.com/media/research.google.com/en//youtube8m/workshop2017/c04.pdf)
  - Rank #2 at [Google Cloud & YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m)
- [Hierarchical Deep Recurrent Architecture for Video Understanding by Luming Tang et al, 2017](https://arxiv.org/abs/1707.03296)
- [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? by Kensho Hara et al, 2017](https://arxiv.org/abs/1711.09577)
  - https://github.com/kenshohara/video-classification-3d-cnn-pytorch
    - trained on the Kinetics dataset from scratch using only RGB input
    - pretrained ResNeXt-101 achieved 94.5% and 70.2% on UCF-101 and HMDB-51
- [Appearance-and-Relation Networks for Video Classification by Limin Wang et al, 2017](https://arxiv.org/abs/1711.09125)
  - https://github.com/wanglimin/ARTNet
    - trained on the Kinetics dataset from scratch using only RGB input
    - 70.9% and 94.3% on HMDB51	UCF101
- [Five video classification methods implemented in Keras and TensorFlow by Matt Harvey, 2017](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5)
  - https://github.com/harvitronix/five-video-classification-methods
- [Video Understanding: From Video Classification to Captioning by Jiajun Sun et al, 2017](http://cs231n.stanford.edu/reports/2017/pdfs/709.pdf)
- [Video Classification using Two Stream CNNs, 2016](https://github.com/wadhwasahil/Video-Classification-2-Stream-CNN) code based on articles below
  - Two-Stream Convolutional Networks for Action Recognition in Videos
  - Fusing Multi-Stream Deep Networks for Video Classification
  - Modeling Spatial-Temporal Clues in a Hybrid Deep Learning Framework for Video Classification
  - Towards Good Practices for Very Deep Two-Stream ConvNets
- [Beyond Short Snippets: Deep Networks for Video Classification by Joe Yue-Hei Ng et al, 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ng_Beyond_Short_Snippets_2015_CVPR_paper.pdf)
  - In order to learn a global description of the video while maintaining a low computational footprint, we propose processing only one frame per second
- [Large-scale Video Classification with Convolutional Neural Networks by Andrej Karpathy et al, 2014](https://cs.stanford.edu/people/karpathy/deepvideo/)
  - 63.3% on UCF-101

## Visualization
- [Google Brain: Big Picture Group](https://research.google.com/bigpicture/)
- [Deeplearn.js](https://github.com/tensorflow/tfjs-core) - open source hardware-accelerated machine intelligence library for the web
- [Facets](https://pair-code.github.io/facets/) - open source visualizations for machine learning datasets
  - [Facets: An Open Source Visualization Tool for Machine Learning Training Data, 2017](https://ai.googleblog.com/2017/07/facets-open-source-visualization-tool.html)
  - https://github.com/PAIR-code/facets
- [Embedding Projector](https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/projector) - an open source, visualization tool for high-dimensional data

## Multiple Modalities
- [Multimodal Classification for Analysing Social Media by Chi Thang Duong et al, 2017](https://arxiv.org/abs/1708.02099)
  - Blog post: [Detecting Emotions with CNN Fusion Models](https://medium.com/dair-ai/detecting-emotions-with-cnn-fusion-models-b066944969c8)
  - https://emoclassifier.github.io/

## Open problems
- Recycled goods (not solved, no dataset)
  - [Recycling symbols explained](https://www.recyclenow.com/recycling-knowledge/packaging-symbols-explained)
  - similar to traffic signs recognition
- Safety symbols on cardboard boxes (not solved, no dataset)  

# Tools
- [50+ Useful Machine Learning & Prediction APIs, 2018](https://www.kdnuggets.com/2018/05/50-useful-machine-learning-prediction-apis-2018-edition.html)
  - Face and Image Recognition
  - Text Analysis, NLP, Sentiment Analysis
  - Language Translation
  - Machine Learning and prediction
- [Command-line tricks data scientists](https://www.kdnuggets.com/2018/06/command-line-tricks-data-scientists.html)
- [Deep Video Analytics](https://www.deepvideoanalytics.com/)
  - Data-centric platform for Computer Vision
   - https://github.com/akshayubhat/deepvideoanalytics

## [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
- Distributed Training: You can’t choose the number of workers and parameter servers independently
- Job Startup Latency: Up to 5 minutes single node
- Hyper Parameters Tuning: In-Preview, and only supports the built-in algorithms
- Batch Prediction: Not supported
- GPU readiness: Bring your own docker image with CUDA installed
- Auto-scale Online Serving: You need to specify the number of nodes
- Training Job Monitoring: No monitoring

## [Apple ARCore](https://developers.google.com/ar/develop/)
- https://github.com/google-ar/arcore-android-sdk
- https://github.com/google-ar/sceneform-android-sdk
- [Cloud Anchors android codelab](https://codelabs.developers.google.com/codelabs/arcore-cloud-anchors/#0)
- https://github.com/google-ar/arcore-ios-sdk

## [Apple Core ML](https://developer.apple.com/documentation/coreml)
iOS framework from Apple to integrate machine learning models into your app.

## [Apple Create ML](https://developer.apple.com/documentation/create_ml)
Apple framework used with familiar tools like Swift and macOS playgrounds to create and train custom machine learning models on your Mac.
- [Introducing Create ML](https://developer.apple.com/videos/play/wwdc2018/703/) on wwdc2018

## [Apple Natural Language Framework](https://developer.apple.com/documentation/naturallanguage)
- [Introducing Natural Language Framework](https://developer.apple.com/videos/play/wwdc2018/713/) on wwdc2018

## [Firebase ML Kit](https://firebase.google.com/docs/ml-kit/)
- [ML Kit: Machine Learning SDK for mobile developers (Google I/O '18)](https://www.youtube.com/watch?v=Z-dqGRSsaBs)
- Uses Google Cloud APIs under the hood
- Uses custom TensorFlow Lite models
- Can compress TensorFlow to TensorFlow Lite models
- Runs on a device (fast, inaccurate) or on a cloud
- Examples and codelabs
  - [Recognize text in images with ML Kit for Firebase: iOS](https://codelabs.developers.google.com/codelabs/mlkit-ios/#0)
  - [Recognize text in images with ML Kit for Firebase: Android](https://codelabs.developers.google.com/codelabs/mlkit-android/#0)
  - [Identify objects in images using custom machine learning models with ML Kit for Firebase: Android](https://codelabs.developers.google.com/codelabs/mlkit-android-custom-model/#0)
  - [ML Vision iOS example](https://github.com/firebase/quickstart-ios/tree/master/mlvision) 
  - [Custom model iOS example](https://github.com/firebase/quickstart-ios/tree/master/mlmodelinterpreter)
  - [android examples](https://github.com/firebase/quickstart-android/tree/master/mlkit)

## [Google Cloud AutoML](https://cloud.google.com/automl/)

Pros:
- let users train their own custom machine learning algorithms from scratch, without having to write a single line of code
- uses Transfer Learning (the more data and customers, the better results)
- is fully integrated with other Google Cloud services (Google Cloud Storage to store data, use Cloud ML or Vision API to customize the model etc.)

Cons:
- limited to image recognition (2018-Q1)
- doesn't allow to download a trained model

## [Google Cloud Dataprep](https://cloud.google.com/dataprep/)
Intelligent data service for visually exploring, cleaning, and preparing structured and unstructured data for analysis. Cloud Dataprep is serverless and works at any scale. Easy data preparation with clicks and no code.

## [Google Cloud ML Engine](https://cloud.google.com/ml-engine/)
- [Samples & Tutorials](https://cloud.google.com/ml-engine/docs/tutorials)
- [Samples for usage](https://github.com/GoogleCloudPlatform/cloudml-samples)
- Distributed Training: Specify number of nodes, types, (workers/PS), associated accelerators, and sizes
- Job Startup Latency: 90 seconds for single node
- Hyper Parameters Tuning: Grid Search, Random Search, and Bayesian Optimisation
- Batch Prediction: You can submit a batch prediction job for high throughputs
- GPU readiness: Out-of-the box, either via scale-tier, or config file
- Auto-scale Online Serving: Scaled up to your specified maximum number of nodes, down to 0 nodes if no requests for 5 minutes
- Training Job Monitoring: Full monitoring to the cluster nodes (CPU, Memory, etc.)
- Automation of ML: AutoML - Vision, NLP, Speech, etc.
- Specialised Hardware: Tensor Processing Units (TPUs)
- SQL-supported ML: [BQML](https://cloud.google.com/blog/big-data/2018/07/bridging-the-gap-between-data-and-insights)

# [Google Cloud Natural language](https://cloud.google.com/natural-language/)
- entiry recognition: extract information about people, places, events, and much more mentioned in text documents, news articles, or blog posts
- sentiment analysis: understand the overall sentiment expressed in a block of text
- multilingual support
- syntax analysis: extract tokens and sentences, identify parts of speech (PoS) and create dependency parse trees for each sentence

## [Google Deep Learning Virtual Machine](https://cloud.google.com/deep-learning-vm/docs/)
- VMs with CPU and GPU 

## [Google Mobile Vision](https://developers.google.com/vision/)
- Detect Faces (finds facial landmarks such as the eyes, nose, and mouth; doesn't identifies a person)
- Scan barcodes
- Recognize Text

## [Google Speech API](https://cloud.google.com/speech-to-text/)
- speech recognition
- word hints: Can provide context hints for improved accuracy.  Especially useful for device and app use cases.
- noise robustness: No need for signal processing or noise cancellation before calling API; can handle noisy audio from a variety of environments
- realtime results: can stream text results, returning partial recognition results as they become available.  Can also be run on buffered or archived audio files.  
- over 80 languages
- can also filter inappropriate content in text results

## [Google Translation API](https://cloud.google.com/translate/)
- Supports more than 100 languages and thousands of language pairs
- automatic language detection
- continuous updates: Translation API is learning from logs analysis and human translation examples. Existing language pairs improve and new language pairs come online at no additional cost

## [Google Video Intelligence](https://cloud.google.com/video-intelligence/)
- Label Detection - Detect entities within the video, such as "dog", "flower" or "car"
- Shot Change Detection - Detect scene changes within the video
- Explicit Content Detection - Detect adult content within a video
- Video Transcription - Automatically transcribes video content in English

## [Google Vision API](https://cloud.google.com/vision/)
- Object recognition: detect broad sets of categories within an image, ranging from modes of transportation to animals
- Facial sentiment and logos: Analyze facial features to detect emotions: joy, sorrow, anger; detect logos
- Extract text: detect and extract text within an image, with support of many languages and automatic language identification
- Detect inapropriate content: fetect different types of inappropriate content from adult to violent content

## Experiments Frameworks
Tools to help you configure, organize, log and reproduce experiments
- https://www.reddit.com/r/MachineLearning/comments/5gyzqj/d_how_do_you_keep_track_of_your_experiments/, 2017
- [How to Plan and Run Machine Learning Experiments Systematically by Jason Brownlee, 2017](https://machinelearningmastery.com/plan-run-machine-learning-experiments-systematically/)
  - using a speadsheet with a template
- https://github.com/IDSIA/sacred

## Jupyter Notebook
- [Top 5 Best Jupyter Notebook Extensions](https://www.kdnuggets.com/2018/03/top-5-best-jupyter-notebook-extensions.html)
- [Version Control for Jupyter Notebook](https://towardsdatascience.com/version-control-for-jupyter-notebook-3e6cef13392d)

## [Lobe](https://lobe.ai/)
Lobe is an easy-to-use _visual_ tool (no coding required) that lets you build custom deep learning models, quickly train them, and ship them directly in your app without writing any code.

## [Microsoft Azure Bot Service](https://azure.microsoft.com/en-us/services/bot-service/)

## [Microsoft Azure Machine Learning](https://azure.microsoft.com/en-us/overview/machine-learning/)

## [Microsoft Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services)

## [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/)

### [Supervisely](https://supervise.ly/)
- Annotate images for computer vision tasks using AI
- https://github.com/supervisely/supervisely

## [Syn Bot Oscova](https://developer.syn.co.in/tutorial/bot/oscova/machine-learning.html)
- [finds similarity between the expressions](https://forum.syn.co.in/viewtopic.php?t=1845&p=3209)
- https://github.com/SynHub/syn-bot-samples
- MS Visual Studio is required (doesn't work with VS Code)
- activating Deep Learning feature requires [license activating](https://developer.syn.co.in/tutorial/bot/activate-license.html)
- number of requests to the server is limited by the license

## [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://www.tensorflow.org/hub/)
- https://github.com/tensorflow/models/tree/master/research
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples
  - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
    - TF Classify
    - TF Detect
    - TF Stylize
    - TF Speech
  - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/examples
    - TF Classify
    - TF Detect
    - TF Speech
  - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo
    - TF classify using tflite model
  - [Freeze tensorflow model graph](https://www.tensorflow.org/mobile/tflite/devguide#freeze_graph)
- [TensorFlow Estimator APIs Tutorials](https://github.com/GoogleCloudPlatform/tf-estimator-tutorials)

## [Turi Create](https://github.com/apple/turicreate)
Apple python framework that simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
- Export models to Core ML for use in iOS, macOS, watchOS, and tvOS apps.
- [A Guide to Turi Create](https://developer.apple.com/videos/play/wwdc2018/712/) from wwdc2018

# Playgrounds
- [Training Data Analyst](https://github.com/GoogleCloudPlatform/training-data-analyst) - Labs and demos for Google Cloud Platform courses 
- [SEEDBANK](http://tools.google.com/seedbank/) - Collection of Interactive Machine Learning Examples
- [AI Lab: Learn to Code with the Cutting-Edge Microsoft AI Platform, 2018](https://blogs.technet.microsoft.com/machinelearning/2018/06/19/ai-lab-learn-about-experience-code-with-the-cutting-edge-microsoft-ai-platform/)
- [Teachable Machine by Google](https://teachablemachine.withgoogle.com/)

## [Google AIY](https://aiyprojects.withgoogle.com)
- [Vision Kit](https://aiyprojects.withgoogle.com/vision/) - Do-it-yourself intelligent camera. Experiment with image recognition using neural networks on Raspberry Pi.
- [Voice Kit](https://aiyprojects.withgoogle.com/voice/) - Do-it-yourself intelligent speaker. Experiment with voice recognition and the Google Assistant on Raspberry Pi.

# IDEs
- https://colab.research.google.com
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/)
  - [Getting Started with Weka - Machine Learning Recipes #10](https://www.youtube.com/watch?v=TF1yh5PKaqI)
  
# Repositories
- https://github.com/bulutyazilim/awesome-datascience
  
# Models

## Decision Trees
Pros:
- can model nonlinearities
- are highly interpretable
- do not require extensive feature preprocessing
- do not require enormous data sets

Cons:
- tend to overfit
  - fixed by building a decision forest with boosting
- unstable/undeterministic (generate different results while trained on the same data)
  - fixed by using bootstrap aggregation/bagging (a boosted forest)
- do mapping directly from the raw input to the label
  - better use neural nets that can learn intermediate representations

Hyperparameters:
- tree depth
- maximum number of leaf nodes

## Distillation
  - trains a model to mimic the behavior of a pretrained model so it can work independently of the pretrained model
  - can train the smaller model with unlabeled examples
  - not all target classes need to be represented in the distillation training set
  - reduces the need for regularization
  - [Distilling the Knowledge in a Neural Network by Geoffrey Hinton et al, 2015](https://arxiv.org/abs/1503.02531)
  - [“Why Should I Trust You?” Explaining the Predictions of Any Classifier by Marco Tulio Ribeiro et al, 2016](https://arxiv.org/abs/1602.04938)
  - [Detecting Bias in Black-Box Models Using Transparent Model Distillation by Sarah Tan et al, 2017](https://arxiv.org/abs/1710.06169)

## Embedding models
- https://github.com/Hironsan/awesome-embedding-models
- [gensim's word2vec](https://code.google.com/archive/p/word2vec/source) (embedded words and phrases)
  - [online vocaburary update tutorial](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/online_w2v_tutorial.ipynb)
  - [How to Develop Word Embeddings in Python with Gensim](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)
- [gensim's doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)
- https://github.com/jhlau/doc2vec
- see recursive autoencoders
- see bag-of-words models

## Evolutionary Algorithms
- [Using Evolutionary AutoML to Discover Neural Network Architectures by by Esteban Real, 2018](https://research.googleblog.com/2018/03/using-evolutionary-automl-to-discover.html)
- [Regularized Evolution for Image Classifier Architecture Search by Esteban Real et al, 2018](https://arxiv.org/abs/1802.01548)
- [Welcoming the Era of Deep Neuroevolution by Jeff Clune, 2017](https://eng.uber.com/deep-neuroevolution/)
- [Hierarchical Representations for Efficient Architecture Search by Hanxiao Liu et al, 2017](https://arxiv.org/abs/1711.00436)
- [Learning Transferable Architectures for Scalable Image Recognition by Barret Zoph et al, 2017](https://arxiv.org/abs/1707.07012)
- [Large-Scale Evolution of Image Classifiers by Esteban Real et al, 2017](https://arxiv.org/abs/1703.01041)
- [Evolving Neural Networks through Augmenting Topologies by  Stanley and Miikkulainen, 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

## Metrics of dataset quality
- Statistical metrics
  - descriptive statistics: dimensionality, unique subject counts, systematic replicates counts, pdfs, cdfs (probability and cumulative distribution fx's)
  - cohort design
  - power analysis
  - sensitivity analysis
  - multiple testing correction analysis
  - dynamic range sensitivity
- Numerical analysis metrics
  - number of clusters
  - PCA dimensions
  - MDS space dimensions/distances/curves/surfaces
  - variance between buckets/bags/trees/branches
  - informative/discriminative indices (i.e. how much does the top 10 features differ from one another and the group)
  - feature engineering differnetiators

## Neural Networks
[Approaches](https://medium.com/@sayondutta/nuts-and-bolts-of-applying-deep-learning-by-andrew-ng-89e1cab8b602) when our model doesn’t work:
- Fetch more data
- Add more layers to Neural Network
- Try some new approach in Neural Network
- Train longer (increase the number of iterations)
- Change batch size
- Try Regularisation
- Check Bias Variance trade-off to avoid under and overfitting
- Use more GPUs for faster computation

Back-propagation problems:
- it requires labeled training data; while almost all data is unlabeled
- the learning time does not scale well, which means it is very slow in networks with multiple hidden layers
- it can get stuck in poor local optima, so for deep nets they are far from optimal.

### Capsule Networks
- [Understanding Hinton’s Capsule Networks by Max Pechyonkin, 2017](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
- [Capsule Networks (CapsNets) – Tutorial, 2017](https://www.youtube.com/watch?v=pPN8d0E3900)

### Convolutional Neural Networks

### Deep Residual Networks
- [Understand Deep Residual Networks — a simple, modular learning framework that has redefined state-of-the-art, 2017](https://blog.waya.ai/deep-residual-learning-9610bb62c355)

### Distributed Neural Networks
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer by Jeff Dean et al](https://arxiv.org/abs/1701.06538)
- [PathNet: Evolution Channels Gradient Descent in Super Neural Networks by deepmind](https://deepmind.com/research/publications/pathnet-evolution-channels-gradient-descent-super-neural-networks/)
- Feature extraction - uses layers of a pretrained model as inputs to another model, effectively chaining two models together

### Feed-Forward Neural Networks
- Perceptrons

### Gated Recurrent Neural Networks
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling by Junyoung Chung et al, 2014](https://arxiv.org/pdf/1412.3555v1.pdf)

### Generative Adversarial Networks
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation by Tero Karras et al, 2017](https://arxiv.org/abs/1710.10196)

### Long-Short Term Memory Networks
- [Exploring LSTMs, 2017](http://blog.echen.me/2017/05/30/exploring-lstms/)
- [Understanding LSTM Networks by Christopher Olah, 2015](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - “Almost all exciting results based on recurrent neural networks are achieved with [LSTMs].”
- [Offline Handwriting Recognition with Multidimensional Recurrent Neural Networks by Graves & Schmidhuber, 2009](http://people.idsia.ch/~juergen/nips2009.pdf)
  - showed that RNNs with LSTM are currently the best systems for reading cursive writing
- [LONG SHORT-TERM MEMORY by Hochreiter & Schmidhuber, 1997](http://www.bioinf.jku.at/publications/older/2604.pdf)

### Recurrent Neural Networks
- [The Unreasonable Effectiveness of Recurrent Neural Networks by Andrej Karpathy, 2015](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- see [Long-Short Term Memory Networks](https://github.com/lankastersky/neuromantic/blob/master/readme.md#long-short-term-memory-networks)

### Symmetrically Connected Networks
- Hopfield Nets (without hidden units)
  - [Neural networks and physical systems with emergent collective computational abilities by Hopfield, 1982](http://www.pnas.org/content/pnas/79/8/2554.full.pdf)
- Boltzmann machines (stochastic recurrent neural network with hidden units)
 -  Restricted Boltzmann Machines by Salakhutdinov and Hinton, 2014
 - [Deep Boltzmann Machines by Salakhutdinov and Hinton, 2012](http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf)

## Reinforcement Learning
- [Introducing a New Framework for Flexible and Reproducible Reinforcement Learning Research, 2018](https://ai.googleblog.com/2018/08/introducing-new-framework-for-flexible.html)
  - https://github.com/google/dopamine
- [Neural Architecture Search with Reinforcement Learning by Barret Zoph et al, 2017](https://arxiv.org/abs/1611.01578)

# Guidelines
- [AI at Google: our principles, 2018](https://blog.google/topics/ai/ai-principles/)
- [Rules of Machine Learning: Best Practices for ML Engineering by Martin Zinkevich, 2018](https://developers.google.com/machine-learning/rules-of-ml/)
- [Practical advice for analysis of large, complex data sets by PATRICK RILEY, 2016](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)
- [What’s your ML test score? A rubric for ML production systems by Eric Breck, 2016](https://research.google.com/pubs/pub45742.html)
- [Machine Learning: The High Interest Credit Card of Technical Debt by D. Sculley et al, 2014](https://research.google.com/pubs/pub43146.html)
  - Complex Models Erode Boundaries
    - Entanglement
    - Hidden Feedback Loops
    - Undeclared Consumers
  - Data Dependencies Cost More than Code Dependencies
    - Unstable Data Dependencies
    - Underutilized Data Dependencies
    - Static Analysis of Data Dependencies
    - Correction Cascades
  - System-level Spaghetti
    - Glue Code
    - Pipeline Jungles
    - Dead Experimental Codepaths
    - Configuration Debt
  - Dealing with Changes in the External World
    - Fixed Thresholds in Dynamic Systems
    - When Correlations No Longer Correlate
    - Monitoring and Testing
- [Principles of Research Code by Charles Sutton, 2012](http://www.theexclusive.org/2012/08/principles-of-research-code.html)
- [Patterns for Research in Machine Learning by Ali Eslami, 2012](http://arkitus.com/patterns-for-research-in-machine-learning/)
- [Lessons learned developing a practical large scale machine learning system by Simon Tong, 2010](https://research.googleblog.com/2010/04/lessons-learned-developing-practical.html)
- [The Professional Data Science Manifesto](http://www.datasciencemanifesto.org/)
- [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/)

## Deep learning
- [Deep Learning: A Critical Appraisal by Gary Marcus, 2018](https://arxiv.org/abs/1801.00631)
  - Deep learning thus far is data hungry 
  - Deep learning thus far is shallow and has limited capacity for transfer
  - Deep learning thus far has no natural way to deal with hierarchical structure
  - Deep learning thus far has struggled with open-ended inference
  - Deep learning thus far is not sufficiently transparent 
  - Deep learning thus far has not been well integrated with prior knowledge
  - Deep learning thus far cannot inherently distinguish causation from correlation
  - Deep learning presumes a largely stable world, in ways that may be problematic
  - Deep learning thus far works well as an approximation, but its answers often cannot be fully trusted
  - Deep learning thus far is difficult to engineer with 
- [Software 2.0 by Andrej Karpathy, 2017](https://medium.com/@karpathy/software-2-0-a64152b37c35)

# Interview preparation
- [20 Questions to Detect Fake Data Scientists and How to Answer Them, 2018](https://medium.com/@kojinoshiba/20-questions-to-detect-fake-data-scientists-and-how-to-answer-them-16c816829294)
- [Собеседование по Data Science: чего от вас ждут, 2018](https://habr.com/company/epam_systems/blog/350654/)
- [Acing AI Interviews](https://medium.com/acing-ai/acing-ai-interviews/home)

# MOOC
## Google oriented courses
- https://developers.google.com/machine-learning/crash-course/
  - for beginners, explains hard things with simple words
  - from google gurus
  - uses TensorFlow and codelabs
- https://www.coursera.org/specializations/gcp-data-machine-learning
  - shows how to use GCP for machine learning

# Books

 ## NLP
 - [Natural Language Processing with Python by Steven Bird et al, 2014](http://www.nltk.org/book/)

## Statistics
- [Bayesian Methods for Hackers: Probabilistic Programming and Bayesian Inference by Cameron Davidson-Pilon, 2015](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [Statistics is Easy! by Dennis Shasha, 2010](https://www.amazon.com/Statistics-Second-Synthesis-Lectures-Mathematics/dp/160845570X)

# Datasets
- https://ai.google/tools/datasets/
- https://toolbox.google.com/datasetsearch
  - [Making it easier to discover datasets, 2018](https://www.blog.google/products/search/making-it-easier-discover-datasets/)
- [Microsoft Research Open Data](https://msropendata.com/)
  - users can also copy datasets directly to an Azure based Data Science virtual machine
  
## 3D
- [ScanNet](http://www.scan-net.org/) - RGB-D video dataset annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations
- [SceneNet](https://robotvault.bitbucket.io/scenenet-rgbd.html) - Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth

## Audios
- [The VU sound corpus](https://github.com/CrowdTruth/vu-sound-corpus) - based on https://freesound.org/ database
  - See article [The VU Sound Corpus by Emiel van Miltenburg et al](http://www.lrec-conf.org/proceedings/lrec2016/pdf/206_Paper.pdf)
- [AudioSet](https://research.google.com/audioset/) - consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos


## Images
- [Conceptual Captions: A New Dataset and Challenge for Image Captioning, 2018](https://ai.googleblog.com/2018/09/conceptual-captions-new-dataset-and.html)
- [Landmarks 2018](https://research.googleblog.com/2018/03/google-landmarks-new-dataset-and.html)
- [ImageNet](http://www.image-net.org/)
- [COCO](http://cocodataset.org/#home)
- [SUN](https://groups.csail.mit.edu/vision/SUN/)
- [Caltech 256](https://authors.library.caltech.edu/7694/)
- [Pascal](https://www.cs.stanford.edu/~roozbeh/pascal-context/)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - 60000 32x32 colour images in 10 classes, with 6000 images per class
  - commonly used to train image classifiers

## Videos
- [Microsoft multimedia challenge dataset, 2017](http://ms-multimedia-challenge.com/2017/dataset)
  - largest dataset in terms of sentence and vocabulary
  - challenge: to automatically generate a complete and natural sentence to describe video content
- [Kinetics, 2017](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
- [YouTube-8M, 2017](https://research.google.com/youtube8m/)
  - large, but annotations are slightly noisy and only video-level labels have been assigned (include frames that do not
relate to target actions)
  - [youtube-dl](https://github.com/rg3/youtube-dl) - Command-line program to download videos from YouTube.com and other video sites
- [Sports-1M by A. Karpathy, 2016](https://github.com/gtoderici/sports-1m-dataset/blob/wiki/ProjectHome.md)
  - large, but annotations are slightly noisy and only video-level labels have been assigned (include frames that do not
relate to target actions)
- [FCVID](http://lsvc17.azurewebsites.net/#data)
- [ActivityNet](http://activity-net.org/download.html)
- http://crcv.ucf.edu/data/UCF101.php 2013
- [Hollywood2](http://www.di.ens.fr/~laptev/actions/hollywood2/)
- [HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [CCV](http://www.ee.columbia.edu/ln/dvmm/CCV/)

# Research Groups
- [DeepMind](https://deepmind.com/)
- [Facebook AI Research (FAIR)](https://research.fb.com/category/facebook-ai-research-fair/)
- [Google Brain](https://research.google.com/teams/brain/)
- [Microsoft Research AI](https://www.microsoft.com/en-us/research/lab/microsoft-research-ai/)
- [OpenAI](http://openai.com/)
- [Sentient Labs](https://www.sentient.ai/)
- [Uber Labs](https://eng.uber.com/tag/ai/) 

# Cartoons
[The Browser of a Data Scientist](https://www.datasciencecentral.com/profiles/blogs/the-browser-of-a-data-scientist)
- ![The Browser of a Data Scientist](https://api.ning.com/files/R-lMgokaLrIz-u5lxqyR-EFKJTkzJjUKjXAFvQDvIWRIP5Hc4x2Q3XnnAzFyw4zPIURUCjfcFzGaBxgHtsZhuYMYmDQJhaS*/oxi.PNG "The Browser of a Data Scientist")
## Jokes
A statistician drowned crossing a river that was only three feet deep
on average
