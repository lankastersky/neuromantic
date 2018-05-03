# TOC
- [Use Cases](#use-cases)
  - [Audio recognition](#audio-recognition)
    - [Speech to Text](#speech-to-text)
  - [Data Agumentation](#data-augmentation)
  - [Design](#design)
  - [Games](#games)
  - [Gesture Recognition](#gesture-recognition)
  - [Image Recognition](#image-recognition)
    - [Face Recognition](#face-recognition)
    - [Image Captioning](#image-captioning)
    - [Person Detection](#person-detection)
    - [Semantic segmentation](#semantic-segmentation)
  - [Interpretability](#interpretability)
  - [Programming and ML](#programming-and-ml)
    - [Predict defects](#predict-defects)  
    - [Predict performance](#predict-performance)  
  - [NLP](#nlp)
    - [Chatbots](#chatbots)
    - [Crossword question answerers](#crossword-question-answerers)
    - [Database queries](#database-queries)
    - [Named entity resolution](#named-entity-resolution)
    - [Reverse dictionaries](#reverse-dictionaries)
    - [Sequence to sequence](#sequence-to-sequence)
    - [Sentiment analysis](#sentiment-analysis)
    - [Spelling](#spelling)
    - [Summarization](#summarization)
    - [Text to Image](#text-to-image)
    - [Text to Speech](#text-to-speech)
  - [Personality recognition](#personality-recognition)
  - [Search](#search)
  - [Transfer Learning](#transfer-learning)
  - [Uber](#uber)
  - [Video recognition](#video-recognition)
    - [Body Recognition](#body-recognition)
    - [Object Detection](#object-detection)
    - [Scene Segmentation](#scene-segmentation)
    - [Video Captioning](#video-captioning)
    - [Video Classification](#video-classification)
- [Tools](#tools)
  - [Google Cloud AutoML](#google-cloud-automl)
  - [Google Cloud ML Engine](#google-cloud-ml-engine)
  - [Google Mobile Vision](#google-mobile-vision)
  - [Google Video Intelligence](#google-video-intelligence)
  - [Experiments Frameworks](#experiments-frameworks)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Microsoft Azure Bot Service](#microsoft-azure-bot-service)
  - [Microsoft Azure Machine Learning](#microsoft-azure-machine-learning)
  - [Microsoft Cognitive Services](#microsoft-cognitive-services)
  - [Microsoft Cognitive Toolkit](#microsoft-cognitive-toolkit)
  - [Syn Bot Oscova](#syn-botoscova)
  - [TensorFlow](#tensorflow)
- [Playgrounds](#playgrounds)
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
- [Guidelines](#guidelines)
  - [Deep learning](#deep-learning)
- [Interview preparation](#interview-preparation)
- [Books](#books)
  - [NLP](#nlp-1)
  - [Statistics](#statistics)
- [MOOC](#mooc)
  - [Google oriented courses](#google-oriented-courses)
- [Datasets](#datasets)
  - [Images](#images)
  - [Videos](#videos)
- [Research groups](#research-groups)
- [Cartoons](#cartoons)

# Use Cases
- [Sample Projects for Data Scientists in Training by V Granville, 2018](https://www.datasciencecentral.com/profiles/blogs/sample-projects-for-data-scientists-in-training)

## Audio Recognition
- [CNN Architectures for Large-Scale Audio Classification, S. Hershey et al, 2017](https://research.google.com/pubs/pub45611.html)
  - [vggish model used to generate google's AudioSet](https://github.com/tensorflow/models/tree/master/research/audioset)
  - [vggish model adapted for Keras](https://github.com/DTaoo/VGGish)
- [Audio Set: An ontology and human-labeled dataset for audio events, 2017](https://research.google.com/pubs/pub45857.html) 
- [Large-Scale Audio Event Discovery in One Million YouTube Videos, A. Jansen et al, ICASSP 2017](https://research.google.com/pubs/pub45760.html)
- [How do I listen for a sound that matches a pre-recorded sound?](https://arduino.stackexchange.com/questions/8781/how-do-i-listen-for-a-sound-that-matches-a-pre-recorded-sound)
- The Sound Sensor Alert App [sentector](http://sentector.com/)

## Speech to Text
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

## Image Recognition
- [MobileNetV2: The Next Generation of On-Device Computer Vision Networks, 2018](https://research.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- [Large-Scale Evolution of Image Classifiers by Esteban Real et al, 2017](https://arxiv.org/pdf/1703.01041.pdf)
- [Rethinking the Inception Architecture for Computer Vision by Christian Szegedy et al, 2015](https://arxiv.org/abs/1512.00567)
  - [Inception in TensorFlow](https://github.com/tensorflow/models/tree/master/research/inception) - 1.4M images and 1000 classes
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications by Andrew G. Howard et al, 2017](https://arxiv.org/abs/1704.04861)
  - [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
  - Similar approach on practice: [How HBO’s Silicon Valley built “Not Hotdog” with mobile TensorFlow, Keras & React Native](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)
- [Going Deeper with Convolutions by C. Szegedy et al, 2014](https://arxiv.org/abs/1409.4842)
- [ImageNet Classification with Deep Convolutional Neural Networks by Alex Krizhevsky et al, 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  - [ImageNet](http://image-net.org/)
  - the model is based on CNN
- [Xception: Deep Learning with Depthwise Separable Convolutions by François Chollet, 2017](https://arxiv.org/abs/1610.02357)
- [ImageNet Classification with Deep Convolutional Neural Networks by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### Face Recognition
- [Умные фотографии ВКонтакте, 2018](https://vk.com/@td-highload-face-recognition) (Smart photos in Vkontakte)
- [FaceNet: A Unified Embedding for Face Recognition and Clustering by Florian Schroff et al, 2015](https://arxiv.org/abs/1503.03832)
  - the model: FaceNet

### Image Captioning
- [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge by Oriol Vinyals et al, 2016](https://arxiv.org/abs/1609.06647)

### Person Detection
- [Automatic Portrait Segmentation for Image Stylization by Xiaoyong Shen1 et al, 2016](http://www.cse.cuhk.edu.hk/leojia/papers/portrait_eg16.pdf)

### Semantic Segmentation
- [Semantic Image Segmentation with DeepLab in Tensorflow, 2018](https://research.googleblog.com/2018/03/semantic-image-segmentation-with.html)
  - model DeepLab-v3+ built on top of CNN
- https://github.com/facebookresearch/Detectron, see links to articles at the end of the page
- [Rethinking Atrous Convolution for Semantic Image Segmentation by Liang-Chieh Chen et al, 2017](https://arxiv.org/pdf/1706.05587.pdf)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs by Liang-Chieh Chen et al, 2017](https://arxiv.org/pdf/1606.00915.pdf)
- [Mask R-CNN by Kaiming He et al, 2017](https://arxiv.org/abs/1703.06870)

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
- [To type or not to type: quantifying detectable bugs in JavaScript by Gao et al, 2017](https://blog.acolyer.org/2017/09/19/to-type-or-not-to-type-quantifying-detectable-bugs-in-javascript/)
- [A Survey of Machine Learning for Big Code and Naturalness by Miltiadis Allamanis et al, 2017](https://arxiv.org/abs/1709.06182)
- https://www.deepcode.ai
- https://codescene.io

### Predict defects
- [Predicting Defects for Eclipse by T Zimmermann at al, 2007](http://thomas-zimmermann.com/publications/files/zimmermann-promise-2007.pdf)
  - used code complexity metrics as features and logistic regression for classification (if file/module has defects) and linear regression for ranking (how many defects)
- [Predicting Component Failures at Design Time by Adrian Schroter et al, 2006](http://thomas-zimmermann.com/publications/files/schroeter-isese-2006.pdf)
  - showed that design data such as import relationships can predict failures
  - used the number of failures in a component as dependent variable and the imported resources used from this component as input features
- [Mining Version Histories to Guide Software Changes by T Zimmermann at al, 2004](http://www.ics.uci.edu/~andre/ics228s2006/zimmermanweissgerberdiehlzeller.pdf)
  - used [apriory algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm) to predict likely changes in files/modules

# Predict performance
  - [3 ways AI will change project management for the better, 2017](https://www.atlassian.com/blog/software-teams/3-ways-ai-will-change-project-management-better)
- [A deep learning model for estimating story points by Morakot Choetkiertikul et al, 2016](https://arxiv.org/pdf/1609.00489.pdf)
  - estimating story points based on long short-term memory and recurrent highway network

## NLP
- [Text Embedding Models Contain Bias. Here's Why That Matters, 2018](https://developers.googleblog.com/2018/04/text-embedding-models-contain-bias.html)
- [How to Clean Text for Machine Learning with Python](https://machinelearningmastery.com/clean-text-machine-learning-python/)

### Chatbots
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

Evaluation:
- [Semantic Textual Similarity Wiki](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page)

### Sentiment analysis
- [Twitter Sentiment Analysis Using Combined LSTM-CNN Models by SOSAVPM, 2018](http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/)
  - https://github.com/pmsosa/CS291K
  - used pre-trained embeddings with LSTM-CNN model with dropouts
  - 75.2% accuracy for binary classification (positive-negative tweet)
- [doc2vec example, 2015](http://linanqiu.github.io/2015/10/07/word2vec-sentiment/)

### Spelling
- [How to Write a Spelling Corrector](http://norvig.com/spell-correct.html)

### Summarization
- [Generating Wikipedia by Summarizing Long Sequences by Peter J. Liu et al, 2018](https://arxiv.org/abs/1801.10198)

### Text to Image
- [ChatPainter: Improving text-to-image generation by using dialogue](https://www.microsoft.com/en-us/research/blog/chatpainter-improving-text-image-generation-using-dialogue/)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks by Tao Xu et al, 2017](https://arxiv.org/abs/1711.10485)
  - [Microsoft researchers build a bot that draws what you tell it to, 2018](https://blogs.microsoft.com/ai/drawing-ai/)

### Text to Speech
- [WaveNet: A Generative Model for Raw Audio, 2016](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

## Personality recognition
- Mining Facebook Data for Predictive Personality Modeling (Dejan Markovikj,Sonja Gievska, Michal Kosinski, David Stillwell)
- Personality Traits Recognition on Social Network — Facebook (Firoj Alam, Evgeny A. Stepanov, Giuseppe Riccardi)
- The Relationship Between Dimensions of Love, Personality, and Relationship Length (Gorkan Ahmetoglu, Viren Swami, Tomas Chamorro-Premuzic)

## Search
- [Neural Architecture Search with Reinforcement Learning by Barret Zoph et al, 2017](https://arxiv.org/abs/1611.01578)
- [Can word2vec be used for search?](https://www.reddit.com/r/MachineLearning/comments/4mw927/can_word2vec_be_used_for_search/)
  - alternative search queries can be built using approximate nearest neighbors in embedding vectors space of terms (using https://github.com/spotify/annoy e.g.)
  - [Improving Document Ranking with Dual Word Embeddings by Eric Nalisnick et al, 2016](https://www.microsoft.com/en-us/research/publication/improving-document-ranking-with-dual-word-embeddings/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2Fdefault.aspx%3Fid%3D260867)

## Transfer Learning
- [Deep Learning & Art: Neural Style Transfer – An Implementation with Tensorflow in Python](https://www.datasciencecentral.com/profiles/blogs/deep-learning-amp-art-neural-style-transfer-an-implementation)
- [Image Classification using Flowers dataset on Cloud ML Enginge](https://cloud.google.com/ml-engine/docs/flowers-tutorial)

## Uber
- [Engineering More Reliable Transportation with Machine Learning and AI at Uber, 2017](https://eng.uber.com/machine-learning/)

## Video recognition

### Body recognition
- [Enabling full body AR with Mask R-CNN2Go by Fei Yang et al, 2018](https://research.fb.com/enabling-full-body-ar-with-mask-r-cnn2go/)

### Object detection
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
  - [YOLOv3: An Incremental Improvement by Joseph Redmon, Ali Farhadi, 2018](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [Mobile Real-time Video Segmentation, 2018](https://research.googleblog.com/2018/03/mobile-real-time-video-segmentation.html)
  - integrated into Youtube stories
- [Supercharge your Computer Vision models with the TensorFlow Object Detection API, 2017](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html)
  - https://github.com/tensorflow/models/tree/master/research/object_detection
- [Ridiculously Fast Shot Boundary Detection with Fully Convolutional Neural Networks by Michael Gygli, 2017](https://arxiv.org/abs/1705.08214)
- [Video Shot Boundary Detection based on Color Histogram by J. Mas and G. Fernandez, 2003](http://www-nlpir.nist.gov/projects/tvpubs/tvpapers03/ramonlull.paper.pdf)

### Scene Segmentation
Detects when one video (shot/scene/chapter) ends and another begins
- [Recurrent Switching Linear Dynamical Systems by Scott W. Linderman et al, 2016](https://arxiv.org/abs/1610.08466)
- [Video Scene Segmentation Using Markov Chain Monte Carlo by Yun Zha et al, 2006](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1658031)
- [Automatic Video Scene Segmentation based on Spatial-Temporal Clues and Rhythm by Walid Mahdi et al, 2000](https://arxiv.org/abs/1412.4470)

### Video captioning
- [DeepStory: Video Story QA by Deep Embedded Memory Networks by Kyung-Min Kim et al, 2017](https://arxiv.org/abs/1707.00836)
  - https://github.com/Kyung-Min/Deep-Embedded-Memory-Networks
- [Video Understanding: From Video Classification to Captioning by Jiajun Sun et al, 2017](http://cs231n.stanford.edu/reports/2017/pdfs/709.pdf)
- [Unsupervised Learning from Narrated Instruction Videos by Jean-Baptiste Alayrac et al, 2015](https://arxiv.org/abs/1506.09215)

### Video classification
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

# Tools
- [50+ Useful Machine Learning & Prediction APIs, 2018](https://www.kdnuggets.com/2018/05/50-useful-machine-learning-prediction-apis-2018-edition.html)
  - Face and Image Recognition
  - Text Analysis, NLP, Sentiment Analysis
  - Language Translation
  - Machine Learning and prediction

## [Google Cloud AutoML](https://cloud.google.com/automl/)

Pros:
- let users train their own custom machine learning algorithms from scratch, without having to write a single line of code
- uses Transfer Learning (the more data and customers, the better results)
- is fully integrated with other Google Cloud services (Google Cloud Storage to store data, use Cloud ML or Vision API to customize the model etc.)

Cons:
- limited to image recognition (2018-Q1)
- doesn't allow to download a trained model

## [Google Cloud ML Engine](https://cloud.google.com/ml-engine/)
- [Samples & Tutorials](https://cloud.google.com/ml-engine/docs/tutorials)

## [Google Mobile Vision](https://developers.google.com/vision/)

Pros:
- Detect Faces (finds facial landmarks such as the eyes, nose, and mouth; doesn't identifies a person)
- Scan barcodes
- Recognize Text

Cons:

## [Google Video Intelligence](https://cloud.google.com/video-intelligence/)
- Label Detection - Detect entities within the video, such as "dog", "flower" or "car"
- Shot Change Detection - Detect scene changes within the video
- Explicit Content Detection - Detect adult content within a video
- Video Transcription - Automatically transcribes video content in English

## Experiments Frameworks
Tools to help you configure, organize, log and reproduce experiments
- https://www.reddit.com/r/MachineLearning/comments/5gyzqj/d_how_do_you_keep_track_of_your_experiments/, 2017
- [How to Plan and Run Machine Learning Experiments Systematically by Jason Brownlee, 2017](https://machinelearningmastery.com/plan-run-machine-learning-experiments-systematically/)
  - using a speadsheet with a template
- https://github.com/IDSIA/sacred

## Jupyter Notebook
- [Top 5 Best Jupyter Notebook Extensions](https://www.kdnuggets.com/2018/03/top-5-best-jupyter-notebook-extensions.html)
- [Version Control for Jupyter Notebook](https://towardsdatascience.com/version-control-for-jupyter-notebook-3e6cef13392d)

## [Microsoft Azure Bot Service](https://azure.microsoft.com/en-us/services/bot-service/)

## [Microsoft Azure Machine Learning](https://azure.microsoft.com/en-us/overview/machine-learning/)

## [Microsoft Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services)

## [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/)

## [Syn Bot Oscova](https://developer.syn.co.in/tutorial/bot/oscova/machine-learning.html)
- [finds similarity between the expressions](https://forum.syn.co.in/viewtopic.php?t=1845&p=3209)
- https://github.com/SynHub/syn-bot-samples
- MS Visual Studio is required (doesn't work with VS Code)
- activating Deep Learning feature requires [license activating](https://developer.syn.co.in/tutorial/bot/activate-license.html)
- number of requests to the server is limited by the license

## [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://www.tensorflow.org/hub/)

# Playgrounds
- [Teachable Machine by Google](https://teachablemachine.withgoogle.com/)

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

# Guidelines
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

## Audios
- [The VU sound corpus](https://github.com/CrowdTruth/vu-sound-corpus) - based on https://freesound.org/ database
  - See article [The VU Sound Corpus by Emiel van Miltenburg et al](http://www.lrec-conf.org/proceedings/lrec2016/pdf/206_Paper.pdf)
- [AudioSet](https://research.google.com/audioset/) - consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos


## Images
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
