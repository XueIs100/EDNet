This is a network for driver steering intention detection.

The data that support the ﬁndings of this study are openly available in Brain4Cars repository at https://www.brain4cars.com.
The data that support the ﬁndings of this study are openly available in Zenodo with the following DOI: http://dx.doi.org/10.5281/zenodo.1009540.

The following methods are compared with our proposed approach:
1. Car that knows before you do: Anticipating maneuvers via learning temporal driving models.
   They propose an autoregressive Input-Output HMM to model the contextual information and operations using in-car and out-of-car recorded videos.
2. Recurrent neural networks for driver activity anticipation via sensory-fusion architecture.
   They present a deep learning approach for prediction in perceptually rich robotics applications. They introduce a sensory fusion architecture that jointly learns to predict and fuse information from multiple sensory streams. The architecture consists of a recurrent neural network (rnn) that uses long short-term memory (LSTM) units to capture long time dependencies.
3. Dilated convolutional neural network for predicting driver’s activity.
   They propose a deep learning model architecture for classifying driver behaviour and activities in real-life scenarios of driving a car under different conditions. Sensory data comes from a variety of sources including a driver facing camera (inside camera), a road facing camera (outside camera), GPS and other car related sensors. It uses convolution and max-pooling pairs to understand spatial relationships within video frames and incorporates dilated deep convolutional structures to capture long temporal dependencies, process and predict driver's activities.
4. Driver intention anticipation based on in-cabin and driving scene monitoring.
   They propose a framework for the detection of the drivers' intention based on both in-cabin and trafﬁc scene videos. More speciﬁcally, they propose a Convolutional-LSTM (ConvLSTM)-based auto-encoder to extract motion features from the out-cabin trafﬁc, train a classiﬁer which considers motions from both in- and outside of the cabin jointly for maneuver intention anticipation. 
5. End-to-end prediction of driver intention using 3d convolutional neural networks.
   They present a vision-based system based on deep neural networks with 3D convolutions and residual learning for anticipating the future maneuver based on driver observation. The architecture consists of three components: a neural network, a 3D residual network and a Long Short-Term Memory network (LSTM) for handling temporal data of varying length.
6. Driver action prediction using deep (bidirectional) recurrent neural network.
   They present such a driver action prediction system, including a real-time data acquisition, processing and learning framework for predicting future or impending driver action. The proposed system incorporates camera-based knowledge of the driving environment and the driver themselves, in addition to traditional vehicle dynamics. It then uses a deep bidirectional recurrent neural network (DBRNN) to learn the correlation between sensory inputs and impending driver behavior achieving accurate and high horizon action prediction.
7. Real-time driver maneuver prediction using LSTM.
   They developed a model to predict driver maneuvers. They propose a deep learning method based on Long Short-Term Memory (LSTM) which utilizes data on the driver's gaze and head position as well as vehicle dynamics data.
8. Cemformer: Learning to predict driver intentions from in-cabin and external cameras via spatial-temporal transformers.
   They introduce a new framework called Cross-View Episodic Memory Transformer (CEMFormer), which employs spatio-temporal transformers to learn uniﬁed memory representations for an improved driver intention prediction. Speciﬁcally, they develop a spatial-temporal encoder to integrate information from both in-cabin and external camera views, along with episodic memory representations to continuously fuse historical data.
9. Temporal Information Fusion Network for Driving Behavior Prediction.
   They propose a lightweight end-to-end model, temporal information fusion network (TIFN). The state update cell (STU) is proposed to introduce the inffuence of environment information into the driver's state modeling, inspired by the selective attention of the human cognition process. Meanwhile, semantic segmentation features are extracted to offer clear clues affecting driver attention in place of motion optical ffow images and binary value vectors. Finally, the driver's intention and environment state are combined to make a joint prediction.
