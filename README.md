# FER using GNNs

This project basically is for the Facial Expression Recognition (FER) using Graph Neural Networks. 

The dataset used is "FER2013", there are seven emotions: angry, disgust, fear, happy, neutral, sad, surpise.

Graph Data preparation steps:
1) We used MediaPipe (developed by google) library for face landmark detection, this gives 3D face landmarks
   with 478 landmarks points and 2556 connections between landmark points.
2) For graph data, we create graph with 468 nodes and 2556 edges and label each graph to the corresponding image
   label (we created directed and undirected graph w/o self-loops for our study.)
3) GNNs are used for feature extractions followed by global average pooling and final classifier.
4) Basically our task is graph classification task, which will classify the graph (corresponding each emotions).

**Note:** We have not achieved expected accuracy in this task, this will be our future work. Just leaned how to
create graph, work with it and about graph neural networks.
