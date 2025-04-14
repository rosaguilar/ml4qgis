## Machine Learning for QGIS

A processing plugin for using machine learning in QGIS.

It exposes the power of [scikit-learn](https://scikit-learn.org/stable/) through QGIS processing.

**Random Forest** is a robust, well-known machine learning algorithm for classification and regression tasks. 
It works by creating multiple decision trees during the training. The output is generated via majority voting in case of classification
or the average of the prediction of the trees in case of regression.
With this plugin, you can execute a Random Forest classifier that uses 300 estimators (trees).
A random_state is set to seven for reproducibility.

The algorithm requires two parameters:

<ul>
  <li>A point vector layer with a numeric field called *label* as input data for training</li>
  <li>The image to process</li>
  <li>optionally a name for the output that is the classified image</li>
</ul>

You only need to prepare your training samples by adding points and label. 
The plugin will directly sample the raster to prepare the training data that is randomly split into training and testing sets.

## Quick intro
Watch the video below on how to run **ml4qgis**


[![Watch the video](https://img.youtube.com/vi/Edn0epdH5A8/0.jpg)](https://www.youtube.com/watch?v=Edn0epdH5A8)


# Development
This plugin is developed in collaboration with Rosa Aguilar (ITC/University of Twente) and Matthias Kuhn (OpenGIS.ch).


