## Machine Learning for QGIS

A processing plugin for using machine learning in QGIS.

It exposes the power of [scikit-learn](https://scikit-learn.org/stable/) through QGIS processing.


**Random Forest** is a robust, well-known machine learning algorithm for classification and regression tasks. 
It works by creating multiple decision trees during the training. The output is generated via majority voting in case of classification
or the average of the prediction of the trees in case of regression.
With this plugin, you can execute a Random Forest classifier that uses 300 estimators (trees).
A random_state is set to seven for reproducibility.

# Development
This plugin is developed in collaboration with Rosa Aguilar (ITC/University of Twente) and Matthias Kuhn (OpenGIS.ch).
