"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
* Author: Rosa Aguilar email: rosamaguilar@gmail.com, r.aguilar@utwente.nl*
***************************************************************************
"""

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsRaster,
)
from qgis.PyQt.QtCore import QCoreApplication
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


class RandomForestProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an algorithm that runs random forest on an image using
    training data specified in another layer.
    """

    TRAINING_DATA = "TRAINING_DATA"
    SOURCE_IMAGE = "SOURCE_IMAGE"
    CLASSIFIED_IMAGE = "CLASSIFIED_IMAGE"

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return RandomForestProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "randomforest"

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("Random Forest")

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr("Image Classification")

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "mlimageclassification"

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr(
            """
            Executes a random forest algorithm to classify an image.
            The training data is randomly split in 2/3 for training the model and 1/3 for testing. The output is a classified image (1 band).
            """
        )

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # input vector features source. It can have any kind of  geometry.
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.TRAINING_DATA,
                self.tr("Training data"),
                [QgsProcessing.SourceType.TypeVectorPoint],
            )
        )
        # Raster image source. A raster format.
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.SOURCE_IMAGE, self.tr("Image to process"), [QgsProcessing.TypeRaster]
            )
        )

        # We add an output to store the result
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.CLASSIFIED_IMAGE, self.tr("Classified"))
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        source = self.parameterAsSource(parameters, self.TRAINING_DATA, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.TRAINING_DATA))
        sourceImage = self.parameterAsRasterLayer(parameters, self.SOURCE_IMAGE, context)
        if sourceImage is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.SOURCE_IMAGE))

        extent = sourceImage.extent()

        xmin = extent.xMinimum()  # Top-left X (origin X)
        ymax = extent.yMaximum()  # Top-left Y (origin Y)

        crs = sourceImage.crs()

        provider = sourceImage.dataProvider()
        num_bands = provider.bandCount()

        label_field_index = source.fields().indexFromName("label")
        if label_field_index < 0:
            raise QgsProcessingException("No attribute named 'label' in layer")

        transform = QgsCoordinateTransform(
            source.sourceCrs(), sourceImage.crs(), context.project()
        )

        feature_list = []
        for feature in source.getFeatures():
            # Identify the raster values at the point for all bands
            results = provider.identify(
                feature.geometry().transform(transform).asPoint(), QgsRaster.IdentifyFormatValue
            )

            # Each sample contains: all band values first and the label last
            if results.isValid():
                values = []
                for band in range(1, num_bands + 1):  # Bands are 1-based
                    values.append(results.results()[band])
                values.append(feature.attributes()[label_field_index])
                feature_list.append(values)
            else:
                feedback.pushError(
                    f"Could not identify raster values at point {feature.geometry().asWkt()}"
                )

        # Convert the list of attributes + geometries to a NumPy array
        feature_array = np.array(feature_list)

        # get pixel size
        pixelSizeX = sourceImage.rasterUnitsPerPixelX()
        pixelSizeY = sourceImage.rasterUnitsPerPixelY()

        dataArray = sourceImage.as_numpy()

        # reshape the dataArray to get bands as columns
        bands, height, width = dataArray.shape
        reshaped_image = dataArray.reshape(bands, height * width).T

        # train/test split - by default the last column is y.
        X = feature_array[:, 0:bands]
        y = feature_array[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

        # instantiate the classifier
        rf = RandomForestClassifier(n_estimators=300, random_state=7)
        rf.fit(X_train, y_train)
        y_ts = rf.predict(X_test)

        OA = accuracy_score(y_test, y_ts)
        CFM = confusion_matrix(y_test, y_ts)
        classifiedImage = rf.predict(reshaped_image)
        feedback.pushInfo(f"Performance is OA{OA}, CFM {CFM}")

        # classified reshaped
        classifiedImage_reshaped = classifiedImage.T.reshape(1, height, width)

        # Squeeze the array to remove the first dimension (1, 224, 224) -> (224, 224)
        class_image_2d = np.squeeze(classifiedImage_reshaped, axis=0)

        geotransform = (xmin, pixelSizeX, 0, ymax, 0, -pixelSizeY)

        output_file = self.parameterAsOutputLayer(parameters, self.CLASSIFIED_IMAGE, context)

        driver = gdal.GetDriverByName("GTiff")
        nbands, rows, cols = classifiedImage_reshaped.shape
        out_raster = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)

        out_raster.SetGeoTransform(geotransform)
        out_raster.SetProjection(crs.toWkt())

        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(class_image_2d)
        out_band.FlushCache()

        return {self.OUTPUT: out_band}
