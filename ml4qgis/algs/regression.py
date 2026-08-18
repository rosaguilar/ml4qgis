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
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsRaster,
)
from qgis.PyQt.QtCore import QCoreApplication

HAS_SKLEARN_DEPENDENCY = True
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
except ImportError:
    HAS_SKLEARN_DEPENDENCY = False

HAS_LIGHTGBM_DEPENDENCY = True
try:
    from lightgbm import LGBMRegressor
except ImportError:
    HAS_LIGHTGBM_DEPENDENCY = False


class RegressionProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    Runs a regression estimator (Random Forest or LightGBM) on an image
    using point training data with a numeric target field.
    """

    TRAINING_DATA = "TRAINING_DATA"
    TARGET_FIELD = "TARGET_FIELD"
    SOURCE_IMAGE = "SOURCE_IMAGE"
    ESTIMATOR = "ESTIMATOR"
    N_ESTIMATORS = "N_ESTIMATORS"
    PREDICTED_IMAGE = "PREDICTED_IMAGE"

    ESTIMATORS = ["Random Forest", "LightGBM"]

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return RegressionProcessingAlgorithm()

    def name(self):
        return "regression"

    def displayName(self):
        return self.tr("Regression")

    def group(self):
        return self.tr("Image Regression")

    def groupId(self):
        return "mlimageregression"

    def shortHelpString(self):
        return self.tr(
            """
            Executes a regression algorithm to predict a continuous value per pixel.
            Choose between Random Forest (scikit-learn, 300 trees, random_state=7)
            and LightGBM (gradient-boosted trees). Training data is randomly split
            2/3 training, 1/3 testing; R2 and RMSE on the test set are reported.

            Requires:
            - A point vector layer with a numeric field as training data
            - The numeric field to regress against (target)
            - The image to process
            - (optional) A name for the output - predicted image
            """
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.TRAINING_DATA,
                self.tr("Training data"),
                [QgsProcessing.SourceType.VectorPoint],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TARGET_FIELD,
                self.tr("Target field (numeric)"),
                parentLayerParameterName=self.TRAINING_DATA,
                type=QgsProcessingParameterField.DataType.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.SOURCE_IMAGE, self.tr("Image to process"), [QgsProcessing.SourceType.Raster]
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ESTIMATOR, self.tr("Estimator"), self.ESTIMATORS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ESTIMATORS,
                self.tr("Number of trees"),
                QgsProcessingParameterNumber.Type.Integer,
                300,
                False,
                1,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.PREDICTED_IMAGE, self.tr("Predicted"))
        )

    def prepareAlgorithm(self, parameters, context, feedback):
        estimator = self.parameterAsEnum(parameters, self.ESTIMATOR, context)
        if not HAS_SKLEARN_DEPENDENCY:
            feedback.reportError(
                self.tr(
                    "The scikit-learn python dependency is missing, please run pip install scikit-learn"
                )
            )
            return False
        if estimator == 1 and not HAS_LIGHTGBM_DEPENDENCY:
            feedback.reportError(
                self.tr("The lightgbm python dependency is missing, please run pip install lightgbm")
            )
            return False
        return True

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.TRAINING_DATA, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.TRAINING_DATA))
        field_name = self.parameterAsString(parameters, self.TARGET_FIELD, context)
        sourceImage = self.parameterAsRasterLayer(parameters, self.SOURCE_IMAGE, context)
        if sourceImage is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.SOURCE_IMAGE))
        estimator_index = self.parameterAsEnum(parameters, self.ESTIMATOR, context)
        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)

        extent = sourceImage.extent()
        xmin = extent.xMinimum()
        ymax = extent.yMaximum()
        crs = sourceImage.crs()

        provider = sourceImage.dataProvider()
        num_bands = provider.bandCount()

        target_field_index = source.fields().indexFromName(field_name)
        if target_field_index < 0:
            raise QgsProcessingException(f"No attribute named '{field_name}' in layer")

        transform = QgsCoordinateTransform(
            source.sourceCrs(), sourceImage.crs(), context.project()
        )

        feature_list = []
        for feature in source.getFeatures():
            results = provider.identify(
                transform.transform(feature.geometry().asPoint()),
                QgsRaster.IdentifyFormat.IdentifyFormatValue,
            )
            if results.isValid():
                target_value = float(feature.attributes()[target_field_index])
                values = [results.results()[band] for band in range(1, num_bands + 1)]
                values.append(target_value)
                feature_list.append(values)
            else:
                feedback.pushError(
                    f"Could not identify raster values at point {feature.geometry().asWkt()}"
                )

        feature_array = np.array(feature_list)

        pixelSizeX = sourceImage.rasterUnitsPerPixelX()
        pixelSizeY = sourceImage.rasterUnitsPerPixelY()

        dataArray = sourceImage.as_numpy()
        bands, height, width = dataArray.shape
        reshaped_image = dataArray.reshape(bands, height * width).T

        X = feature_array[:, 0:bands]
        y = feature_array[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

        if estimator_index == 0:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=7)
        else:
            model = LGBMRegressor(n_estimators=n_estimators, random_state=7, verbosity=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        feedback.pushInfo(f"Performance: R2={r2:.4f}, RMSE={rmse:.4f}")

        predicted = model.predict(reshaped_image)
        predicted_reshaped = predicted.T.reshape(1, height, width)
        predicted_2d = np.squeeze(predicted_reshaped, axis=0)

        geotransform = (xmin, pixelSizeX, 0, ymax, 0, -pixelSizeY)
        output_file = self.parameterAsOutputLayer(parameters, self.PREDICTED_IMAGE, context)

        driver = gdal.GetDriverByName("GTiff")
        out_raster = driver.Create(output_file, width, height, 1, gdal.GDT_Float32)
        out_raster.SetGeoTransform(geotransform)
        out_raster.SetProjection(crs.toWkt())

        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(predicted_2d)
        out_band.FlushCache()

        return {self.PREDICTED_IMAGE: output_file}
