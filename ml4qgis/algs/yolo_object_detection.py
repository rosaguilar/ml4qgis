"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
* Author: Mathieu Pellerin email: mathieu@opengis.ch                      *
***************************************************************************
"""

import math
import os

import numpy as np
import pandas as pd
from qgis.core import (
    QgsFeature,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMapRendererSequentialJob,
    QgsMapSettings,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsRectangle,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication, QSize, QVariant
from qgis.PyQt.QtGui import QImage

HAS_YOLO_DEPENDENCY = True
try:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    from ultralytics import YOLO
except ImportError:
    HAS_YOLO_DEPENDENCY = False


def generate_wkt(extent, mmup, xyxy):
    xmin, ymin, xmax, ymax = xyxy
    xmin = extent.xMinimum() + xmin * mmup
    ymin = extent.yMaximum() - ymin * mmup
    xmax = extent.xMinimum() + xmax * mmup
    ymax = extent.yMaximum() - ymax * mmup
    return f"POLYGON(({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))"


def get_boxes(boxes):
    class_ids = []
    confs = []
    xyxys = []
    xyxyns = []
    for box in boxes:
        conf = box.conf[0].item()
        confs.append(conf)
        cls = int(box.cls[0].item())
        class_ids.append(cls)
        xyxys.append(box.xyxy[0].numpy())
        xyxyns.append(box.xyxyn[0].numpy())

    # make a data frame
    data = {
        "class_ids": class_ids,
        "confs": confs,
        "xyxy": xyxys,
        "xyxyn": xyxyns,
    }
    df = pd.DataFrame.from_dict(data)
    return df


class YoloObjectDetectionProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an algorithm that uses YOLO trained models to conduct
    object detection against a given raster layer.
    """

    INPUT = "INPUT"
    MUPP = "MUPP"
    AREAS = "AREAS"
    MODEL = "MODEL"
    MINIMUM_CONFIDENCE = "MINIMUM_CONFIDENCE"
    OUTPUT = "OUTPUT"

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return YoloObjectDetectionProcessingAlgorithm()

    def name(self):
        return "yoloobjectdetection"

    def groupId(self):
        return "siemens"

    def displayName(self):
        return self.tr("YOLO Object Detection")

    def group(self):
        return self.tr("Object Detection")

    def shortHelpString(self):
        return self.tr(
            'Detect objects using trained YOLO (You Only Look Once) model weight files (.pt). These models offer accurate and speedy real-time object detection.\n\nTo know how to train YOLO models, follow <a href="https://docs.ultralytics.com/quickstart/">this quickstart guide</a>.'
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.MODEL, self.tr("YOLO model file")))

        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, self.tr("Input raster")))

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MUPP,
                self.tr("Map units per pixel (MUPP)"),
                QgsProcessingParameterNumber.Type.Double,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MINIMUM_CONFIDENCE,
                self.tr("Minimum confidence threshold (value between 0.0 to 1.0)"),
                QgsProcessingParameterNumber.Type.Double,
                0.0,
                False,
                0.0,
                1.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.AREAS,
                self.tr("Areas to conduct object detection"),
                [QgsProcessing.SourceType.VectorPolygon],
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("Detected objects"))
        )

    def prepareAlgorithm(self, parameters, context, feedback):
        if not HAS_YOLO_DEPENDENCY:
            feedback.reportError(
                self.tr(
                    "The ultralytics YOLO python dependency is missing, please run pip install ultralytics"
                )
            )
            return False

        return True

    def processAlgorithm(self, parameters, context, feedback):
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)

        minimum_confidence = self.parameterAsDouble(parameters, self.MINIMUM_CONFIDENCE, context)
        mupp = self.parameterAsDouble(parameters, self.MUPP, context)

        areas_source = self.parameterAsSource(parameters, self.AREAS, context)

        model_file = self.parameterAsString(parameters, self.MODEL, context)

        output_fields = QgsFields()
        output_fields.append(QgsField("names", QVariant.String))
        output_fields.append(QgsField("confidence", QVariant.Double))
        output_sink, output_filename = self.parameterAsSink(
            parameters, self.OUTPUT, context, output_fields, QgsWkbTypes.Polygon, input_layer.crs()
        )

        model = YOLO(model_file)

        tile_width_pixel = 640
        output_size = QSize(tile_width_pixel, tile_width_pixel)
        tile_width_mu = tile_width_pixel * mupp

        ms = QgsMapSettings()
        ms.setOutputDpi(96)
        ms.setOutputSize(output_size)
        ms.setDestinationCrs(input_layer.crs())
        ms.setLayers([input_layer])

        request = QgsFeatureRequest()
        request.setDestinationCrs(ms.destinationCrs(), context.transformContext())

        feedback.pushInfo("Calculating number of tiles to be processed")

        raster_extent = input_layer.extent()
        tiles_total = 0
        it = areas_source.getFeatures(request)
        for feature in it:
            if feedback.isCanceled():
                break

            feature_bbox = feature.geometry().boundingBox()
            intersected_bbox = feature_bbox.intersect(raster_extent)
            if intersected_bbox.isEmpty():
                continue

            dfs = []
            (columns, rows, _, _) = self.calculateColumnsRowsStarts(
                intersected_bbox, tile_width_mu
            )
            tiles_total = tiles_total + (columns * rows)

        feedback.pushInfo(f"{tiles_total} tiles will be processed")

        tiles_current = 0
        it = areas_source.getFeatures(request)
        for feature in it:
            if feedback.isCanceled():
                break

            feature_bbox = feature.geometry().boundingBox()
            intersected_bbox = feature_bbox.intersect(raster_extent)
            if intersected_bbox.isEmpty():
                feedback.pushInfo(f"Skipping feature ID {feature.id()}, not covering raster area")
                continue

            feedback.pushInfo(f"Looking for objects around area feature ID {feature.id()}")

            dfs = []
            (columns, rows, start_x, start_y) = self.calculateColumnsRowsStarts(
                intersected_bbox, tile_width_mu
            )
            for row in range(rows):
                if feedback.isCanceled():
                    break

                for column in range(columns):
                    if feedback.isCanceled():
                        break

                    tiles_current = tiles_current + 1
                    feedback.setProgress(tiles_current / tiles_total * 100)

                    extent = QgsRectangle(
                        start_x + row * tile_width_mu,
                        start_y + column * tile_width_mu,
                        start_x + (row + 1) * tile_width_mu,
                        start_y + (column + 1) * tile_width_mu,
                    )
                    ms.setExtent(extent)

                    job = QgsMapRendererSequentialJob(ms)
                    job.start()
                    job.waitForFinished()

                    img = job.renderedImage().convertToFormat(QImage.Format_BGR888)
                    ptr = img.constBits()
                    ptr.setsize(tile_width_pixel * tile_width_pixel * 3)
                    arr = np.frombuffer(ptr, np.uint8).reshape(
                        (tile_width_pixel, tile_width_pixel, 3)
                    )

                    results = model.predict(
                        device="cpu",
                        source=[arr],
                        imgsz=tile_width_pixel,
                        show=False,
                        save_txt=False,
                        max_det=3,
                        save_conf=False,
                        verbose=False,
                    )
                    for res in results:
                        if len(res.boxes) > 0:
                            df = get_boxes(res.boxes)
                            if len(df["xyxy"]) > 0:
                                df["names"] = res.names
                                df["extent"] = ms.extent()
                                df["mmup"] = ms.mapUnitsPerPixel()
                                dfs.append(df)

            if len(dfs) > 0:
                # Concat data frames
                df_all = pd.concat(dfs)
                df_all["wkt"] = df_all.apply(
                    lambda x: generate_wkt(x.extent, x.mmup, x.xyxy), axis=1
                )
                for index, row in df_all.iterrows():
                    if row["confs"] >= minimum_confidence:
                        f = QgsFeature(output_fields)
                        f.setAttribute("names", row["names"])
                        f.setAttribute("confidence", row["confs"])
                        f.setGeometry(QgsGeometry.fromWkt(row["wkt"]))
                        output_sink.addFeature(f, QgsFeatureSink.Flag.FastInsert)

        output_sink.flushBuffer()
        del output_sink

        return {self.OUTPUT: output_filename}

    def calculateColumnsRowsStarts(self, bounding_box, tile_width_mu):
        columns = math.ceil(bounding_box.width() / tile_width_mu)
        start_x = bounding_box.xMinimum()
        if bounding_box.width() % tile_width_mu > 0:
            start_x -= (bounding_box.width() % tile_width_mu) / 2

        rows = math.ceil(bounding_box.height() / tile_width_mu)
        start_y = bounding_box.yMinimum()
        if bounding_box.height() % tile_width_mu > 0:
            start_y -= (bounding_box.height() % tile_width_mu) / 2

        return (columns, rows, start_x, start_y)
