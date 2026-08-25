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

import os
import random

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtCore import QCoreApplication

HAS_PIL_DEPENDENCY = True
try:
    from PIL import Image
except ImportError:
    HAS_PIL_DEPENDENCY = False


class YoloDataPrepProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    Tiles a raster on a fixed pixel grid and writes an Ultralytics YOLO-ready
    dataset (images/labels, train/val split) from a polygon label layer.
    Only tiles that intersect at least one polygon are kept; polygons are
    converted to axis-aligned bounding boxes.
    """

    INPUT = "INPUT"
    POLYGONS = "POLYGONS"
    CLASS_FIELD = "CLASS_FIELD"
    TILE_SIZE = "TILE_SIZE"
    OVERLAP = "OVERLAP"
    TRAIN_SPLIT = "TRAIN_SPLIT"
    OUTPUT_DIR = "OUTPUT_DIR"

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return YoloDataPrepProcessingAlgorithm()

    def name(self):
        return "yolodataprep"

    def displayName(self):
        return self.tr("YOLO Data Preparation")

    def group(self):
        return self.tr("Data Preparation")

    def groupId(self):
        return "datapreparation"

    def shortHelpString(self):
        return self.tr(
            """
            Prepares an Ultralytics YOLO training dataset from a polygon label
            layer and a raster. The raster is tiled on a fixed pixel grid
            (default 640x640); only tiles intersecting at least one polygon
            are kept. Polygons are converted to axis-aligned bounding boxes
            in YOLO normalised format. Class labels come from a text field
            on the polygon layer and are auto-mapped to integer ids.

            Output structure (Ultralytics-compatible):
            images/train, images/val, labels/train, labels/val, classes.txt, data.yaml

            Assumes a north-up raster; the first band (or first 3, treated
            as RGB) are stretched to 8-bit for the output PNG tiles.
            """
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, self.tr("Input raster")))
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.POLYGONS,
                self.tr("Polygon labels"),
                [QgsProcessing.SourceType.VectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.CLASS_FIELD,
                self.tr("Class field (text)"),
                parentLayerParameterName=self.POLYGONS,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TILE_SIZE,
                self.tr("Tile size (pixels)"),
                QgsProcessingParameterNumber.Type.Integer,
                640,
                False,
                32,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OVERLAP,
                self.tr("Overlap between adjacent tiles (pixels)"),
                QgsProcessingParameterNumber.Type.Integer,
                0,
                False,
                0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TRAIN_SPLIT,
                self.tr("Train split fraction"),
                QgsProcessingParameterNumber.Type.Double,
                0.8,
                False,
                0.1,
                0.95,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_DIR, self.tr("Output dataset folder"))
        )

    def prepareAlgorithm(self, parameters, context, feedback):
        if not HAS_PIL_DEPENDENCY:
            feedback.reportError(self.tr("The Pillow python dependency is missing, please run pip install Pillow"))
            return False
        return True

    def processAlgorithm(self, parameters, context, feedback):
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster_layer is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        polygons_source = self.parameterAsSource(parameters, self.POLYGONS, context)
        if polygons_source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.POLYGONS))
        class_field = self.parameterAsString(parameters, self.CLASS_FIELD, context)
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        if overlap >= tile_size:
            raise QgsProcessingException("Overlap must be smaller than the tile size")
        train_split = self.parameterAsDouble(parameters, self.TRAIN_SPLIT, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        class_field_index = polygons_source.fields().indexFromName(class_field)
        if class_field_index < 0:
            raise QgsProcessingException(f"No attribute named '{class_field}' in layer")

        # Output folder structure
        for split in ("train", "val"):
            os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

        ds = gdal.Open(raster_layer.source())
        if ds is None:
            raise QgsProcessingException(f"Could not open raster with GDAL: {raster_layer.source()}")

        gt = ds.GetGeoTransform()
        px_w, px_h = gt[1], gt[5]  # px_h is negative for north-up rasters
        raster_w, raster_h = ds.RasterXSize, ds.RasterYSize
        band_count = min(ds.RasterCount, 3)

        # Simple global min/max per band for an 8-bit stretch
        band_ranges = []
        for b in range(1, band_count + 1):
            band = ds.GetRasterBand(b)
            bmin, bmax = band.ComputeRasterMinMax(False)
            band_ranges.append((bmin, bmax if bmax > bmin else bmin + 1))

        # Reproject polygon bboxes into raster CRS and build class map
        transform = QgsCoordinateTransform(
            polygons_source.sourceCrs(), raster_layer.crs(), context.project()
        )
        class_map = {}
        polygon_boxes = []  # (xmin, ymin, xmax, ymax, class_id) in map coords
        for feature in polygons_source.getFeatures():
            geom = feature.geometry()
            geom.transform(transform)
            bbox = geom.boundingBox()
            class_name = str(feature.attributes()[class_field_index])
            if class_name not in class_map:
                class_map[class_name] = len(class_map)
            polygon_boxes.append(
                (bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum(), class_map[class_name])
            )

        if not polygon_boxes:
            raise QgsProcessingException("No polygon features found in the label layer")

        def compute_offsets(total, tile, stride):
            if total <= tile:
                return [0]
            offsets = list(range(0, total - tile + 1, stride))
            if offsets[-1] != total - tile:
                offsets.append(total - tile)
            return offsets

        stride = tile_size - overlap
        x_offsets = compute_offsets(raster_w, tile_size, stride)
        y_offsets = compute_offsets(raster_h, tile_size, stride)
        cols, rows = len(x_offsets), len(y_offsets)
        total_tiles = cols * rows
        feedback.pushInfo(f"Scanning {total_tiles} candidate tiles ({cols} x {rows}, stride={stride}px)")

        random.seed(7)
        tiles_written = 0
        tile_index = 0
        for row in range(rows):
            if feedback.isCanceled():
                break
            for col in range(cols):
                if feedback.isCanceled():
                    break
                tile_index += 1
                feedback.setProgress(tile_index / total_tiles * 100)

                x_off = x_offsets[col]
                y_off = y_offsets[row]
                win_w = min(tile_size, raster_w - x_off)
                win_h = min(tile_size, raster_h - y_off)

                tile_xmin = gt[0] + x_off * px_w
                tile_xmax = gt[0] + (x_off + tile_size) * px_w
                tile_ymax = gt[3] + y_off * px_h
                tile_ymin = gt[3] + (y_off + tile_size) * px_h

                # Keep only boxes that intersect this tile
                labels = []
                for xmin, ymin, xmax, ymax, class_id in polygon_boxes:
                    if xmax < tile_xmin or xmin > tile_xmax or ymax < tile_ymin or ymin > tile_ymax:
                        continue
                    # Clip to tile extent
                    cxmin, cxmax = max(xmin, tile_xmin), min(xmax, tile_xmax)
                    cymin, cymax = max(ymin, tile_ymin), min(ymax, tile_ymax)

                    # Map coords -> pixel coords within the tile -> YOLO normalised
                    px0 = (cxmin - tile_xmin) / px_w
                    px1 = (cxmax - tile_xmin) / px_w
                    py0 = (cymax - tile_ymax) / px_h
                    py1 = (cymin - tile_ymax) / px_h

                    x_center = ((px0 + px1) / 2) / tile_size
                    y_center = ((py0 + py1) / 2) / tile_size
                    box_w = abs(px1 - px0) / tile_size
                    box_h = abs(py1 - py0) / tile_size
                    labels.append((class_id, x_center, y_center, box_w, box_h))

                if not labels:
                    continue

                # Read and stretch the tile to 8-bit, padding to tile_size x tile_size
                tile_arr = np.zeros((tile_size, tile_size, band_count), dtype=np.uint8)
                for b in range(1, band_count + 1):
                    band = ds.GetRasterBand(b)
                    data = band.ReadAsArray(x_off, y_off, win_w, win_h).astype(np.float32)
                    bmin, bmax = band_ranges[b - 1]
                    stretched = np.clip((data - bmin) / (bmax - bmin) * 255.0, 0, 255).astype(np.uint8)
                    tile_arr[:win_h, :win_w, b - 1] = stretched

                split = "train" if random.random() < train_split else "val"
                tile_name = f"tile_{row}_{col}"
                img = Image.fromarray(tile_arr.squeeze() if band_count == 1 else tile_arr)
                img.save(os.path.join(output_dir, "images", split, f"{tile_name}.png"))
                with open(os.path.join(output_dir, "labels", split, f"{tile_name}.txt"), "w") as f:
                    for class_id, xc, yc, w, h in labels:
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                tiles_written += 1

        # classes.txt and data.yaml for Ultralytics
        sorted_classes = sorted(class_map.items(), key=lambda kv: kv[1])
        with open(os.path.join(output_dir, "classes.txt"), "w") as f:
            for name, class_id in sorted_classes:
                f.write(f"{class_id} {name}\n")

        names_yaml = ", ".join(f"{cid}: {name}" for name, cid in sorted_classes)
        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            f.write(f"path: {output_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(sorted_classes)}\n")
            f.write(f"names: {{{names_yaml}}}\n")

        feedback.pushInfo(f"Wrote {tiles_written} tiles across {len(class_map)} classes to {output_dir}")

        return {self.OUTPUT_DIR: output_dir}
