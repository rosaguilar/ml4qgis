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

import json
import os
import random

import numpy as np
from osgeo import gdal, ogr, osr
from qgis.core import (
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
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
     Create patches from a raster on a fixed pixel grid and generates a training dataset from
       a polygon label layer: image patches, optional binary and multiclass mask
       patches (segmentation), and optional bounding-box labels (YOLO txt or
       COCO json - object detection).
       Patches are split into random train/val/test splits.
       Class ids are auto-assigned from the distinct values in the
       chosen class field.
    """

    INPUT = "INPUT"
    POLYGONS = "POLYGONS"
    CLASS_FIELD = "CLASS_FIELD"
    TILE_SIZE = "TILE_SIZE"
    OVERLAP = "OVERLAP"
    TRAIN_SPLIT = "TRAIN_SPLIT"
    TEST_SPLIT = "TEST_SPLIT"
    OUTPUT_MASKS = "OUTPUT_MASKS"
    LABEL_FORMAT = "LABEL_FORMAT"
    KEEP_EMPTY_TILES = "KEEP_EMPTY_TILES"
    SKIP_NODATA = "SKIP_NODATA"
    IMAGE_FORMAT = "IMAGE_FORMAT"
    OUTPUT_DIR = "OUTPUT_DIR"

    LABEL_FORMATS = ["YOLO (txt)", "COCO (json)", "None"]
    IMAGE_FORMATS = ["PNG (RGB, 8-bit stretch)", "GeoTIFF (all bands, float32 reflectance)"]

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return YoloDataPrepProcessingAlgorithm()

    def name(self):
        return "yolodataprep"

    def displayName(self):
        return self.tr("Image + Mask + Label Data Preparation")

    def group(self):
        return self.tr("Data Preparation")

    def groupId(self):
        return "datapreparation"

    def shortHelpString(self):
        return self.tr(
            """
             Creates raster patches on a fixed pixel grid (default 640x640) and
             generates a training dataset from a polygon label layer, split
             into train/val/test.
 
             Output structure:
             - images/train, images/val, images/test 
             - masks_binary/, masks_multiclass/ - train/val/test (if "Output masks" is on;
             - Binary mask in single-band GeoTIFF, 
             - Multiclass mask in multi-band GeoT (pixel value = class id, 0 = background),
             georeferenced to match the image tile for visual alignment checks)
             YOLO: labels/train, labels/val, labels/test (.txt), classes.txt, data.yaml
             COCO: annotations_train.json, annotations_val.json,
             annotations_test.json (bbox in absolute pixel [x,y,w,h])
 
             Class ids are assigned automatically to each distinct value of
             the class field, in the order encountered (starting at 1; 0 is
             background). The same ids are used in both the masks and the
             bounding-box labels.
 
             "Keep empty tiles" controls whether tiles with no polygon
             coverage are kept (useful as background examples for
             segmentation) or dropped (usual choice for detection-only use).
            """               )

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
                self.tr("Class field"),
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
                0.7,
                False,
                0.1,
                0.95,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TEST_SPLIT,
                self.tr("Test split fraction (remainder after train goes to val)"),
                QgsProcessingParameterNumber.Type.Double,
                0.15,
                False,
                0.0,
                0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.OUTPUT_MASKS, self.tr("Output masks (binary + multiclass, georeferenced GeoTIFF)"), defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LABEL_FORMAT, self.tr("Bounding-box label format"), self.LABEL_FORMATS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.KEEP_EMPTY_TILES,
                self.tr("Keep tiles with no polygon coverage (background examples)"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SKIP_NODATA,
                self.tr("Exclude NoData/black-padding pixels from tiles and masks"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.IMAGE_FORMAT, self.tr("Image format"), self.IMAGE_FORMATS, defaultValue=0
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
        test_split = self.parameterAsDouble(parameters, self.TEST_SPLIT, context)
        if train_split + test_split >= 1.0:
            raise QgsProcessingException("Train split + test split must leave room for a val split (sum < 1)")
        val_split = 1.0 - train_split - test_split
        feedback.pushInfo(f"Split fractions: train={train_split}, val={val_split:.3f}, test={test_split}")
        output_masks = self.parameterAsBoolean(parameters, self.OUTPUT_MASKS, context)
        label_format = self.parameterAsEnum(parameters, self.LABEL_FORMAT, context)
        keep_empty_tiles = self.parameterAsBoolean(parameters, self.KEEP_EMPTY_TILES, context)
        skip_nodata = self.parameterAsBoolean(parameters, self.SKIP_NODATA, context)
        image_format = self.parameterAsEnum(parameters, self.IMAGE_FORMAT, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        class_field_index = polygons_source.fields().indexFromName(class_field)
        if class_field_index < 0:
            raise QgsProcessingException(f"No attribute named '{class_field}' in layer")

        # Output folder structure
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
            if output_masks:
                os.makedirs(os.path.join(output_dir, "masks_binary", split), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "masks_multiclass", split), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "masks_valid", split), exist_ok=True)
            if label_format == 0:  # YOLO
                os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

        ds = gdal.Open(raster_layer.source())
        if ds is None:
            raise QgsProcessingException(f"Could not open raster with GDAL: {raster_layer.source()}")

        gt = ds.GetGeoTransform()
        px_w, px_h = gt[1], gt[5]  # px_h is negative for north-up rasters
        raster_w, raster_h = ds.RasterXSize, ds.RasterYSize

        # Prefer QGIS's own resolved CRS over the raw file's embedded
        # projection - a file can have a valid geotransform but a missing
        # or malformed embedded CRS tag, and raster_layer.crs() is reliable
        # even then.
        tile_wkt = raster_layer.crs().toWkt() if raster_layer.crs().isValid() else ds.GetProjection()
        if not tile_wkt:
            feedback.pushWarning(
                "No valid CRS found on the input raster or layer - output tiles will be unreferenced."
            )

        if image_format == 0:  # PNG
            band_count = min(ds.RasterCount, 3)
            # Simple global min/max per band for an 8-bit stretch
            band_ranges = []
            for b in range(1, band_count + 1):
                band = ds.GetRasterBand(b)
                bmin, bmax = band.ComputeRasterMinMax(False)
                band_ranges.append((bmin, bmax if bmax > bmin else bmin + 1))
        else:  # GeoTIFF, all bands
            band_count = ds.RasterCount
            band_ranges = None
            feedback.pushInfo(f"Writing all {band_count} bands as float32 reflectance (/10000)")

        band_nodata = [ds.GetRasterBand(b).GetNoDataValue() for b in range(1, band_count + 1)]
        if skip_nodata:
            if any(v is not None for v in band_nodata):
                feedback.pushInfo(f"NoData per band: {band_nodata}")
            else:
                feedback.pushInfo("No NoData value declared on any band - falling back to "
                                   "'all bands exactly 0' as the padding heuristic")

        # Reproject polygons into raster CRS; assign class ids to each
        # distinct class-field value in the order encountered (no hardcoded
        # vocabulary). Used consistently for both bbox labels and masks.
        transform = QgsCoordinateTransform(
            polygons_source.sourceCrs(), raster_layer.crs(), context.project()
        )
        class_map = {}
        polygon_boxes = []  # (xmin, ymin, xmax, ymax, class_id) in map coords

        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        mem_ds = None
        mem_layer = None
        if output_masks:
            mem_ds = ogr.GetDriverByName("Memory").CreateDataSource("mem")
            mem_layer = mem_ds.CreateLayer("veg", srs, ogr.wkbMultiPolygon)
            mem_layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))

        for feature in polygons_source.getFeatures():
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                continue
            geom.transform(transform)
            class_name = str(feature.attributes()[class_field_index]).strip()
            if class_name not in class_map:
                class_map[class_name] = len(class_map) + 1
            class_id = class_map[class_name]

            bbox = geom.boundingBox()
            polygon_boxes.append(
                (bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum(), class_id)
            )

            if output_masks:
                ogr_geom = ogr.CreateGeometryFromWkt(geom.asWkt())
                if ogr_geom is not None and not ogr_geom.IsEmpty():
                    ogr_feat = ogr.Feature(mem_layer.GetLayerDefn())
                    ogr_feat.SetGeometry(ogr_geom)
                    ogr_feat.SetField("class_id", class_id)
                    mem_layer.CreateFeature(ogr_feat)

        if not polygon_boxes:
            raise QgsProcessingException("No polygon features found in the label layer")
        feedback.pushInfo(f"{len(class_map)} classes from '{class_field}': {class_map}")

        # Intersection of the image's extent and the polygon layer's extent -
        # used below only to SKIP individual tiles that fall outside it, not
        # to shift where tile numbering starts. Tile (0,0) always stays the
        # image's own top-left, regardless of the polygon layer's coverage.
        image_xmin, image_xmax = gt[0], gt[0] + raster_w * px_w
        image_ymax, image_ymin = gt[3], gt[3] + raster_h * px_h  # px_h negative
        veg_extent = transform.transformBoundingBox(polygons_source.sourceExtent())
        clip_xmin = max(image_xmin, veg_extent.xMinimum())
        clip_xmax = min(image_xmax, veg_extent.xMaximum())
        clip_ymin = max(image_ymin, veg_extent.yMinimum())
        clip_ymax = min(image_ymax, veg_extent.yMaximum())
        if clip_xmin >= clip_xmax or clip_ymin >= clip_ymax:
            raise QgsProcessingException("The image and the polygon layer do not overlap")
        feedback.pushInfo(
            f"Image/polygon intersection: {clip_xmin:.1f},{clip_ymin:.1f} : {clip_xmax:.1f},{clip_ymax:.1f} "
            f"(tiles entirely outside this are skipped; grid still starts at the image's own top-left)"
        )

        # Rasterize binary + multiclass masks once, over the full raster grid
        # (in-memory, no disk write) - patches are sliced from these below,
        # instead of re-rasterizing per tile.
        binary_full = multiclass_full = None
        if output_masks:
            mem_drv = gdal.GetDriverByName("MEM")
            binary_ds = mem_drv.Create("", raster_w, raster_h, 1, gdal.GDT_Byte)
            binary_ds.SetGeoTransform(gt)
            binary_ds.SetProjection(ds.GetProjection())
            gdal.RasterizeLayer(binary_ds, [1], mem_layer, burn_values=[1])
            binary_full = binary_ds.GetRasterBand(1).ReadAsArray()

            multiclass_ds = mem_drv.Create("", raster_w, raster_h, 1, gdal.GDT_Byte)
            multiclass_ds.SetGeoTransform(gt)
            multiclass_ds.SetProjection(ds.GetProjection())
            gdal.RasterizeLayer(multiclass_ds, [1], mem_layer, options=["ATTRIBUTE=class_id"])
            multiclass_full = multiclass_ds.GetRasterBand(1).ReadAsArray()

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

        def write_geotiff_tile(path, array, tile_gt, dtype, log_transform=False):
            """array: (bands, H, W) or (H, W) for single-band. Raises on failure
            instead of silently writing an unreferenced file."""
            if array.ndim == 2:
                array = array[np.newaxis, ...]
            n_bands = array.shape[0]
            out_ds = gdal.GetDriverByName("GTiff").Create(
                path, tile_size, tile_size, n_bands, dtype, options=["COMPRESS=DEFLATE"]
            )
            if out_ds is None:
                raise QgsProcessingException(f"GDAL failed to create output file: {path}")
            if out_ds.SetGeoTransform(tile_gt) != 0:
                raise QgsProcessingException(f"GDAL SetGeoTransform failed for {path} (transform={tile_gt})")
            if tile_wkt and out_ds.SetProjection(tile_wkt) != 0:
                raise QgsProcessingException(f"GDAL SetProjection failed for {path}")
            for b in range(1, n_bands + 1):
                out_ds.GetRasterBand(b).WriteArray(array[b - 1])
            out_ds.FlushCache()
            if log_transform:
                feedback.pushInfo(f"First tile geotransform written: {out_ds.GetGeoTransform()}")
            out_ds = None

        random.seed(7)
        tiles_written = 0
        tile_index = 0
        coco = {
            "train": {"images": [], "annotations": []},
            "val": {"images": [], "annotations": []},
            "test": {"images": [], "annotations": []},
        }
        next_image_id = 1
        next_annotation_id = 1

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

                # Skip tiles entirely outside the image/polygon intersection
                # (grid position/numbering is unaffected - only this tile is skipped)
                if tile_xmax < clip_xmin or tile_xmin > clip_xmax or tile_ymax < clip_ymin or tile_ymin > clip_ymax:
                    continue

                # Keep only boxes that intersect this tile
                labels = []
                for xmin, ymin, xmax, ymax, class_id in polygon_boxes:
                    if xmax < tile_xmin or xmin > tile_xmax or ymax < tile_ymin or ymin > tile_ymax:
                        continue
                    cxmin, cxmax = max(xmin, tile_xmin), min(xmax, tile_xmax)
                    cymin, cymax = max(ymin, tile_ymin), min(ymax, tile_ymax)

                    px0 = (cxmin - tile_xmin) / px_w
                    px1 = (cxmax - tile_xmin) / px_w
                    py0 = (cymax - tile_ymax) / px_h
                    py1 = (cymin - tile_ymax) / px_h

                    x_center = ((px0 + px1) / 2) / tile_size
                    y_center = ((py0 + py1) / 2) / tile_size
                    box_w = abs(px1 - px0) / tile_size
                    box_h = abs(py1 - py0) / tile_size
                    px_xmin = min(px0, px1)
                    px_ymin = min(py0, py1)
                    labels.append(
                        (class_id, x_center, y_center, box_w, box_h, px_xmin, px_ymin, abs(px1 - px0), abs(py1 - py0))
                    )

                if not labels and not keep_empty_tiles:
                    continue

                r = random.random()
                split = "train" if r < train_split else ("val" if r < train_split + val_split else "test")
                tile_name = f"tile_{row}_{col}"

                raw_bands = [
                    ds.GetRasterBand(b).ReadAsArray(x_off, y_off, win_w, win_h).astype(np.float32)
                    for b in range(1, band_count + 1)
                ]

                valid_mask = np.zeros((tile_size, tile_size), dtype=bool)
                valid_mask[:win_h, :win_w] = True
                if skip_nodata:
                    has_declared_nodata = any(v is not None for v in band_nodata)
                    for b in range(band_count):
                        data = raw_bands[b]
                        nodata = band_nodata[b]
                        if nodata is not None:
                            valid_mask[:win_h, :win_w] &= data != nodata
                        valid_mask[:win_h, :win_w] &= ~np.isnan(data)
                    if not has_declared_nodata:
                        all_zero = np.all([raw_bands[b] == 0 for b in range(band_count)], axis=0)
                        valid_mask[:win_h, :win_w] &= ~all_zero

                    if not valid_mask.any():
                        continue  # tile is entirely NoData/padding

                if image_format == 0:  # PNG, stretched to 8-bit
                    tile_arr = np.zeros((tile_size, tile_size, band_count), dtype=np.uint8)
                    for b in range(1, band_count + 1):
                        bmin, bmax = band_ranges[b - 1]
                        stretched = np.clip((raw_bands[b - 1] - bmin) / (bmax - bmin) * 255.0, 0, 255).astype(np.uint8)
                        tile_arr[:win_h, :win_w, b - 1] = stretched
                    image_filename = f"{tile_name}.png"
                    img = Image.fromarray(tile_arr.squeeze() if band_count == 1 else tile_arr)
                    img.save(os.path.join(output_dir, "images", split, image_filename))
                else:  # GeoTIFF, all bands, raw reflectance (/10000)
                    tile_arr = np.zeros((band_count, tile_size, tile_size), dtype=np.float32)
                    for b in range(1, band_count + 1):
                        tile_arr[b - 1, :win_h, :win_w] = raw_bands[b - 1] / 10000.0
                    image_filename = f"{tile_name}.tif"
                    tile_gt = (tile_xmin, px_w, 0, tile_ymax, 0, px_h)
                    write_geotiff_tile(
                        os.path.join(output_dir, "images", split, image_filename),
                        tile_arr, tile_gt, gdal.GDT_Float32, log_transform=(row == 0 and col == 0),
                    )

                if output_masks:
                    tile_gt = (tile_xmin, px_w, 0, tile_ymax, 0, px_h)

                    binary_patch = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    binary_patch[:win_h, :win_w] = binary_full[y_off : y_off + win_h, x_off : x_off + win_w]
                    binary_patch[~valid_mask] = 0
                    write_geotiff_tile(
                        os.path.join(output_dir, "masks_binary", split, f"{tile_name}.tif"),
                        binary_patch, tile_gt, gdal.GDT_Byte,
                    )

                    # 255 marks "no image data here" (matches PyTorch's
                    # CrossEntropyLoss ignore_index=255 convention) - different
                    # from 0, which means "confirmed not vegetation".
                    multiclass_patch = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    multiclass_patch[:win_h, :win_w] = multiclass_full[
                        y_off : y_off + win_h, x_off : x_off + win_w
                    ]
                    multiclass_patch[~valid_mask] = 255
                    write_geotiff_tile(
                        os.path.join(output_dir, "masks_multiclass", split, f"{tile_name}.tif"),
                        multiclass_patch, tile_gt, gdal.GDT_Byte,
                    )

                    # Binary mask has no room for a third "no data" state
                    # (it's strictly 0/1), so validity is a separate mask:
                    # 1 = real image data, 0 = padding/NoData - multiply this
                    # into a per-pixel BCE loss before reduction.
                    write_geotiff_tile(
                        os.path.join(output_dir, "masks_valid", split, f"{tile_name}.tif"),
                        valid_mask.astype(np.uint8), tile_gt, gdal.GDT_Byte,
                    )

                if label_format == 0:  # YOLO
                    with open(os.path.join(output_dir, "labels", split, f"{tile_name}.txt"), "w") as f:
                        for class_id, xc, yc, w, h, *_ in labels:
                            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                elif label_format == 1:  # COCO
                    image_id = next_image_id
                    next_image_id += 1
                    coco[split]["images"].append(
                        {"id": image_id, "file_name": image_filename, "width": tile_size, "height": tile_size}
                    )
                    for class_id, _, _, _, _, px_xmin, px_ymin, px_w_, px_h_ in labels:
                        coco[split]["annotations"].append(
                            {
                                "id": next_annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "bbox": [round(px_xmin, 2), round(px_ymin, 2), round(px_w_, 2), round(px_h_, 2)],
                                "area": round(px_w_ * px_h_, 2),
                                "iscrowd": 0,
                            }
                        )
                        next_annotation_id += 1

                tiles_written += 1

        sorted_classes = sorted(class_map.items(), key=lambda kv: kv[1])

        if output_masks or label_format == 0:
            with open(os.path.join(output_dir, "classes.txt"), "w") as f:
                for name, class_id in sorted_classes:
                    f.write(f"{class_id} {name}\n")

        if label_format == 0:  # YOLO
            names_yaml = ", ".join(f"{cid}: {name}" for name, cid in sorted_classes)
            with open(os.path.join(output_dir, "data.yaml"), "w") as f:
                f.write(f"path: {output_dir}\n")
                f.write("train: images/train\n")
                f.write("val: images/val\n")
                f.write("test: images/test\n")
                f.write(f"nc: {len(sorted_classes)}\n")
                f.write(f"names: {{{names_yaml}}}\n")
        elif label_format == 1:  # COCO
            categories = [{"id": cid, "name": name} for name, cid in sorted_classes]
            for split in ("train", "val", "test"):
                coco_out = {
                    "images": coco[split]["images"],
                    "annotations": coco[split]["annotations"],
                    "categories": categories,
                }
                with open(os.path.join(output_dir, f"annotations_{split}.json"), "w") as f:
                    json.dump(coco_out, f)

        feedback.pushInfo(f"Wrote {tiles_written} tiles across {len(class_map)} classes to {output_dir}")

        return {self.OUTPUT_DIR: output_dir}
