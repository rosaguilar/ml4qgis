# -----------------------------------------------------------
# Copyright (C) 2025 Rosa Aguilar
# -----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# ---------------------------------------------------------------------

from qgis.core import QgsApplication

from .provider import Ml4QgisProcessingProvider


def classFactory(iface):
    return Ml4QgisPlugin(iface)


class Ml4QgisPlugin:
    def __init__(self, iface):
        self.iface = iface

    def initGui(self):
        self.processing_provider = Ml4QgisProcessingProvider()
        print(self.processing_provider.svgIconPath())
        QgsApplication.processingRegistry().addProvider(self.processing_provider)

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self.processing_provider)
