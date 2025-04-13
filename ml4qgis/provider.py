import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algs.random_forest import RandomForestProcessingAlgorithm


class Ml4QgisProcessingProvider(QgsProcessingProvider):
    def __init__(self):
        QgsProcessingProvider.__init__(self)

        self.activate = True

        # Load algorithms
        self.alglist = [RandomForestProcessingAlgorithm()]

        for alg in self.alglist:
            alg.provider = self

    def getAlgs(self):
        return self.alglist

    def id(self):
        return "ml4qgis"

    def name(self):
        """This is the name that will appear on the toolbox group.

        It is also used to create the command line name of all the
        algorithms from this provider.
        """
        return "Machine Learning"

    def icon(self):
        return QIcon(self.svgIconPath())

    def svgIconPath(self):
        basepath = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(basepath, "icon.svg")

    def loadAlgorithms(self):
        self.algs = self.getAlgs()
        for a in self.algs:
            self.addAlgorithm(a)
