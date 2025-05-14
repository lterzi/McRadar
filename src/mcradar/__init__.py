# -*- coding: utf-8 -*-

"""Top-level package for McRadar."""

__version__ = "1.0.0"

from .settings import loadSettings
from .tableOperator import getMcSnowTable

from .spectraOperator import getMultFrecSpec
from .spectraOperator import convoluteSpec
from .zeOperator import calcParticleZe
from .kdpOperator import getIntKdp
from .attenuationOperator import getHydroAtmAtt

from .fullRadarOperator import fullRadar
from .fullRadarOperator import fullRadarParallel

from .fullRadarOperator import singleParticleTrajectories
from .fullRadarOperator import singleParticleScat

from .utilities import *


