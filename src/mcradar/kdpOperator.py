# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import xarray as xr

class KdpOperator:
	"""
	Operator for calculating integrated KDP from a distribution of particles.
	Usage: KdpOperator().compute(mcTable, centerHeight)
	"""
	def __init__(self, config=None):
		self.config = config

	def compute(self, mcTable, centerHeight):
		"""
		Calculates the integrated kdp of a distribution of particles.

		Parameters
		----------
		mcTable: McSnow table returned from calcParticleKDP()
		centerHeight: height of the center of the distribution of particles

		Returns
		-------
		tmpKdp: kdp calculated of a distribution of particles, separated for monomers and aggregates
		tmpKdp: dims=(range)
		"""
		tmpKdp = xr.Dataset()
		sKDP = mcTable.sKDP * mcTable['sMult']
		tmpKdp['KDP'] = sKDP.sum(dim='index').expand_dims({'range': np.asarray(centerHeight).reshape(1)})
		# now differently for Aggregates and Monomers
		mcTableMono = mcTable.where(mcTable['sNmono'] == 1, drop=True)  # select only crystals
		mcTableAgg = mcTable.where(mcTable['sNmono'] > 1, drop=True)
		melted_particle = (mcTable.m_w > 0) & (mcTable.mass_all_ice > 0)
		mcTableMelted = mcTable.where(melted_particle, drop=True)  # select only melted particles
		liquid_particles = (mcTable.m_w > 0) & (mcTable.mass_all_ice == 0)
		mcTableLiquid = mcTable.where(liquid_particles, drop=True)  # remove particles with liquid water present
		mcTableIce = mcTable.where(mcTable.m_w == 0, drop=True)  # remove particles with liquid water present
		mcTableIce['rimefraction'] = mcTableIce.m_r / mcTableIce.mass_all_ice
		rimedparticles = (mcTableIce.rimefraction > 0.5) & (mcTableIce.sPhi > 0.7) & (mcTableIce.sPhi < 1.3)
		mcTableRimed = mcTableIce.where(rimedparticles, drop=True)

		KDPMono = mcTableMono['sKDP'] * mcTableMono['sMult']
		tmpKdp['KDPMono'] = KDPMono.sum(dim='index').expand_dims({'range': np.asarray(centerHeight).reshape(1)})
		KDPAgg = mcTableAgg['sKDP'] * mcTableAgg['sMult']
		tmpKdp['KDPAgg'] = KDPAgg.sum(dim='index').expand_dims({'range': np.asarray(centerHeight).reshape(1)})
		KDPMelted = mcTableMelted['sKDP'] * mcTableMelted['sMult']
		tmpKdp['KDPMelted'] = KDPMelted.sum(dim='index').expand_dims({'range': np.asarray(centerHeight).reshape(1)})
		KDPLiquid = mcTableLiquid['sKDP'] * mcTableLiquid['sMult']
		tmpKdp['KDPLiquid'] = KDPLiquid.sum(dim='index').expand_dims({'range': np.asarray(centerHeight).reshape(1)})
		KDPIce = mcTableIce['sKDP'] * mcTableIce['sMult']
		tmpKdp['KDPIce'] = KDPIce.sum(dim='index').expand_dims({'range': np.asarray(centerHeight).reshape(1)})
		KDPRimed = mcTableRimed['sKDP'] * mcTableRimed['sMult']
		tmpKdp['KDPRimed'] = KDPRimed.sum(dim='index').expand_dims({'range': np.asarray(centerHeight).reshape(1)})
		return tmpKdp


