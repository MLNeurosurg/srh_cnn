
'''
imreg_dft code
https://anaconda.org/conda-forge/imreg_dft
'''

# standard python
import os
import sys
from collections import defaultdict, OrderedDict
# data science
import numpy as np
import pandas as pd
from scipy import stats
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

import math
import numpy
from numpy.fft import fft2, ifft2, fftshift
import scipy.ndimage.interpolation as ndii

def translation(im0, im1):
	"""Return translation vector to register images."""
	shape = im0.shape
	f0 = fft2(im0)
	f1 = fft2(im1)
	ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
	t0, t1 = numpy.unravel_index(numpy.argmax(ir), shape)
	if t0 > shape[0] // 2:
		t0 -= shape[0]
	if t1 > shape[1] // 2:
		t1 -= shape[1]
	return [t0, t1]

def highpass(shape):
	"""Return highpass filter to be multiplied with fourier transform."""
	x = numpy.outer(
		numpy.cos(numpy.linspace(-math.pi/2., math.pi/2., shape[0])),
		numpy.cos(numpy.linspace(-math.pi/2., math.pi/2., shape[1])))
	return (1.0 - x) * (2.0 - x)

def logpolar(image, angles=None, radii=None):
	"""Return log-polar transformed image and log base."""
	shape = image.shape
	center = shape[0] / 2, shape[1] / 2
	if angles is None:
		angles = shape[0]
	if radii is None:
		radii = shape[1]
	theta = numpy.empty((angles, radii), dtype='float64')
	theta.T[:] = numpy.linspace(0, numpy.pi, angles, endpoint=False) * -1.0
	# d = radii
	d = numpy.hypot(shape[0] - center[0], shape[1] - center[1])
	log_base = 10.0 ** (math.log10(d) / (radii))
	radius = numpy.empty_like(theta)
	radius[:] = numpy.power(log_base,
							numpy.arange(radii, dtype='float64')) - 1.0
	x = radius * numpy.sin(theta) + center[0]
	y = radius * numpy.cos(theta) + center[1]
	output = numpy.empty_like(x)
	ndii.map_coordinates(image, [x, y], output=output)
	return output, log_base

def similarity(im0, im1):
	"""
	Return similarity transformed image im1 and transformation parameters.

	Transformation parameters are: isotropic scale factor, rotation angle (in
	degrees), and translation vector.

	A similarity transformation is an affine transformation with isotropic
	scale and without shear.

	Limitations:
	Image shapes must be equal and square.
	All image areas must have same scale, rotation, and shift.
	Scale change must be less than 1.8.
	No subpixel precision.
	"""
	if im0.shape != im1.shape:
		raise ValueError('Images must have same shapes.')
	elif len(im0.shape) != 2:
		raise ValueError('Images must be 2 dimensional.')

	f0 = fftshift(abs(fft2(im0)))
	f1 = fftshift(abs(fft2(im1)))

	h = highpass(f0.shape)
	f0 *= h
	f1 *= h
	del h

	f0, log_base = logpolar(f0)
	f1, log_base = logpolar(f1)

	f0 = fft2(f0)
	f1 = fft2(f1)
	r0 = abs(f0) * abs(f1)
	ir = abs(ifft2((f0 * f1.conjugate()) / r0))
	i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
	angle = 180.0 * i0 / ir.shape[0]
	scale = log_base ** i1

	if scale > 1.8:
		ir = abs(ifft2((f1 * f0.conjugate()) / r0))
		i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
		angle = -180.0 * i0 / ir.shape[0]
		scale = 1.0 / (log_base ** i1)
		if scale > 1.8:
			raise ValueError('Images are not compatible. Scale change > 1.8')

	if angle < -90.0:
		angle += 180.0
	elif angle > 90.0:
		angle -= 180.0

	im2 = ndii.zoom(im1, 1.0/scale)
	im2 = ndii.rotate(im2, angle)

	if im2.shape < im0.shape:
		t = numpy.zeros_like(im0)
		t[:im2.shape[0], :im2.shape[1]] = im2
		im2 = t
	elif im2.shape > im0.shape:
		im2 = im2[:im0.shape[0], :im0.shape[1]]

	f0 = fft2(im0)
	f1 = fft2(im2)
	ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
	t0, t1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)

	if t0 > f0.shape[0] // 2:
		t0 -= f0.shape[0]
	if t1 > f0.shape[1] // 2:
		t1 -= f0.shape[1]

	im2 = ndii.shift(im2, [t0, t1])

	# correct parameters for ndimage's internal processing
	if angle > 0.0:
		d = int((int(im1.shape[1] / scale) * math.sin(math.radians(angle))))
		t0, t1 = t1, d+t0
	elif angle < 0.0:
		d = int((int(im1.shape[0] / scale) * math.sin(math.radians(angle))))
		t0, t1 = d+t1, d+t0
	scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

	return im2, scale, angle, [-t0, -t1]

def similarity_matrix(scale, angle, vector):
	"""Return homogeneous transformation matrix from similarity parameters.

	Transformation parameters are: isotropic scale factor, rotation angle (in
	degrees), and translation vector (of size 2).

	The order of transformations is: scale, rotate, translate.

	"""
	S = numpy.diag([scale, scale, 1.0])
	R = numpy.identity(3)
	angle = math.radians(angle)
	R[0, 0] = math.cos(angle)
	R[1, 1] = math.cos(angle)
	R[0, 1] = -math.sin(angle)
	R[1, 0] = math.sin(angle)
	T = numpy.identity(3)
	T[:2, 2] = vector
	return numpy.dot(T, numpy.dot(R, S))

def fft_register(image1, image2):
	"""
	Function take two single channel images and returns two channel registered image
	"""
	registered_image = np.zeros((image1.shape[0], image1.shape[1], 2), dtype=np.float)
	image3, scale, angle, (t0, t1) = similarity(image1, image2)
	registered_image[:,:,0] = image1
	registered_image[:,:,1] = image3
	return registered_image

def register_mosaic(CH2_mosaic, CH3_mosaic):
	"""
	Function that accepts yield
	"""
	step = 1000
	height = CH2_mosaic.shape[0]
	width = CH2_mosaic.shape[1]

	registered_image = np.zeros((CH2_mosaic.shape[0], CH2_mosaic.shape[1], 2), dtype=np.float)
	for x in np.arange(0, width, step = step):
		for y in np.arange(0, height, step = step):
			CH2_strip = CH2_mosaic[y:y+step, x:x + step]
			CH3_strip = CH3_mosaic[y:y+step, x:x + step]
			registered_image[y:y+step, x:x + step, :] = fft_register(CH2_strip, CH3_strip)
	return registered_image

def compare_pre_post_reg(reference_image, pre_image, post_image):
	"""
	Compare pre and post registration images
	"""

	def subtract_channels(CH2, CH3):
		CH3_minus_CH2 = CH3.astype(np.float) - CH2.astype(np.float)
		CH3_minus_CH2[CH3_minus_CH2 < 0] = 0
		return CH3_minus_CH2.astype(np.uint8)

	# pre-registered image
	pre_reg_image = np.zeros((reference_image.shape[0], reference_image.shape[1], 3), dtype=np.uint8)
	pre_reg_image[:,:,0] = subtract_channels(reference_image, pre_image)
	pre_reg_image[:,:,1] = reference_image
	pre_reg_image[:,:,2] = pre_image

	# postregistered image
	post_reg_image = np.zeros((reference_image.shape[0], reference_image.shape[1], 3), dtype=np.uint8)
	post_reg_image[:,:,0] = subtract_channels(reference_image, post_image)
	post_reg_image[:,:,1] = reference_image
	post_reg_image[:,:,2] = post_image

	plt.figure()
	plt.subplot(121)
	plt.title("Raw SRH channels")
	plt.imshow(pre_reg_image)

	plt.subplot(122)
	plt.imshow(post_reg_image)
	plt.title("Post-FFT registration")
	plt.show()


if __name__ == '__main__':
	pass

