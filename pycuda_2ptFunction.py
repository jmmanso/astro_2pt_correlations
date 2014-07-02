## CALCULATES THE 2-point CLUSTERING OF GALAXIES
## USING THE PYCUDA INTERFACE TO THE GPU


# -*- coding: utf-8 -*-
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import os
import matplotlib.pyplot as plt
import time


# DEFINE SOURCE MODULE THAT WILL BE COMPILED AND PASSED TO THE GPU
mod = SourceModule("""
__global__ void autocorr(float *xd,float *yd,float *zd,float *cd, float *ZZ, int number_lines , int points_per_degree, float max_degree)
{

    const int dim_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= number_lines) return;
    for (int i=0;i< number_lines;i++)
    {       
	  float dist = sqrtf( powf(xd[i] - xd[dim_idx] , 2) + powf(yd[i] - yd[dim_idx] , 2) + powf(zd[i] - zd[dim_idx] , 2));
		float angle = 2*asinf(dist/2)*180.0/3.14159265;
		if (angle >= max_degree) continue;
		atomicAdd( &ZZ[int(angle*points_per_degree )],  cd[i]*cd[dim_idx]  );
    }
}
__global__ void autocorr_int(float *xd,float *yd,float *zd, int *ZZ, int number_lines , int points_per_degree, float max_degree)
{
    const int dim_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= number_lines) return;
    for (int i=0;i< number_lines;i++)
    {       
	  float dist = sqrtf( powf(xd[i] - xd[dim_idx] , 2) + powf(yd[i] - yd[dim_idx] , 2) + powf(zd[i] - zd[dim_idx] , 2));
		float angle = 2*asinf(dist/2)*180.0/3.14159265;
		if (angle >= max_degree) continue;
		atomicAdd( &ZZ[int(angle*points_per_degree )],  1  );
    }
}

__global__ void cross_float(float *xd,float *yd,float *zd,float *cd, float *xdr,float *ydr,float *zdr,float *cdr ,float *ZZ, int number_lines ,int number_linesR, int points_per_degree, float max_degree)
{
// to maximize threading, we will thread over the random catalog and loop over the real one, since generally rand.size>real.size

    const int dim_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= number_linesR) return;
    for (int i=0;i< number_lines;i++)
    {       
	  float dist = sqrtf( powf(xd[i] - xdr[dim_idx] , 2) + powf(yd[i] - ydr[dim_idx] , 2) + powf(zd[i] - zdr[dim_idx] , 2));
		float angle = 2*asinf(dist/2)*180.0/3.14159265;
		if (angle >= max_degree) continue;
		atomicAdd( &ZZ[int(angle*points_per_degree )],  cd[i]  );
    }
}
__global__ void cross_int(float *xd,float *yd,float *zd,float *xdr,float *ydr,float *zdr, int *ZZ, int number_lines ,int number_linesR, int points_per_degree, float max_degree)
{
// to maximize threading, we will thread over the random catalog and loop over the real one, since generally rand.size>real.size

    const int dim_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= number_linesR) return;
    for (int i=0;i< number_lines;i++)
    {       
	  float dist = sqrtf( powf(xd[i] - xdr[dim_idx] , 2) + powf(yd[i] - ydr[dim_idx] , 2) + powf(zd[i] - zdr[dim_idx] , 2));
		float angle = 2*asinf(dist/2)*180.0/3.14159265;
		if (angle >= max_degree) continue;
		atomicAdd( &ZZ[int(angle*points_per_degree )],  1  );
    }
}

""")


# These are python functions that process the input position arrays of galaxies and feed them 
# to the source module
def tpcf(cat1,cat2=None, maxdeg=7.0, ptsperdeg=2000):
	inicio = time.time()
	theta1 = cat1[:,1]*np.pi/180.0
	phi1 = cat1[:,0]*np.pi/180.0
	cth1 = np.sin(theta1)
	if cat1.shape[1]==2: cat1 = np.hstack(( cat1, np.tile(1.0,cat1.shape[0]).reshape((cat1.shape[0],1)) ))
	c_weight1 = 1.0/cat1[:,2]
	xd1 = np.cos(phi1)*np.cos(theta1)
	yd1 = np.sin(phi1)*np.cos(theta1)
	zd1 = cth1
	if cat2 != None:
		if cat2.shape[1]==2: cat2 = np.hstack(( cat2, np.tile(1.0,cat2.shape[0]).reshape((cat2.shape[0],1)) ))
		c_weightr = 1.0/cat2[:,2]
		thetar = cat2[:,1]*np.pi/180.0
		phir = cat2[:,0]*np.pi/180.0
		cthr = np.sin(thetar)
		xdr = np.cos(phir)*np.cos(thetar)
		ydr = np.sin(phir)*np.cos(thetar)
		zdr = cthr
		xdr = xdr.astype(np.float32)
		ydr = ydr.astype(np.float32)
		zdr = zdr.astype(np.float32)
		c_weightr = c_weightr.astype(np.float32)

	xd1 = xd1.astype(np.float32)
	yd1 = yd1.astype(np.float32)
	zd1 = zd1.astype(np.float32)
	c_weight1 = c_weight1.astype(np.float32)
	
	# Bear in mind the maximum number of blocks is 65535
	
	# elements in ZZ (int32) cannot hold a number greater than 2^31 =  2,147,483,648, but that is easily reachable. BEWARE
	
	block_size = 256
	auto = mod.get_function("autocorr")
	cross = mod.get_function("cross")
	
	nlines1 = xd1.size
	blocks1 = nlines1/block_size
	if (nlines1 % block_size != 0): blocks1 += 1

	out_arr = np.zeros(int(ptsperdeg*maxdeg),'float32')
	angles = np.linspace(1./ptsperdeg , maxdeg , int(ptsperdeg*maxdeg))
	print "Ready to load GPU. Elapsed time: ", np.floor(time.time()-inicio), " seconds"
	if cat2 == None:
		auto(cuda.In(xd1), cuda.In(yd1),cuda.In(zd1),cuda.In(c_weight1),cuda.InOut(out_arr),np.int32(nlines1),np.int32(ptsperdeg),np.float32(maxdeg), grid=(blocks1,1), block=(block_size,1,1))
	
	if cat2 != None:
		nlinesr = xdr.size
		blocksr = nlinesr/block_size
		if (nlinesr % block_size != 0): blocksr += 1
		cross(cuda.In(xd1), cuda.In(yd1),cuda.In(zd1),cuda.In(c_weight1),cuda.In(xdr), cuda.In(ydr),cuda.In(zdr),cuda.In(c_weightr), cuda.InOut(out_arr),np.int32(nlines1),np.int32(nlinesr),np.int32(ptsperdeg),np.float32(maxdeg), grid=(blocksr,1), block=(block_size,1,1))

	return out_arr


def auto_int(cat1, maxdeg=7.0, ptsperdeg=2000):
	inicio = time.time()
	theta1 = cat1[:,1]*np.pi/180.0
	phi1 = cat1[:,0]*np.pi/180.0
	cth1 = np.sin(theta1)
	xd1 = np.cos(phi1)*np.cos(theta1)
	yd1 = np.sin(phi1)*np.cos(theta1)
	zd1 = cth1
	xd1 = xd1.astype(np.float32)
	yd1 = yd1.astype(np.float32)
	zd1 = zd1.astype(np.float32)

	block_size = 256
	auto = mod.get_function("autocorr_int")

	nlines1 = xd1.size
	blocks1 = nlines1/block_size
	if (nlines1 % block_size != 0): blocks1 += 1

	out_arr = np.zeros(int(ptsperdeg*maxdeg),'int32')
	angles = np.linspace(1./ptsperdeg , maxdeg , int(ptsperdeg*maxdeg))
	print "Ready to load GPU. Elapsed time: ", np.floor(time.time()-inicio), " seconds"
	auto(cuda.In(xd1), cuda.In(yd1),cuda.In(zd1),cuda.InOut(out_arr),np.int32(nlines1),np.int32(ptsperdeg),np.float32(maxdeg), grid=(blocks1,1), block=(block_size,1,1))
	print nlines1
	out_arr[0] = out_arr[0] - nlines1
	return out_arr

def cross_int(cat1,cat2,  maxdeg=7.0, ptsperdeg=2000):
	inicio = time.time()
	theta1 = cat1[:,1]*np.pi/180.0
	phi1 = cat1[:,0]*np.pi/180.0
	cth1 = np.sin(theta1)
	xd1 = np.cos(phi1)*np.cos(theta1)
	yd1 = np.sin(phi1)*np.cos(theta1)
	zd1 = cth1
	xd1 = xd1.astype(np.float32)
	yd1 = yd1.astype(np.float32)
	zd1 = zd1.astype(np.float32)
	thetar = cat2[:,1]*np.pi/180.0
	phir = cat2[:,0]*np.pi/180.0
	cthr = np.sin(thetar)
	xdr = np.cos(phir)*np.cos(thetar)
	ydr = np.sin(phir)*np.cos(thetar)
	zdr = cthr
	xdr = xdr.astype(np.float32)
	ydr = ydr.astype(np.float32)
	zdr = zdr.astype(np.float32)
	#
	cross = mod.get_function("cross_int")
	block_size = 256
	nlines1 = xd1.size
	blocks1 = nlines1/block_size
	if (nlines1 % block_size != 0): blocks1 += 1
	#
	nlinesr = xdr.size
	blocksr = nlinesr/block_size
	if (nlinesr % block_size != 0): blocksr += 1
	#
	out_arr = np.zeros(int(ptsperdeg*maxdeg),'int32')
	print "Ready to load GPU. Elapsed time: ", np.floor(time.time()-inicio), " seconds"
	cross(cuda.In(xd1), cuda.In(yd1),cuda.In(zd1),cuda.In(xdr), cuda.In(ydr),cuda.In(zdr), cuda.InOut(out_arr),np.int32(nlines1),np.int32(nlinesr),np.int32(ptsperdeg),np.float32(maxdeg), grid=(blocksr,1), block=(block_size,1,1))
	return out_arr


def cross_float(cat1,cat2,  maxdeg=7.0, ptsperdeg=2000):
	inicio = time.time()
	theta1 = cat1[:,1]*np.pi/180.0
	phi1 = cat1[:,0]*np.pi/180.0
	weight1 = 1.0/cat1[:,2]
	xd1 = np.cos(phi1)*np.cos(theta1)
	yd1 = np.sin(phi1)*np.cos(theta1)
	zd1 = np.sin(theta1)
	xd1 = xd1.astype(np.float32)
	yd1 = yd1.astype(np.float32)
	zd1 = zd1.astype(np.float32)
	weight1 = weight1.astype(np.float32)
	thetar = cat2[:,1]*np.pi/180.0
	phir = cat2[:,0]*np.pi/180.0
	xdr = np.cos(phir)*np.cos(thetar)
	ydr = np.sin(phir)*np.cos(thetar)
	zdr = np.sin(thetar)
	if cat2.shape[1]==3:  weightr = 1.0/cat2[:,2]
	else: weightr = np.ones(xdr.size)
	xdr = xdr.astype(np.float32)
	ydr = ydr.astype(np.float32)
	zdr = zdr.astype(np.float32)
	weightr = weightr.astype(np.float32)
	#
	cross = mod.get_function("cross_float")
	block_size = 256
	nlines1 = xd1.size
	blocks1 = nlines1/block_size
	if (nlines1 % block_size != 0): blocks1 += 1
	#
	nlinesr = xdr.size
	blocksr = nlinesr/block_size
	if (nlinesr % block_size != 0): blocksr += 1
	#
	out_arr = np.zeros(int(ptsperdeg*maxdeg),'float32')
	print "Ready to load GPU. Elapsed time: ", np.floor(time.time()-inicio), " seconds"
	cross(cuda.In(xd1), cuda.In(yd1),cuda.In(zd1), cuda.In(weight1), cuda.In(xdr), cuda.In(ydr),cuda.In(zdr),cuda.In(weightr) ,cuda.InOut(out_arr),np.int32(nlines1),np.int32(nlinesr),np.int32(ptsperdeg),np.float32(maxdeg), grid=(blocksr,1), block=(block_size,1,1))
	return out_arr


def estimator(gal, ran, rr, Nsamples , maxdeg = 14.0,  ptsperdeg = 3600):
	root_path = '/home/j.martinez/project/'
	angles = np.linspace(1./ptsperdeg , maxdeg , int(ptsperdeg*maxdeg))
	Nr = ran.shape[0]
	min_compl = np.min(gal[:,2]); max_compl = np.max(gal[:,2]);
	compl_boundaries = np.linspace(min_compl, max_compl, Nsamples+1)
	gal_list = []
	N_list = []
	weight_list = []
	for j in range(Nsamples):
		whichones = np.where((gal[:,2]>compl_boundaries[j] )&(gal[:,2]<compl_boundaries[j+1] ))[0]
		gal_list.append(gal[whichones,:2 ])
		weight_list.append(np.sum(1./gal[whichones,2 ]))
		N_list.append(gal_list[j].shape[0])
	#
	n_list = Nr/(1.0*np.array(N_list))
	total_weight = sum(weight_list)
	weight_list = np.array(weight_list)/total_weight
	dumpH = np.zeros(int(maxdeg*ptsperdeg) ); dumpLS = np.zeros_like(dumpH)
	for k in range(Nsamples):
		for j in range(k+1):
			ab = cross_int(gal_list[k], gal_list[j], maxdeg=maxdeg , ptsperdeg=ptsperdeg)
			if k==j: ab[0] = ab[0] - gal_list[k].shape[0]
			ar = cross_int(gal_list[k], ran, maxdeg=maxdeg , ptsperdeg=ptsperdeg)
			br = cross_int(gal_list[j], ran, maxdeg=maxdeg , ptsperdeg=ptsperdeg)
			H = (1.0*ab)*(1.0*rr)/(1.0*ar*br) -1.0
			LS = (1.0*ab*n_list[j]*n_list[k] - 1.0*ar*n_list[k]- 1.0*br*n_list[j])/(1.0*rr) +1.0
			H_w = H * weight_list[k]*weight_list[j]
			LS_w = LS * weight_list[k]*weight_list[j]
			if k==j: dumpH += H_w; dumpLS += LS_w
			if k!=j: dumpH += 2*H_w; dumpLS += 2*LS_w
	#
	return np.array(zip(angles,dumpH))














