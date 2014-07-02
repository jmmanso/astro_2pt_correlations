# THIS CODE TAKES IN 2-BAND IMAGES FROM THE SPITZER INFRARED TELESCOPE.
# PRODUCES PHOTOMETRIC CATALOG OF SOURCES, AND ANALYZES EACH OF THEM
# TO ASSESS BLENDING, INCOMPLETNESS AND DEPTH LEVELS AS A FUNCTION
# OF FLUX, COLOR AND IMAGE REGION.
# HAS SEVERAL DEPENDENCIES THAT ARE NOT INCLUDED.


# -*- coding: utf-8 -*-
import pywcs
import numpy as np
import pyfits
import os
import sys
import time
import pygfit
import congrid
from pygfit.pygsti import slop
from scipy import interpolate
from multiprocessing import Pool, Array
import subprocess

##### PHOTOMETRIC DEPTH FOR CH1 IN DUAL MODE ####
# Completeness for CH2 can be run independently (to save time).
# We are interested in the combined photometric depth of a source with given magnitudes in
# CH1 and CH2. Selection is made in CH2. The main reason to do this analysis in dual
# mode instead of independently in both channels is because here we enforce the
# centroid positions and apertures to be the same. Due to crowding/faintness, the centroid can
# shift from its real position. We want to fix that position in CH2, use it to measure CH1 magnitudes.
# Thus, in general, we will recover worse photometric quality in dual mode for CH1 than doing it independently.

#For a small set of CH2 mags, (14,14.5,15,15.5 ...), we will produce for each one random positions in an
# image and feed them also random CH1 mags with color=[0,1] (to reduce computation time, we constrain ourselves only
# to those colors that matter).

# 0. RA  
# 1. DEC
# 2. CH2_INPUT_MAG.
# 3. CH1_INPUT_MAG.
# 4. ID
# 5. MEAN COVERAGE OF OBJECT
# 6. DATA-WEIGHTED MEAN COVERAGE OF OBJECT
# 7. RECOVERED_CH2_FLUX(3'')
# 8. RECOVERED_CH2_FLUX(4'')
# 9. RECOVERED_CH2_FLUX(5'')
# 10. RECOVERED_CH2_FLUX(6'')
# 11. RECOVERED_CH2_FLUX(12'')
# 12. RECOVERED_CH1_FLUX(3'')
# 13. RECOVERED_CH1_FLUX(4'')
# 14. RECOVERED_CH1_FLUX(5'')
# 15. RECOVERED_CH1_FLUX(6'')
# 16. RECOVERED_CH1_FLUX(12'')
# 17. DETECTION INDICATOR (1 for detected, 0 not detected)
# 18. NUMBER OF CH2 SOURCES IN SEARCH APERTURE FROM ORIGINAL CATALOG
# 19. NUMBER OF CH2 SOURCES IN SEARCH APERTURE FROM RECOVERED CATALOG 
# 20. DISTANCE IN ARCSEC FROM CH2 INPUT TO RECOVERED POSITION 
# 21. IDENTIFIER OF DETECTION METHOD
q = {'ra':0,'dec':1,'ch2in':2,'ch1in':3,'id':4,'cov':5,'wcov':6,'ch2_3':7,
'ch2_4':8,'ch2_5':9,'ch2_6':10,'ch2_12':11,'ch1_3':12,'ch1_4':13,'ch1_5':14,
'ch1_6':15,'ch1_12':16,'det':17,'Norig':18,'Nsim':19,'dist':20,'method':21}

qfmt = {'ra':'%1.6f','dec':'%1.6f','ch2in':'%1.2f','ch1in':'%1.2f','id':'%1.0f','cov':'%1.2f',
'wcov':'%1.0f','ch2_3':'%1.4f','ch2_4':'%1.4f','ch2_5':'%1.4f','ch2_6':'%1.4f','ch2_12':'%1.4f',
'ch1_3':'%1.4f','ch1_4':'%1.4f','ch1_5':'%1.4f','ch1_6':'%1.4f','ch1_12':'%1.4f','det':'%1.0f',
'Norig':'%1.0f','Nsim':'%1.0f','dist':'%1.2f','method':'%1.0f'}

fmt_string = ''
for i in range(len(q)): 
	keypos = np.where(np.array(q.values())==i)[0][0];
	fmt_string = fmt_string + ' '+qfmt[q.keys()[keypos]]

if len(q) != len(qfmt): raise ValueError( "Number of formats and table entries is different")

odic = {'ra':0,'dec':1,'ch2_3':2}
cdic = {'ra':0,'dec':1,'number':2,'ch2_3':3,'ch2_4':4,'ch2_5':5,'ch2_6':6,'ch2_12':7,
'ch1_3':8,'ch1_4':9,'ch1_5':10,'ch1_6':11,'ch1_12':12}

tile = sys.argv[1]
cov_option = sys.argv[2]# 'cov' makes coverage
sim_path = sys.argv[3]
sexpath = '/astro/data/siesta1/PROJECTS/SSDF/completeness/sex/'
filepath = '/astro/data/siesta1/DATA/reduced/SSDF/latest/'
orig_ch1_image = filepath+'I1_SSDF'+tile+'_mosaic.fits'
orig_ch2_image = filepath+'I2_SSDF'+tile+'_mosaic.fits'
psf_ch1_image = '/astro/data/siesta1/PROJECTS/SSDF/completeness/psfs/psf1_circ.fits'
psf_ch2_image = '/astro/data/siesta1/PROJECTS/SSDF/completeness/psfs/psf2_circ.fits'
cov_ch2_image = '/astro/data/siesta1/DATA/reduced/SSDF/I2_SSDF'+tile+'_mosaic_cov.fits'
zpt_ch1 = 18.789; zpt_ch2 = 18.316;# from README file in Matt's catalog
zpts = [zpt_ch1,zpt_ch2]
vega2ab_ch1 = 2.78; vega2ab_ch2 = 3.26;


def dist(a ,b):
	if b.ndim==1: return np.sqrt(((a[0]-b[0])*np.cos(b[1]*np.pi/180.0))**2 + (a[1]-b[1])**2)*3600.
	else: return np.sqrt(((a[0]-b[:,0])*np.cos(b[:,1]*np.pi/180.0))**2 + (a[1]-b[:,1])**2)*3600.

# Flux in ADU
def mag2flux(mag, chann):
	return 10.**(0.4*(zpts[chann-1] - mag ))

def flux2mag(flux, chann):
	return zpts[chann-1] - 2.5*np.log10(flux)

def mag2flux_Jy(mag, chann):
	return 10.**(0.4*(23.9 - mag - [2.78,3.26][chann-1]))

def flux2mag_Jy(flux, chann):
	return 23.9 - 2.5*np.log10(flux) - [2.78,3.26][chann-1]

def add_mags(maglist, chann):
	fluxes = np.zeros(0)
	for t in range(maglist.size): fluxes = np.hstack((fluxes, mag2flux(maglist[t], chann)))
	return flux2mag( sum(fluxes) ,chann)

# The relationship between input and output auto and 2'' mags for sources in normal background
# is biased, but we have that curve, and correct for it
flux_bias_tbl = np.loadtxt('/astro/data/siesta1/PROJECTS/SSDF/completeness/aper_corr/corr0.dat')
# this is a function representing {auto,2''}mag - input_mag, for both channels.
# That is, if you are comparing input and auto magnitudes, you could do input-> input + mag_bias(input)

def flux_bias(ch2fluxes, aperture):
	if aperture=='ap3': aper=1;
	if aperture=='ap5': aper=2; # CAREFUL about the order of rows in corr file
	flux_bias_intr = interpolate.interp1d( flux_bias_tbl[:,aper], flux_bias_tbl[:,0], kind='linear')
	minflux = min(flux_bias_tbl[:,aper])
	maxflux = max(flux_bias_tbl[:,aper])
	def aux(flux):
		if flux < minflux: #extrapolate linearly tot he fainter end using the 2 faintest points
			m = (flux_bias_tbl[0,0]-flux_bias_tbl[0,aper])/(flux_bias_tbl[1,0]-flux_bias_tbl[1,aper])
			return flux * m + flux_bias_tbl[0,0]-flux_bias_tbl[0,aper]*m
		#
		if flux > maxflux: return flux_bias_intr(maxflux) * flux /maxflux
		return flux_bias_intr(flux) #* flux
	#
	vaux = np.vectorize(aux)
	return vaux(ch2fluxes)


def remove_matches(aux1, aux2):
	if (aux2.size ==0) or (aux1.size==0): return aux1, aux2
	nonmatched_cat = np.copy(aux2)
	nonmatched_orig = np.copy(aux1)
	# for each source in the simulation, we calculate the distance to each original source, finding
	# the closest one. If that distance is less than 0.9'' (1px=0.6'') AND the 3'' magnitudes differ by
	# less than 0.5 mags, we consider it a match, and remove these sources from the nonmatched lists.
	for p in range(aux2.size):
		h = dist(cat[aux2[p]],original_cat[aux1]) # distance
		hmin = np.argmin(h) # find index of minimum distance (h indexes aux1 array)
		if (h[hmin]<0.9) and max(original_cat[aux1[hmin],odic['ch2_3']], cat[aux2[p],cdic['ch2_3']])/min(original_cat[aux1[hmin],odic['ch2_3']], cat[aux2[p],cdic['ch2_3']]) <1.5: 
			nonmatched_cat = np.delete(nonmatched_cat, np.where(nonmatched_cat==aux2[p])[0])
			nonmatched_orig = np.delete(nonmatched_orig, np.where(nonmatched_orig==aux1[hmin])[0])
	aux2 = nonmatched_cat.copy()
	aux1 = nonmatched_orig.copy()
	return aux1, aux2

def make_sims(ch2_mag ):
	# load images and psfs
	ch1 = pyfits.open( orig_ch1_image )
	ch2 = pyfits.open( orig_ch2_image )
	psf1 = pyfits.open( psf_ch1_image )
	psf2 = pyfits.open( psf_ch2_image )
	# import data arrays and headers
	data1 = ch1[0].data.copy()
	data2 = ch2[0].data.copy()
	hdr1 = ch1[0].header
	hdr2 = ch2[0].header
	# We don't want to place sources near the image edges or inside bright star haloes.
	# Thus, we define a margin in pixels to avoid edges and load the star halo catalog
	# to avoid placing sources in them. Also, we define the number of fake sources that each
	# simulation iteration has.
	x1=15.;x2=21.;y1=1500.;y2=4000.; m_lin = (y2-y1)/(x2-x1);b_lin=y1-m_lin*x1; # nsources is drawn from custom linear relation
	nsources = int(m_lin * ch2_mag + b_lin )
	margin = 50
	pobject = pywcs.WCS(hdr2)
	# Load stars
	stars = np.loadtxt('/astro/data/siesta1/PROJECTS/SSDF/catalog/stars/holes_K12.dat')
	# Transform star coordinates to pixels
	stars[:,:2] = np.array(pobject.wcs_sky2pix(  np.array( zip(stars[:,0],stars[:,1])  )  , 1))
	# Just keep those stars that are centered in pixels of the tile
	stars = stars[ (stars[:,0]>1)&(stars[:,0]<hdr2['NAXIS1'])& (stars[:,1]>1)&(stars[:,1]<hdr2['NAXIS2'])]
	# Transform radii from degrees to px
	stars[:,2] = stars[:,2]/abs(hdr2['CD1_1'])
	# Create random source positions and only keep those ones that are outside the star haloes.
	xpos=np.zeros(0); ypos=np.zeros(0); #covpos=np.zeros(0);
	min_dist = 80.0-ch2_mag*3.3333# 30px for 15 and 10px for 21
	while xpos.size < nsources:
		yy = margin + np.random.random()*(data2.shape[0] - 2*margin)
		xx = margin + np.random.random()*(data2.shape[1] - 2*margin)
		if np.any( np.sqrt((xx-stars[:,0])**2 + (yy-stars[:,1])**2) < stars[:,2]) == True : continue
		# check that this fake source is not overlapping with another fake source previosly placed
		if np.any( np.sqrt((xx-xpos)**2 + (yy-ypos)**2) < min_dist) == True : continue
		# it's all good now, record fake source position
		xpos = np.append(xpos, xx); 
		ypos = np.append(ypos, yy);
		#covpos = np.append(covpos, cov[yy,xx]);
	
	# Project pixels coords to sky
	coords = np.array(pobject.wcs_pix2sky(  np.array( zip(xpos,ypos)  )  , 1))
	
	# Initalize slop object
	pad_length = 25
	generator_ch1 = slop( ch1[0].data, psf1[0].data, zpt_ch1, pad_length, sim_path, hdr=hdr1 )
	generator_ch2 = slop( ch2[0].data, psf2[0].data, zpt_ch2, pad_length, sim_path, hdr=hdr2 )
	
	# Create list of ch2_mags. It is just a list with ch2_mag values repeated
	mags2 = np.tile(ch2_mag,nsources)
	
	# Define the allowed color (i.e. allowed range of ch1 mag)
	color_array = np.arange(0.2,1.1,0.2) # set color array from 0.2 to 1, in steps of 0.1
	# make ch1_mags
	mags1 = ch2_mag + np.tile( color_array, int(nsources/color_array.size)+1)
	mags1 = mags1[:nsources] # make sure mag1 has size = nsources
	#####mags1 = ch2_mag + start_respect_ch2 + np.random.random(nsources)*delta_color
	
	# Create the fake input catalogs
	# Start with ch2
	fake2 =  {'id': np.arange(nsources).astype('|S3'), 'ra': coords[:,0], 
	'dec': coords[:,1], 'model': np.tile('point',nsources ), 
	'mag': mags2, 're': np.zeros( nsources ), 're_pix': np.zeros( nsources ), 
	'n': np.zeros( nsources ), 'ba': np.zeros( nsources ), 'pa': np.zeros( nsources )}
	
	# For ch1, the only thing that changes is the mags array
	fake1 =  {'id': np.arange(nsources).astype('|S3'), 'ra': coords[:,0], 
	'dec': coords[:,1], 'model': np.tile('point',nsources ), 
	'mag': mags1, 're': np.zeros( nsources ), 're_pix': np.zeros( nsources ), 
	'n': np.zeros( nsources ), 'ba': np.zeros( nsources ), 'pa': np.zeros( nsources )}
	
	# save fake source catalog, [ra,dec,ch2mag, ch1mag]
	simcat = np.concatenate((coords , mags2.reshape(nsources,1), mags1.reshape(nsources,1)),axis=1)
	np.savetxt(sim_path+'simcat.dat',simcat,fmt=('%1.9g %1.8g %1.4g %1.4g'))
	
	# place fake images on the main image
	generator_ch2.create_models( fake2 )
	generator_ch1.create_models( fake1 )
	
	# Create fits files and save them. Clobber=True overwrites files if existent
	sim_img_ch2 = generator_ch2.get_full_model()
	pyfits.writeto( sim_path+'sim_img_ch2.fits', sim_img_ch2 ,hdr2 , clobber=True )
	
	sim_img_ch1 = generator_ch1.get_full_model()
	pyfits.writeto( sim_path+'sim_img_ch1.fits', sim_img_ch1 ,hdr1 , clobber=True )
	return None;


# make original catalog
start_time = time.time()
# Run SEx on the tile image without simulated stars. It is done in DUAL mode, selecting and extracting targets 
# in CH2. This returns two 
# catalogs ch'+channel+'.'+tile+'_original.cat in sim_path. Both catalogs have one-to-one correspondence in their rows
os.system('rm -f '+sim_path+'control_dual.'+tile+'.dat')

os.system('rm -f '+sim_path+'ch2.'+tile+'_original.cat')
os.system('sex '+ filepath +'I2_SSDF'+tile+'_mosaic.fits, '+ filepath +'I2_SSDF'+tile+'_mosaic.fits -c '+ sexpath +'make_ch2.sex -PARAMETERS_NAME '+ 
sexpath +'newiracparams.param -CATALOG_NAME '+sim_path+'ch2.'+tile+'_original.cat -MAG_ZEROPOINT '+str(zpt_ch2)+'  -CHECKIMAGE_NAME '+sim_path+'I2_check_bg.'+tile+'.fits,'+sim_path+'I2_check_seg.'+tile+'.fits -FLAG_IMAGE '+ 
filepath +'I2_SSDF'+tile+'_flag.fits -WEIGHT_IMAGE '+ filepath +'I2_SSDF'+tile+'_mosaic_cov.fits')

os.system('rm -f '+sim_path+'ch1.'+tile+'_original.cat')
os.system('sex '+ filepath +'I2_SSDF'+tile+'_mosaic.fits, '+ filepath +'I1_SSDF'+tile+'_mosaic.fits -c '+ sexpath +'make_ch1.sex -PARAMETERS_NAME '+ 
sexpath +'newiracparams.param -CATALOG_NAME '+sim_path+'ch1.'+tile+'_original.cat -MAG_ZEROPOINT '+str(zpt_ch1)+'  -CHECKIMAGE_NAME '+sim_path+'I1_check_bg.'+tile+'.fits,'+sim_path+'I1_check_seg.'+tile+'.fits -FLAG_IMAGE '+ 
filepath +'I2_SSDF'+tile+'_flag.fits -WEIGHT_IMAGE '+ filepath +'I2_SSDF'+tile+'_mosaic_cov.fits')

# Now we want to create the "original catalog" by merging the photometry of both catalogs just obtained.
# This catalog will have columns of [ra,dec, ch2_magisocor, ch2_mag3'']. 
#original_cat = np.loadtxt('/astro/data/siesta1/PROJECTS/SSDF/completeness/slop/ch2.'+tile+'_original.cat',dtype='float',usecols=(0,1,7))
original_cat = np.loadtxt(sim_path+'ch2.'+tile+'_original.cat',dtype='float',usecols=(0,1,8))
# Start simulation iterations
mag_list = np.arange(15.,21.5,.5)
for ch2_mag in mag_list: 
	for k in range(4):
		# Control will be the array where we save the simulation results
		control = np.zeros(len(q))
		# Make sims. This creates N artificial stars and dumps them in the science image, creating sim_path/sim_img_ch?.fits (ie sim_image)
		make_sims(ch2_mag)
		#Run SEX to obtain catalogs of the simulated images, ch?.'+tile+'.cat
		os.system('rm -f '+sim_path+'ch2.'+tile+'.cat')
		os.system('sex '+ sim_path +'sim_img_ch2.fits, '+ sim_path +'sim_img_ch2.fits -c '+ sexpath +'make_ch2.sex -PARAMETERS_NAME '+ 
		sexpath +'newiracparams.param -CATALOG_NAME '+sim_path+'ch2.'+tile+'.cat -MAG_ZEROPOINT '+str(zpt_ch2)+'  -CHECKIMAGE_NAME '+sim_path+'I2_check_bg.'+tile+'.fits,'+sim_path+'I2_check_seg.'+tile+'.fits -FLAG_IMAGE '+ 
		filepath +'I2_SSDF'+tile+'_flag.fits -WEIGHT_IMAGE '+ filepath +'I2_SSDF'+tile+'_mosaic_cov.fits')
		
		os.system('rm -f '+sim_path+'ch1.'+tile+'.cat')
		os.system('sex '+ sim_path +'sim_img_ch2.fits, '+ sim_path +'sim_img_ch1.fits -c '+ sexpath +'make_ch1.sex -PARAMETERS_NAME '+ 
		sexpath +'newiracparams.param -CATALOG_NAME '+sim_path+'ch1.'+tile+'.cat -MAG_ZEROPOINT '+str(zpt_ch1)+' -CHECKIMAGE_NAME '+sim_path+'I1_check_bg.'+tile+'.fits,'+sim_path+'I1_check_seg.'+tile+'.fits -FLAG_IMAGE '+ 
		filepath +'I2_SSDF'+tile+'_flag.fits -WEIGHT_IMAGE '+ filepath +'I2_SSDF'+tile+'_mosaic_cov.fits')
		
		# merge catalogs 
		# This new simulation catalog has columns [ra, dec, number, ch2_flux(3''), ch2_flux(4''), ch2_flux(6''), ch2_flux(12'')]. 
		cat = np.loadtxt(sim_path+'ch2.'+tile+'.cat',dtype='float',usecols=(0,1,2,8,9,10,11,12))
		# append the aperture fluxes for ch1 as well
		cat = np.hstack((cat, np.loadtxt(sim_path+'ch1.'+tile+'.cat',dtype='float',usecols=(8,9,10,11,12))))
		# Load input catalog of simulated sources (i.e., it is a catalog of only the fake sources),
		# cols=ra,dec,input_mag_ch2,input_mag_ch1 
		fake_gals = np.loadtxt(sim_path+'simcat.dat',dtype='float') 
		# Create another data storing array, new_fake_gals, by adding N columns to fake_gals.
		# Basically, for each fake_gal (each line), we will store the photometric information
		# of the simulation.
		new_fake_gals = np.zeros(( np.shape(fake_gals)[0] , np.shape(fake_gals)[1]+len(q)-4  ))
		new_fake_gals[:,0:4] = np.copy(fake_gals) # copy ra,dec, magch2,magch1
		# Now, we have three catalogs: the fake input sources, the recovered total simulated catalog that includes
		# those sources, and the original catalog without fake sources.
		# For each input fake sources, we do the following:
		for i in range(np.shape(fake_gals)[0]):
			# aux1 and aux2 and index arrays of those sources in simulated and original catalogs
			# that are closer than a certain distance to the fake source
			aux1 = np.where(dist(new_fake_gals[i], original_cat) < 6.0)[0]
			aux2 = np.where(dist(new_fake_gals[i], cat) < 6.0)[0]
			# We record the sizes of these arrays in the control array. Note that aux1 are indices of original_cat
			# while aux2 are indices of cat
			new_fake_gals[i, q['Norig']] = aux1.size
			new_fake_gals[i, q['Nsim']] = aux2.size
			# remove matches
			aux1 , aux2 = remove_matches(aux1, aux2)
			# We focus on these sources found within such circle. If the simulated catalog has more sources
			# there than the original one, it means that we have detected the fake source. 
			if (aux2.size >=1) and (aux1.size==0): 
				max_arg = np.argmax(cat[aux2, q['ch2_3']])# done in 3''flux. Choose brightest
				# For detected sources, we register it as "1"
				new_fake_gals[i, q['det']] = 1
				new_fake_gals[i, q['id']] = cat[aux2[max_arg],cdic['number']] # record id number
				# We record the measured magnitude and error.
				# If fake source "splits" into several detections, record one with minimum CH2 3'' mag
				new_fake_gals[i, q['ch2_3']] = cat[aux2[max_arg],cdic['ch2_3']] # mag2(3'')
				new_fake_gals[i, q['ch2_4']] = cat[aux2[max_arg], cdic['ch2_4']] # mag2(4'')
				new_fake_gals[i, q['ch2_5']] = cat[aux2[max_arg], cdic['ch2_5']] # mag2(5'')
				new_fake_gals[i, q['ch2_6']] = cat[aux2[max_arg], cdic['ch2_6']] # mag2(6'')
				new_fake_gals[i, q['ch2_12']] = cat[aux2[max_arg], cdic['ch2_12']] # mag2(12'')
				new_fake_gals[i, q['ch1_3']] = cat[aux2[max_arg], cdic['ch1_3']] # mag1(3'')
				new_fake_gals[i, q['ch1_4']] = cat[aux2[max_arg], cdic['ch1_4']] # mag1(4'')
				new_fake_gals[i, q['ch1_5']] = cat[aux2[max_arg], cdic['ch1_5']] # mag1(5'')
				new_fake_gals[i, q['ch1_6']] = cat[aux2[max_arg], cdic['ch1_6']] # mag1(6'')
				new_fake_gals[i, q['ch1_12']] = cat[aux2[max_arg], cdic['ch1_12']] # mag1(12'')
				# Record distance between input fake and recovered source
				new_fake_gals[i, q['dist']] = dist(new_fake_gals[i] ,cat[aux2[max_arg]] )
				# This is just an identifier for the specific method of detection
				new_fake_gals[i, q['method']] = 10
			# In this case, there were already more original sources in addition to the detected fake one.
			if (aux2.size > aux1.size) and (aux1.size > 0):
				# Record detection
				new_fake_gals[i, q['det']] = 1
				# We have to figure out which one of the detected sources is the match to the fake, among
				# the other original ones. 
				# Take the two closest detected distances to the fake input source in cat. Take the 
				# closest distance to that fake source in original cat. You have two candidates
				# in cat, to match with the input fake and the one closest original. If the original
				# source is brighter than the input fake one, then assume that the closest to the original 
				# is indeed the match to the original, and thus we match the fake source to the
				# candidate that is further from the original. If the fake source is brighter than the
				# original, then we match the fake to its closest candidate.
				#
				# Find id's to the 2 closest candidates to the fake source
				idx1_cat = aux2[ np.argmin( dist(new_fake_gals[i],cat[aux2]))]
				# If there is only one candidate, set id2=id1
				if aux2.size==1: idx2_cat = idx1_cat 
				else:
					new_aux2 = np.delete(aux2,np.where(aux2==idx1_cat)[0][0])
					idx2_cat = new_aux2[ np.argmin( dist(new_fake_gals[i],cat[new_aux2]))]
				# Find id of closest original source to fake
				idx1_ocat = aux1[ np.argmin( dist(new_fake_gals[i],original_cat[aux1]))]
				# If original is brighter than fake, it picks out matches respect to its own relative
				# distance to candidates. Note that we apply the mag_bias on the original source.
				o_flux = flux_bias(original_cat[idx1_ocat, odic['ch2_3']],'ap3')#-0.6166#mag_bias('output2input',original_cat[idx1_ocat,3],'ap3','ch2')# done in 3''mag
				if o_flux > mag2flux(new_fake_gals[i, q['ch2in']],2): 
					carg = np.argmax( dist(original_cat[idx1_ocat], cat[[idx1_cat, idx2_cat ]]))
				# If fake source is brighter, then we match it to its closest candidate
				else:
					carg = np.argmin( dist(new_fake_gals[i], cat[[idx1_cat, idx2_cat ]]))
				# Determine id of winner candidate
				cidx = [idx1_cat, idx2_cat ][carg]
				# Record values
				new_fake_gals[i, q['id']] = cat[cidx, cdic['number']] # record id number
				new_fake_gals[i, q['ch2_3']] = cat[cidx, cdic['ch2_3']] # mag2(3'')
				new_fake_gals[i, q['ch2_4']] = cat[cidx, cdic['ch2_4']] # mag2(4'')
				new_fake_gals[i, q['ch2_5']] = cat[cidx, cdic['ch2_5']] # mag2(5'')
				new_fake_gals[i, q['ch2_6']] = cat[cidx, cdic['ch2_6']] # mag2(6'')
				new_fake_gals[i, q['ch2_12']] = cat[cidx, cdic['ch2_12']] # mag2(12'')
				new_fake_gals[i, q['ch1_3']] = cat[cidx, cdic['ch1_3']] # mag1(3'')
				new_fake_gals[i, q['ch1_4']] = cat[cidx, cdic['ch1_4']] # mag1(4'')
				new_fake_gals[i, q['ch1_5']] = cat[cidx, cdic['ch1_5']] # mag1(5'')
				new_fake_gals[i, q['ch1_6']] = cat[cidx, cdic['ch1_6']] # mag1(6'')
				new_fake_gals[i, q['ch1_12']] = cat[cidx, cdic['ch1_12']] # mag1(12'')
				new_fake_gals[i, q['dist']] = dist(new_fake_gals[i] ,cat[cidx] )
				new_fake_gals[i, q['method']] = 20
			# Now, we consider the more complicated case of blended sources. The input fake stars blend
			# with original ones, and we recover equal or less sources in the simulation. However, there can be
			# additional sources that have not been blended. Thus, it is important to figure out which sources have
			# been blended and which haven't. We approach this by trying to match sources in the original and simulated
			# catalog. Those matches are, obviously, original sources, so we will disregard them.
			if (aux1.size >= aux2.size) and (aux2.size > 0):
				# find closest cat source to fake.
				idx1_cat = aux2[ np.argmin( dist(new_fake_gals[i],cat[aux2]))]
				# Assuming this is the match to the fake one, is the stuff it has blended with dimmer
				# than the fake? Note that we apply the mag_bias on the fake source.
				cat_flux = flux_bias(cat[idx1_cat, cdic['ch2_5']],'ap5')#-0.164#mag_bias('output2input',cat[idx1_cat,7],'ap5','ch2')
				flux_ratio = mag2flux(new_fake_gals[i, q['ch2in']],2)/ cat_flux # 
				if (flux_ratio > 0.5) and (flux_ratio < 2):# set also a dim limit
						new_fake_gals[i, q['det']]=1
						new_fake_gals[i, q['id']] = cat[idx1_cat, cdic['number']] # record id number
						new_fake_gals[i, q['ch2_3']] = cat[idx1_cat, cdic['ch2_3']] # mag2(3'')
						new_fake_gals[i, q['ch2_4']] = cat[idx1_cat, cdic['ch2_4']] # mag2(4'')
						new_fake_gals[i, q['ch2_5']] = cat[idx1_cat, cdic['ch2_5']] # mag2(5'')
						new_fake_gals[i, q['ch2_6']] = cat[idx1_cat, cdic['ch2_6']] # mag2(6'')
						new_fake_gals[i, q['ch2_12']] = cat[idx1_cat, cdic['ch2_12']] # mag2(12'')
						new_fake_gals[i, q['ch1_3']] = cat[idx1_cat, cdic['ch1_3']] # mag1(3'')
						new_fake_gals[i, q['ch1_4']] = cat[idx1_cat, cdic['ch1_4']] # mag1(4'')
						new_fake_gals[i, q['ch1_5']] = cat[idx1_cat, cdic['ch1_5']] # mag1(5'')
						new_fake_gals[i, q['ch1_6']] = cat[idx1_cat, cdic['ch1_6']] # mag1(6'')
						new_fake_gals[i, q['ch1_12']] = cat[idx1_cat, cdic['ch1_12']] # mag1(12'')
						new_fake_gals[i, q['dist']] = dist(new_fake_gals[i] ,cat[idx1_cat] )
						new_fake_gals[i, q['method']] = 30
						#
		if cov_option == 'cov':
			# load segmentation and coverage images, and for each fake source record average coverage 
			cov = pyfits.open( cov_ch2_image )[0].data.copy()
			covhead = pyfits.open( cov_ch2_image )[0].header
			pobject = pywcs.WCS(covhead)
			seg = pyfits.open( sim_path+ 'I2_check_seg.'+tile+'.fits')[0].data.copy()
			# for detected sources, use segmentation map to identify pixels to take coverage from
			source_array = np.where(new_fake_gals[:, q['det']]==1)[0]
			def func(i): 
				w = np.where(np.ravel(seg) == new_fake_gals[source_array[i],q['id']]  )[0] 
				return np.mean(np.ravel(cov)[w])
			#
			dump_array = Pool(processes=6).map(func,range(source_array.size) )
			new_fake_gals[source_array,q['cov']] = dump_array
			#for i in np.where(new_fake_gals[:,15]==1)[0]:
				#get the segmentation pixels that correspond to each detected source, as a flattened array
				#pixels = np.where(np.ravel(seg) == new_fake_gals[i,4]  )[0] 
				# record mean coverage of those same pixels in the coverage image
				#new_fake_gals[i,5] = np.mean(np.ravel(cov)[pixels])
			# Undetected sources obviosly don't have a segmentation map. However, we can estimate one. The idea
			# is to take the average segmentation size of the detected sources for a given magnitude.
			# With that amount of pixels, we create a circular region to take the coverage values from. 
			# This averaging of the sizes of detected sources is not run for all of them, since that would take too long. 
			# We just take a sample of ~100, and only once per magnitude iteration (at k=0).
			# The objects radius_px and stamp_mask are created
			# here and become global, until they are overriden with the next magnitude iteration (at k=0)
			if k==0:
				detected = np.where(new_fake_gals[:,q['det']]==1)[0]
				pixels_per_source = np.zeros(0)
				for i in detected[:100]:#get the NUMBER of segmentation pixels that correspond to each detected source
					pixels_per_source = np.append(pixels_per_source, np.where(np.ravel(seg) == new_fake_gals[i,4])[0].size)
				# get the average number of those pixels
				avg_pixels_per_source = np.median(pixels_per_source)
				os.system( 'echo At magnitude '+str(ch2_mag)+' the average segmentation size is '+str(avg_pixels_per_source)+' , with scatter of '+str(np.std(pixels_per_source))+'  pixels >> log'+tile )
				# determine an equivalent radius of this chunk of pixels
				radius_px = int(np.sqrt( avg_pixels_per_source /np.pi))
				# create a circular mask with that radius
				stamp_mask = np.ones((2*radius_px+1 , 2*radius_px+1 ))
				for ii in range(2*radius_px+1):
					for jj in range(2*radius_px+1):
						if np.sqrt((ii-radius_px)**2+(jj-radius_px)**2)>radius_px:  stamp_mask[ii,jj]=0.0
			# Now that we know the estimated segmentation stamp for the non-detections, we will proceed to apply it on the coverage image
			for i in np.where(new_fake_gals[:,q['det']]==0)[0]:
				# get pixel value of fake input position for each non-detection
				centralpix = np.array(pobject.wcs_sky2pix( [new_fake_gals[i,:2]]   , 1))[0]
				# convert from ds9 to python indexing
				centralpix = (centralpix-.5).astype('int')
				# crop a square stamp from the coverage image, centered in centralpix. Note that x-axis in python goes in second position
				stamp = cov[centralpix[1] - radius_px : centralpix[1] + radius_px+1, centralpix[0] - radius_px : centralpix[0] + radius_px+1]
				# apply the circular mask and take the mean of the coverage
				stamp = stamp*stamp_mask
				new_fake_gals[i, q['cov']] = np.mean( stamp[np.nonzero(stamp)])
			cov=0; seg=0; # release memory
		# Here we stack the control array from each simulation [16.]#"new_fake_gals" to the general control array
		control = np.vstack((control, new_fake_gals))
		control = np.delete(control, 0 , axis=0)# eliminate first row of zeroes
		#remove detections with dist>3''
		control[control[:, q['dist']]>3.0, q['det']] = 0.0
		np.savetxt(sim_path+'aux_control_dual.'+tile+'.dat',control,fmt=(fmt_string))
		os.system('cat '+sim_path+'aux_control_dual.'+tile+'.dat >> '+sim_path+'control_dual.'+tile+'.dat')
	os.system( 'echo magnitude '+str(ch2_mag)+' done by time '+str(np.floor(time.time()-start_time))+' seconds >> log'+tile )


end_time = time.time()
print "Elapsed time: ", np.floor(end_time-start_time), " seconds"


