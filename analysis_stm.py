from __future__ import division, unicode_literals, print_function 
import sys
import trackpy as tp
import sxmreader
import importlib
from collections import namedtuple
from sklearn.linear_model import LinearRegression
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats
from scipy.stats import linregress
from scipy.optimize import curve_fit
import imageio
import seaborn as sns
import itertools
#%matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  
import os
from ntpath import basename
from shutil import copyfile
import pims
import glob
from sxmreader import SXMReader
import yaml
from sklearn.cluster import KMeans
from matplotlib.colors import BoundaryNorm
import pySPM as spm 
import scipy.misc
from scipy import ndimage
from frame_correct import Frame_correct_loop


{
    "tags": [
        "hide-input",
    ]
}

class MotionAnalyzer:
    """
    A class used to analyze motion of particles on a series of images. Note that some attributes
    are only populated after running analyze(). Originally written to run on sets of voltages over a given gate, which is why there
    are seemingly unnecessary lists everywhere.

    Attributes
    ----------
   ANALYSIS_FOLDER : str
    The folder where analysis results will be saved.

    DIFFUSION_TIME : float
        Diffusion time parameter from the configuration.
    
    MOVIE_FOLDER : str
        Folder where movie outputs will be saved.
    
    PARAMS : list[Params]
        List of parameter sets (one per voltage/temperature).
        
    SXM_PATH : list[list[str]]
        Paths to the SXM image files for each frame set.
    
    SET_NAME : str
        Name of the dataset set, derived from file range.
    
    PARAMS_FILENAME : str
        File name for saving/loading parameter settings.

        
    ## Initialized/nonempty after analyze():##

    D_constants : list[float]
        Diffusion constants calculated from displacement variance. (I don't like this method personally)
    
    D_constants2 : list[float]
        Diffusion constants calculated from MSD slope. See plot_msd in DiffusionPlotter for more precise fitting.
    
    displacements : list[float]
        Displacement values for all tracked particles.
    
    drift_correction : bool
        Whether ensemble drift correction is enabled: subtracts ensemble drift.
    
    drifts : list[pandas.DataFrame]
        Ensemble drift vectors subtracted from the data.
    
    em : list[pandas.DataFrame]
        Ensemble Mean Square Displacements (from `trackpy.emsd`).

    ed : list[list[pandas.Series]]
        Ensemble-averaged displacements along x and y axes, from em.
    
    fileranges : list[range]
        Ranges of file indices corresponding to each image set.
    
    frame_drift_par : list[dict] or None
        Parameters for frame-by-frame image stabilization.
        
    frames : self.frames : list[list[np.ndarray]]
        Image frames analysis is performed on.
    
    heater : bool
        Whether temperature (instead of voltage) is used as label.
    
    initial_size : int
        Original image width in pixels.
    
    msd_intercept : list[float]
        Intercepts of MSD fit curves.
    
    msd_slope : list[float]
        Slopes of MSD fit curves (used to compute D).
    
    mu_hats : list[pandas.Series]
        Average displacement per frame.
    
    removed_particles : list[list[int]]
        Indices of particles to exclude from analysis.
    
    results : list[str]
        Collection of many attributes names used for lookup.
    
    rotated_D_constants : list[float]
        Diffusion constants of rotated particles.
    
    rotation_check : bool
        Enables rotation tracking and debugging.
    

    t3s : list[DataFrame]
        Final dataframe from trackpy. Measurements are in pixels.
    
    tear_correct : list[list[int]]
        List of visual tear corrections by [frame, row]. e.g. [[[1,2],[3,8]],[[6,45]]], 
        has tears in the first data set on frame 1 at row 2, frame 3 at row 8, and the second dataset
        on frame 6 row 45.

    total_molecules : list[float]
        Total number of molecules tracked in each set.
    
    total_moved : list[float]
        Total number of particles that moved.
    
    total_rotated : list[float]
        Total number of rotated particles.
    
    total_translated : list[float]
        Total number of translated particles.
    
    tracked_particles : pandas.DataFrame
        Filtered DataFrame of tracked particles.
    
    translated_D_constants : list[float]
        Diffusion constants of translated particles.
    
    v_drift_mag : list[float]
        Magnitudes of average drift vectors per frame.
    
    voltages_temperatures : list[float]
        Voltages or temperatures corresponding to each dataset.
    
    NM_PER_PIXEL : float
        Conversion factor from pixels to nanometers.

    Note: All attributes in results also have a version with _C at the end (i.e self.t3s_C).
    self.attribute_C[0] is with the ensemble drift subtracted,  self.attribute_C[1] is without.


    Methods
    -------
    analyze():
        Performs batch particle tracking and linking of tracks, calculates diffusion constants
        and drifts.

    Correct_drift():
        Switches between results with and without ensemble drift correction.
    """
           
    def __init__(
        self, 
        fileranges=None, 
        voltages_temperatures=None, 
        folder_name = None, 
        heater = False, 
        drift_correction = False, 
        frame_drift_par = None,
        correct=None,
        rotation_check=True,):
        
        if isinstance(fileranges,type(None)):
            print('No file range specified.')
            return
        self.frame_drift_par=frame_drift_par
        self.heater = heater
        self.drift_correction = drift_correction
        self.fileranges = fileranges
        self.voltages_temperatures = voltages_temperatures    
        self.SXM_PATH = [[folder_name + "/Image_{0:03}.sxm".format(i) for i in fileranges[j]] for j in range(len(fileranges))]
        self.SET_NAME = "{}-{}/".format(min([min(x) for x in fileranges]), max([max(x) for x in fileranges]))
        self.ANALYSIS_FOLDER = "./analysis/" + folder_name + "_" + self.SET_NAME
        self.MOVIE_FOLDER = self.ANALYSIS_FOLDER + "movies/"
        self.PARAMS_FILENAME = "params.yaml"
        if not os.path.exists(self.ANALYSIS_FOLDER):
            os.makedirs(self.ANALYSIS_FOLDER)
        if not os.path.exists(self.ANALYSIS_FOLDER):
            os.makedirs(self.MOVIE_FOLDER)
        self._set_search_params()
        self.correct=correct
        self.rotation_check=rotation_check
        ## tracked particles
        self.removed_particles = [[]]*self.voltages_temperatures.size
        self.tear_correct= [[]]*self.voltages_temperatures.size
        # list of list of line tears. Each sublist holds tuples of form (frame_number,y_position_of_tear)

    def analyze(self, plot_gif=False,refilter=False):

        #probably better to make this a dictionary
        
        self.drifts = []
        self.v_drift_mag = []
        self.D_constants = []
        self.D_constants2 = []
        self.msd_slope = []
        self.msd_intercept = []
        self.mu_hats = []
        self.ed = []
        self.em = []
        self.displacements = []
        self.testframe = []
        self.total_molecules = []
        self.total_moved = []
        self.total_rotated= []
        self.total_translated = []
        self.rotated_D_constants = []
        self.translated_D_constants = []
        self.t3s=[]

       #Drift_Correction vs Not
        self.v_drift_mag_C = [[],[]]
        self.D_constants_C = [[],[]]
        self.D_constants2_C = [[],[]]
        self.msd_slope_C = [[],[]]
        self.msd_intercept_C = [[],[]]
        self.mu_hats_C = [[],[]]
        self.ed_C = [[],[]]
        self.em_C = [[],[]]
        self.displacements_C = [[],[]]
        self.total_molecules_C = [[],[]]
        self.total_moved_C = [[],[]]
        self.total_rotated_C= [[],[]]
        self.total_translated_C = [[],[]]
        self.rotated_D_constants_C = [[],[]]
        self.translated_D_constants_C = [[],[]]
        
        ts=[]
        if not refilter:
            self.frames = []

            self.t3s_C=[[],[]]

            for i, path in enumerate(self.SXM_PATH):
                molecule_size, min_mass, max_mass, separation, min_size, max_ecc, adaptive_stop, search_range,threshold, _ = self.PARAMS[i]
    
                frames = SXMReader(path, correct=self.correct)
                self.initial_size=frames.shape[1]
     
                self.NM_PER_PIXEL = frames.meters_per_pixel * 1e9 
                #This part is to change it from pims sequence to arrays to do tear correction because pims immutable
                frames_=[]
                for frame in frames:
                    frames_.append(frame)
                frames=frames_
    
                for correction in self.tear_correct[i]:
                    frame_new=np.array(frames[correction[0]])
                    try:
                        frame_new[correction[1],:]=(frame_new[correction[1]+1,:]+frame_new[correction[1]-1,:])/2
                    except:
                        try:
                            frame_new[correction[1],:]=frame_new[correction[1]+1,:]
                        except:                     
                            frame_new[correction[1],:]=frame_new[correction[1]-1,:]

                        
                    frame_new = (frame_new - frame_new.min()) / (frame_new.max() - frame_new.min())  # scale to [0, 1]
                    frame_new = frame_new * 2 - 1    
                    frame_new = pims.Frame(frame_new, frame_no=correction[0])
                    frames[correction[0]]=frame_new
                if self.frame_drift_par[i]!= None:
                    frames=Frame_correct_loop(frames,self.PARAMS[i],**self.frame_drift_par[i])
                
                self.frames.append(frames)
                f = tp.batch(frames, molecule_size, minmass=min_mass,threshold=threshold,separation=separation,engine='python')
                t = tp.link(f, search_range=search_range, adaptive_stop=adaptive_stop,memory=0)
                ts.append(t)
                if plot_gif == True:
                    moviename = "{}-{}".format(min(self.fileranges[i]), max(self.fileranges[i]))
                    singlemoviefolder = self.MOVIE_FOLDER + moviename + "/"
                    if not os.path.exists(singlemoviefolder):
                        os.makedirs(singlemoviefolder)
                    mpl.rcParams.update({'font.size': 10, 'font.weight':'bold'})
                    mpl.rc('image', origin='lower')
                    mpl.rc('text',usetex =False)
                    mpl.rc('text',color='black')
    
    
                    fns = []
                    for j, frame in enumerate(frames):
                        self.testfig= plt.figure(figsize=(5,5))
                        plt.axis('off')
                        self.testframe = tp.plot_traj(t2[(t2['frame']<=j)], superimpose=frames[j], label=False, origin='lower')
                        self.testframe.invert_yaxis()
                        self.testfig=self.testframe.figure
                        fn = singlemoviefolder + "Image_{}.png".format(self.fileranges[i][j])
                        
                        self.testfig.savefig(fn,bbox_inches='tight')
                        fns.append(fn)
                        plt.clf()
                        plt.axis('off')
    
                    mpl.rc('text',color='black')
                    images = []
                    for fn in fns:
                        images.append(imageio.imread(fn))
                    imageio.mimsave(singlemoviefolder + moviename + '.gif', images, duration=0.5)
            t2s=self.filter_particles(ts,)
        else:
            t2s=self.filter_particles(self.t3s_C[0],)
            self.t3s_C=[[],[]]

        for index,t2 in enumerate(t2s):
            # Compute drifts
            d = tp.compute_drift(t2)
            d.loc[0] = [0, 0]
            t3 = t2.copy()
            t3_C = tp.subtract_drift(t2) #Subtract the drift
            t_both=[t3,t3_C]
            self.t3s_C[0].append(t3)
            self.t3s_C[1].append(t3_C)
            self.drifts.append(d)
            

            # Method 1 of calculating D: variance of all displacements of Delta_t=1
            for i in [0,1]:
                displacements = self._calculate_displacements(t_both[i])
                self.displacements_C[i].append(displacements)
                
                self.D_constants_C[i].append((displacements.dx.var() + displacements.dy.var()) / 4/ self.DIFFUSION_TIME) # r^2 = x^2 + y^2 = 2Dt + 2Dt
                self.mu_hats_C[i].append(np.mean(displacements[['dx', 'dy']], axis=0))
                # Compute number of rotated molecules
                if self.rotation_check:

                    rotated = displacements.rotated.sum()
                    moved = len( displacements[displacements.dr > 0.1].index)

                    total_molecules = len(displacements.index)
                    self.total_molecules_C[i].append(total_molecules)
                    self.total_moved_C[i].append(moved)
                    self.total_rotated_C[i].append(rotated)
                    self.total_translated_C[i].append(moved-rotated)
                    # Compute D separately for rotated and translated molecules
                    rotated_displacements = displacements[(displacements.rotated==True)]
                    translated_displacements = displacements[(displacements.rotated==False)]
                    self.rotated_D_constants_C[i].append((rotated_displacements.dx.var() + rotated_displacements.dy.var()) / 4/ self.DIFFUSION_TIME)
                    self.translated_D_constants_C[i].append((translated_displacements.dx.var() + translated_displacements.dy.var()) / 4/ self.DIFFUSION_TIME) 
                #   Method 2 of calculating D: linear fit to MSD with weights
                em = tp.emsd(t_both[i], self.NM_PER_PIXEL, 1/self.DIFFUSION_TIME, max_lagtime=len(self.frames[index]) ,detail=True)
                self.em_C[i].append(em)
                self.ed_C[i].append([em['<x>'],em['<y>']])
                X = em.index * self.DIFFUSION_TIME
                X = X.values.reshape(-1, 1)
                w = em.N
                y = em['msd']
                model = LinearRegression().fit(X, y, sample_weight = w)
                self.msd_slope_C[i].append(model.coef_[0])
                self.msd_intercept_C[i].append(model.intercept_)
                self.D_constants2_C[i].append(model.coef_[0]/4)
            
        self.v_drift_mag_C[0]= np.linalg.norm(self.mu_hats_C[0], 2, axis=1)
        self.v_drift_mag_C[1]= np.linalg.norm(self.mu_hats_C[1], 2, axis=1)

        if self.drift_correction:
            drift=1
        else:
            drift=0
         #really should make it a dictionary
        self.results=[
            'v_drift_mag',
            'D_constants',
            'D_constants2',
            'msd_slope',
            'msd_intercept',
            'mu_hats',
            'ed',
            'em', 
            'displacements',
            'total_molecules',
            'total_moved',
            'total_rotated',
            'total_translated',
            'rotated_D_constants',
            'translated_D_constants',
            't3s']
        self.results_C=[
            self.v_drift_mag_C ,
            self.D_constants_C ,
            self.D_constants2_C ,
            self.msd_slope_C ,
            self.msd_intercept_C ,
            self.mu_hats_C ,
            self.ed_C ,
            self.em_C ,
            self.displacements_C ,
            self.total_molecules_C ,
            self.total_moved_C ,
            self.total_rotated_C,
            self.total_translated_C,
            self.rotated_D_constants_C ,
            self.translated_D_constants_C ,
            self.t3s_C]
        for index, item in enumerate(self.results):
            setattr(self, item, self.results_C[index][drift])
    def filter_particles(self,ts,):
        t2s=[]
        for i,t in enumerate(ts):
                print(f'Set {i}')

                molecule_size, min_mass, max_mass, separation, min_size, max_ecc, adaptive_stop, search_range,threshold, _ = self.PARAMS[i]
                delta=self.initial_size/2-molecule_size #removes fake particles caused by boundary
                center=self.frames[i][0].shape[1]/2
                print('Initial:', t['particle'].nunique())

                t_=t[t['frame']==0]
                outside_x=np.abs(t_['x']-center)>delta
                outside_y=np.abs(t_['y']-center)>delta
                bad_y=t_[outside_y]['particle']
                bad_x=t_[outside_x]['particle']
                t=t[~t['particle'].isin(bad_y)]     
                t=t[~t['particle'].isin(bad_x)]
                t_=t[t['frame']==0]
                inside_x=np.abs(t_['x']-center)<delta
                inside_y=np.abs(t_['y']-center)<delta
                good_y=t_[inside_x]['particle']
                good_x=t_[inside_y]['particle']
                t=t[t['particle'].isin(good_y)]     
                t=t[t['particle'].isin(good_x)] 

                t1 = t[((t['mass'] > min_mass) & (t['size'] > min_size) &
                     (t['ecc'] < max_ecc)) & (t['mass'] < max_mass)]
            
                if not hasattr(self, "removed_particles"):
                    self.removed_particles = [[]]*self.voltages_temperatures.size
                #Filter out particles
                self.tracked_particles = t1[~t1['particle'].isin(self.removed_particles[i])]

                # Apply filter_stubs to the updated tracked_particles instead of t1
                stubs_max=int(np.shape(self.frames[i])[0]/3) #filters out trajectories with smaller frame lifetime than stubs_max
                t2 = tp.filter_stubs(self.tracked_particles, stubs_max)
                t2s.append(t2)
                print('starting frame particles filter:', t['particle'].nunique())
                print('Parameter filter:', t1['particle'].nunique())
                print('Specified Removed Particle filter:', self.tracked_particles['particle'].nunique())
                print('Short trajectory filter:', t2['particle'].nunique())
        return t2s
    def CorrectDrift(self,drift_correction=True):
        self.drift_correction=drift_correction
        if drift_correction:
            drift=1
        else:
            drift=0
        for index, item in enumerate(self.results):
            setattr(self, item, self.results_C[index][drift])
    
            
    def _cleanup_png(self, singlemoviefolder):
        filelist = glob.glob(os.path.join(singlemoviefolder, "*.png"))
        for f in filelist:
            os.remove(f)
            
    def _snap_to_orientation(self, angles):
        kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(angles).reshape(-1, 1))
        idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(3)
        return lut[kmeans.labels_]
                        
    def _calculate_displacements(self, t, delta=1):
        displacements = pd.DataFrame()
        for j in range(t.frame.max()+1 - delta):
            displacements = pd.concat([displacements, tp.relate_frames(t, j, j + delta)], ignore_index=True)
            
           
        displacements = displacements.dropna()
        physical_columns = ['x', 'y', 'x_b', 'y_b', 'dx', 'dy', 'dr']
        displacements[physical_columns] *= self.NM_PER_PIXEL
        offset_theta = -15/180*np.pi
        if self.rotation_check:
            displacements['orientation'] = self._snap_to_orientation(displacements.angle)
            displacements['orientation_b'] = self._snap_to_orientation(displacements.angle_b)
            displacements["rotated"] = (displacements['orientation']!=displacements['orientation_b']).astype("int")
        return displacements
    def _set_search_params(self):
        with open('params.yaml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)       
        Params = namedtuple(
                    'Params', 
                    ['molecule_size', 
                     'min_mass',
                     'max_mass',
                     'separation',
                     'min_size',
                     'max_ecc',
                     'adaptive_stop',
                     'search_range',
                     'threshold',
                     'diffusion_time',])
        self.DIFFUSION_TIME = params['diffusion_time']
        self.PARAMS = [Params(**params) for i in range(len(self.voltages_temperatures))]
        copyfile(self.PARAMS_FILENAME, self.ANALYSIS_FOLDER + self.PARAMS_FILENAME)



class ExpMetaData:
    
    def __init__(self, sets=None, Vg = None, voltages_temperatures=None, folder = None):
            self.sets = sets
            self.Vg =Vg
            self.voltages_temperatures = voltages_temperatures
            self.folder = folder
            return None

from sklearn.neighbors import NearestNeighbors
from matplotlib import patches

class DiffusionPlotter(MotionAnalyzer):
    
    """
    A class used to plot diffusive motion of particles using results from a MotionAnalyzer.
    
    Attributes
    ----------
   Inherits all attributes of MotionAnalyzer.
   
    Ea : float
        Activation energy extracted from an Arrhenius fit on the main dataset (`D_constants`).
        Initialized in `__init__`, updated in `plot_diffusion()`.

    Ea_err : float
        Standard error associated with `Ea`, computed during nonlinear curve fitting.
        Set in both `__init__` and updated in `plot_diffusion()`.

    Ea_rotated : float
        Activation energy extracted from a rotated-frame diffusion dataset.
        Initialized in `__init__`, and updated in `plot_rotated_and_translated_diffusion()`.

    Ea_err_rotated : float
        Error on `Ea_rotated`. Assigned in `plot_rotated_and_translated_diffusion()`.

    Ea_translated : float
        Activation energy computed from translated-frame diffusion data.
        Set in `plot_rotated_and_translated_diffusion()`.

    Ea_err_translated : float
        Standard error of the translated-frame activation energy.
        Set in `plot_rotated_and_translated_diffusion()`.

    C0 : float
        Pre-exponential factor from the main Arrhenius fit: D = C0 * exp(-Ea / kT).
        Set in `__init__`, updated in `plot_diffusion()`.

    C0_err : float
        Fitting error associated with `C0`. Assigned in `__init__`.

    Ea2 : float
        Activation energy from a second dataset (`D_constants2`), fit in parallel to `Ea`.
        Initialized in `__init__`, updated in `plot_diffusion()`.

    Ea2_err : float
        Error estimate for `Ea2`. Set during second fit in `plot_diffusion()`.

    C02 : float
        Pre-exponential factor for second Arrhenius fit (associated with `Ea2`).
        Set in `__init__` and updated in `plot_diffusion()`.

    n_frame_displacements : list of pandas.DataFrame
        Populated using `self._calculate_displacements(...)` during analysis.
        Stores per-frame displacement data used in MSD and diffusion constant calculation.

    rots : list
        Filled by appending values during rotational analysis loops.
        Likely stores applied rotation angles or transformations.

    trunc_lattice : list
        Initialized as empty. Intended to store truncated lattice vectors for symmetry analysis.

    btw_trunc_lattice : list
        Like `trunc_lattice`, initialized but not filled in this class.
        Likely used for intermediate lattice spacing or neighbor relationships.

    a : float
        Lattice constant or characteristic step length, used to normalize displacement data.

    hexbin_counts : list of pandas.DataFrame
        Populated with spatially binned statistics (e.g. jump direction or density)
        using `pd.concat(...)` during plotting.

    mStyles : list of str or int
        Matplotlib marker styles used to distinguish data series in plotting routines.

    cmap : matplotlib.colors.Colormap
        Colormap object initialized from a matplotlib colormap (e.g. `plt.cm.*`).
        Used to assign colors by data group.

    colors : list or ndarray
        Color values sampled from `cmap` during `__init__` for visualizing categories.

    tpmv : list of float
        Sorted list of `self.voltages_temperatures`, aligned with `D_constants`.
        Used as the x-axis in Arrhenius and other diffusion plots.
        Assigned in `__init__`.

    _sorted_D_constants : list of float
        Diffusion constants sorted in correspondence with `tpmv`.
        Used in nonlinear fitting for activation energy extraction.
        Assigned in `__init__`.


    Methods
    -------
    plot_drift_data()
    plot_diffusion()
    plot_msd()
    plot_ed()
    plot_v_over_D()
    
    """
    
    # Optionally, tweak styles.
    rc('animation', html='html5')
    mpl.rc('figure',  figsize=(10, 10))
    mpl.rc('image', cmap='gray')
    mpl.rc('image', origin='lower')
    mpl.rc('text',color='black')
    #mpl.rcParams.update({'font.size': 24, 'font.weight':'bold'})
    
    def __init__(self, ma: MotionAnalyzer):
        self.__dict__ = ma.__dict__.copy()
        self.Ea = 0
        self.Ea_err = 0
        self.Ea_rotated = 0
        self.Ea_err_rotated = 0
        self.Ea_translated = 0
        self.Ea_err_translated = 0
        self.C0 = 0
        self.C0_err = 0
        self.Ea2 = 0
        self.Ea2_err = 0
        self.C02 = 0
        self.n_frame_displacements = []
        self.rots = []
        self.trunc_lattice = []
        self.btw_trunc_lattice = []
        self.a = 0.246
        self.hexbin_counts = []
        
        for i, voltagei in enumerate(self.voltages_temperatures):
            self.displacements[i]['VSD'] = "{0:.2f}".format(voltagei)
        self._calculate_rot()
        self.trunc_lattice, self.btw_trunc_lattice = self._calculate_lattice_points(self.a)
        
        self.mStyles = ["o","v","^","<",">","s","p","P","*","h","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
        self.cmap = plt.cm.get_cmap("rainbow")
        self.colors = self.cmap(np.linspace(0,1,len(self.voltages_temperatures)))
        tmpv, _sorted_D_constants = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.D_constants))))
        tmpv, _sorted_D_constants2 = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.D_constants2))))
        result = linregress(np.reciprocal(tmpv), np.log(_sorted_D_constants))
        result2 = linregress(np.reciprocal(tmpv), np.log(_sorted_D_constants2))
        self.Ea = -result.slope
        self.Ea_err = result.stderr
        self.C0 = result.intercept
        self.Ea2 = -result2.slope
        self.Ea2_err = result2.stderr
        self.C02 = result2.intercept
        self.tpmv=tmpv
        self._sorted_D_constants=_sorted_D_constants

        
    def _rotate(self, coords, theta):
        x0, y0 = coords.dx, coords.dy
        x1 = x0*np.cos(theta) - y0*np.sin(theta)
        y1 = x0*np.sin(theta) + y0*np.cos(theta)
        return [x1, y1]

    def _calculate_rot(self):
        for test in self.displacements:    
            orientations = sorted(test.orientation.unique())
            rot_1 = test[test.orientation == orientations[0]].reset_index()
            rot_2 = test[test.orientation == orientations[1]].reset_index()
            rot_3 = test[test.orientation == orientations[2]].reset_index()

            ndr = pd.DataFrame(rot_1.apply(self._rotate, args = [-np.pi/3 + np.pi/2], axis=1).to_list())
            rot_1[['dx', 'dy']] = ndr
            ndr = pd.DataFrame(rot_2.apply(self._rotate, args = [-np.pi + np.pi/2], axis=1).to_list())
            rot_2[['dx', 'dy']] = ndr
            ndr = pd.DataFrame(rot_3.apply(self._rotate, args = [-5*np.pi/3 + np.pi/2], axis=1).to_list())
            rot_3[['dx', 'dy']] = ndr
            rot = pd.concat([rot_1,rot_2,rot_3])
            self.rots.append(rot)
     
    def plot_msd(self, ax = None,scale='log',end=None,linearfit=False,intercept=False):
        """Plot mean square displacement (MSD) over time for tracked particles. 
        Parameters
        ----------
        scale : str, optional
            Type of axis scale to use ('linear' or 'log'). Default is 'linear'.
        crop_end : int, optional
            Index at which to truncate the MSD data for fitting. Default is None (use all points).
        linearfit : bool, optional
            Whether to force a linear fit. Default is False.
        intercept: bool, optional
            Whether to add an additional constant to the fitting function. i.e d*x^a+c as opposed
            to the theoretical d*x^a. Default is False.

        """
        fig = plt.figure(figsize=(14,10+0.6*np.size(self.voltages_temperatures)))
        if ax is None:
            ax = plt.gca()
        for a in range(len(self.em)):
            i=(-a-1)%len(self.em)



            if self.heater:
                labeltext = "{:.0f} $K$".format(self.voltages_temperatures[i])
            else:
                labeltext = "{:.2f} $V_S$".format(self.voltages_temperatures[i])
            #p = ax.plot(self.em[i].index * self.DIFFUSION_TIME, self.em[i]['msd']- self.msd_intercept[i], 
                       # label= labeltext, markersize=30, marker= self.mStyles[i], mfc = self.colors[i], mec=self.colors[i],
                        #linestyle='None')
           
            #ax.legend()
            
            ax.set_xscale(scale)
            ax.set_yscale(scale)
   #ax.set_xticks([])
            ax.xaxis.set_major_locator(plt.MultipleLocator(self.DIFFUSION_TIME))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2d'))
            if self.heater:
                
                ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ (nm$^2$)',fontweight='normal')
                ax.set_xlabel('Diffusion Time (s)')
            else:
                ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ (nm$^2$)',
                       xlabel='lag time $t$ (s)')

            if end==None:
                time=self.em[i]['lagt']
                data=self.em[i]['msd']
                uncertainty=1/np.sqrt(self.em[i]['N'])

            else:
                time=(self.em[i]['lagt'])[:end]
                data=(self.em[i]['msd'])[:end]
                uncertainty=1/np.sqrt(self.em[i]['N'])[:end]
     
            x = np.linspace(0, self.DIFFUSION_TIME *
                            len(time)*1.1,100)
            if linearfit:
                if intercept:
                   
                    fitfunc=lambda x,d,inter:d*x+inter
                    variables=['d','intercept']
                    fname='d*x+inter'
                else:
                    fitfunc=lambda x,d:d*x
                    variables=['d']
                    fname='d*x'
            else:
                if intercept:
                    fitfunc=lambda x,d,a,inter:d*(x**a)+inter
                    variables=['d','a','intercept']
                    fname='d*(x^a)+inter'
                else:
                    fitfunc=lambda x,d,a:d*(x**a)
                    variables=['d','a',]
                    fname='d*(x^a)'
            
            par,cov=curve_fit(fitfunc,time, data,sigma=uncertainty)
            print(fr' {labeltext}')
            for index in range(len(variables)):
                print(f'{variables[index]}={par[index]}')
            ax.plot(x,fitfunc(x,*par), '--',  linewidth=6, color = self.colors[i],label=fname)
            if np.sum(data<0)>0.5:
                print('WARNING: A data point exists outside the graph')
            ax.plot(time,data,label= labeltext, markersize=20,marker='o', mfc = self.colors[i], mec=self.colors[i],linestyle='None')   
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #ax.legend(loc='center left')
            ax.minorticks_off()
        ax.set_ylim(bottom=self.msd_slope[0]*x[0]/4)
        ax.set_xlim(left=self.DIFFUSION_TIME)
        plt.tick_params(axis='both', which='major', length=12)
        plt.savefig(self.ANALYSIS_FOLDER + "msd.png",bbox_inches = "tight")
        return fig,par
        
    
    def plot_drift_vectors(self, plotrange = 20, ax = None,paperfigure=False,color1='PuRd',color2='YlOrBr',range1=(0,1),range2=(0,1)):
        """
        Plot average drift vectors of particles as head-to-tail arrows.
    
        For each condition in `self.drifts`, this function draws arrow sequences indicating
        the cumulative motion of particles across frames. The drift vectors are scaled to
        real-world units using `self.NM_PER_PIXEL`.
    
        paperfigure is a parameter we used for a specific paper to seperate positive and negative source drain. 
        I've left the logic in case anyone wants to do something similar.

    
        Parameters
        ----------
        plotrange : float, optional
            Range for x and y axis limits in nanometers. Default is 20.
        ax : matplotlib.axes.Axes, optional
            Axis object to plot on. If None, a new axis is created 
        paperfigure : bool, optional
            If True, use combined colormaps `color1` and `color2` with value ranges `range1` and `range2`.
            This is used to customize colors for our paper. The length of positive and negative source drains
            are hard coded to 7 to match our data. Default is False. 
        color1 : str, optional
            Name of the first colormap to use if `paperfigure=True`. Default is 'PuRd'.
        color2 : str, optional
            Name of the second colormap to use if `paperfigure=True`. Default is 'YlOrBr'.
        range1 : tuple of float, optional
            Value range (start, end) to sample from `color1`. Default is (0, 1).
        range2 : tuple of float, optional
            Value range (start, end) to sample from `color2`. Default is (0, 1).
        """
        plt.figure(figsize=(10, 10))
        if ax is None:
            ax = plt.gca()
        #colors = ['r', 'k', 'b', 'g', 'tab:orange', 'tab:purple', 'm']
        cmap = plt.cm.get_cmap("cool")
        cmap2=plt.cm.get_cmap(color1)
        cmap3=plt.cm.get_cmap(color2)

               
        colors = cmap(np.linspace(0,0.8,len(self.voltages_temperatures)),)
        if paperfigure:
            #colors = cmap(np.concatenate((np.linspace(0.5,0.1,7),np.linspace(0.55,0.9,len(self.voltages_temperatures)))))
            colors = np.concatenate((cmap2(np.linspace(*range1,7)),cmap3(np.linspace(*range2,len(self.voltages_temperatures)-7))))
        arrs = []
        

        for j, d in enumerate(self.drifts):
            #d['x']=-d['x']
            for i in range(1, len(d)):
                d0, d1 = d.loc[i - 1] * self.NM_PER_PIXEL, d.loc[i] * self.NM_PER_PIXEL
                ax.arrow(d0.x,d0.y,d1.x-d0.x, d1.y-d0.y, 
                shape='full', color=colors[j], length_includes_head=True, 
                zorder=0, head_length=0.5, head_width=0.5,linewidth=1.5)
            else:
                d0, d1 = d.loc[i - 1] * self.NM_PER_PIXEL, d.loc[i] * self.NM_PER_PIXEL
                arrs.append(plt.arrow(d0.x,d0.y,d1.x-d0.x, d1.y-d0.y, 
                shape='full', color=colors[j], length_includes_head=True, 
                zorder=0, head_length=0.5, head_width=0.5,linewidth=1.5, label=str(self.voltages_temperatures[j])))
        new_labels, arrs = zip(*sorted(zip(self.voltages_temperatures, arrs)))
        new_labels=["{:.2f}".format(s) + ' V' for s in new_labels]
       
       # plt.gca().add_artist(first_legend)
        #second_legend=ax.legend(arrs[7:], new_labels[7:], fontsize=15,loc='lower left')
        #first_legend=ax.legend(arrs[:7], new_labels[:7], fontsize=15, loc='lower left')
        both_legend=ax.legend(arrs, new_labels, fontsize=15, loc='lower left',ncol=2)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=4, direction='in', length=5, top=True, right=True)
        
        #plt.title("Ensemble Drift, " + SXM_PATH[0][0] + " to {}".format(SXM_PATH[-1][-1]))
        ax.set_xlabel("x (nm)",fontsize=24,fontweight='bold')
        ax.set_ylabel("y (nm)",fontsize=24,fontweight='bold')
        ax.set_xlim(-plotrange, plotrange)
        ax.set_ylim(-plotrange, plotrange)
        ax.set_xticks(np.linspace(-plotrange,plotrange, 5))
        ax.set_yticks(np.linspace(-plotrange,plotrange, 5))

        ax.set_aspect('equal', 'box')
        #plt.savefig(self.ANALYSIS_FOLDER + "drift_vectors.png")
    
    def plot_drift_scalar(self,):
        """Plots drift the magnitude of drift velocity as a function of voltage. 
        
        
        """
                                
    #def _calculate_displacements(self, t, delta=1):
   # displacements = pd.DataFrame()
   # for j in range(t.frame.max() - delta):
   #         displacements = displacements.append(tp.relate_frames(t, j, j + delta) * self.NM_PER_PIXEL, ignore_index=True)
   # displacements = displacements.dropna()
   # offset_theta = -15/180*np.pi
   # displacements['orientation'] = self._snap_to_orientation(displacements.angle)
   # displacements['orientation_b'] = self._snap_to_orientation(displacements.angle_b)
   # displacements["rotated"] = (displacements['orientation']!=displacements['orientation_b']).astype("int")
   # return displacements  
        #LUC mu_hats is : self.mu_hats.append(np.mean(displacements[['dx', 'dy']], axis=0))
            
        
        mag_displace = np.linalg.norm(self.mu_hats, 2, axis=1)
        new_labels, n_mag_displace, ord_D_constants = zip(*sorted(zip(self.voltages_temperatures, mag_displace, self.D_constants)))
        mpl.rcParams.update({'font.size' : 28, 'font.weight' : 'bold'})
        plt.figure(figsize=(10, 10))
        plt.plot(self.voltages_temperatures, mag_displace / self.DIFFUSION_TIME, '-o', markersize=18, linewidth=4)
        # plt.plot(xx, yy / 1.5)
        plt.ylabel('drift velocity (nm / s)')
        plt.xlabel('Voltage (V)')
        plt.title('drift velocity magnitude')
        plt.savefig(self.ANALYSIS_FOLDER + "drift_scalar.png")
        
        plt.figure(figsize=(10, 10))
        mean_mu_hat = self._calculate_mean_axis(self.mu_hats)
        proj_mag_displace = np.array(self._project_to_mean_axis(self.mu_hats,mean_mu_hat))
        plt.plot(self.voltages_temperatures,  proj_mag_displace / self.DIFFUSION_TIME, '-o', markersize=18, linewidth=4)
        plt.ylabel('drift velocity (nm / s)')
        plt.xlabel('Voltage (V)')
        plt.title('drift velocity projected onto average drift direction')
        plt.savefig(self.ANALYSIS_FOLDER + "drift_scalar_projected.png")

    def _label_axes(self, ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    def plot_diffusion(self, suppress_output=False):
        """Fit and plot diffusion constants vs temperature or voltage.

        Sorts and aligns diffusion constants (`D_constants`) with the corresponding temperatures
        or voltages, then fits the data using a linearized Arrhenius equation:
            log(D) = log(C₀) - Ea / x
        where x = temperature or voltage.
        
        This method updates the following attributes:
        - self.tpmv, self._sorted_D_constants
        - self.Ea, self.Ea_err, self.C0
        - self.Ea2, self.Ea2_err, self.C02
        
        Generates and saves two plots: D vs T (or V), and log(D) vs 1/T (or 1/V).
        """
        tmpv, _sorted_D_constants = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.D_constants))))
        tmpv, _sorted_D_constants2 = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.D_constants2))))
        result = linregress(np.reciprocal(tmpv), np.log(_sorted_D_constants))
        result2 = linregress(np.reciprocal(tmpv), np.log(_sorted_D_constants2))
        self.Ea = -result.slope
        self.Ea_err = result.stderr
        self.C0 = result.intercept
        self.Ea2 = -result2.slope
        self.Ea2_err = result2.stderr
        self.C02 = result2.intercept
        self.tpmv=tmpv
        self._sorted_D_constants=_sorted_D_constants
        
        
        #self.C0_err = result.intercept_stderr
        
        if not suppress_output:
            font = {
            'weight' : 'bold',
            'size'   : 22}

            mpl.rc('font', **font)
            mpl.rc('text',usetex =False)
            fig, ax = plt.subplots(figsize=(10,10))
            ax.plot(np.array(tmpv), _sorted_D_constants,'o-')
            #ax.plot(np.array(tmpv), _sorted_D_constants2,'o-')

            if self.heater == True:
                self._label_axes(ax,'Temperature (K)','Diffusion constant ($nm^2$ / s)')
            else:
                self._label_axes(ax,'Voltage (V)','Diffusion constant ($nm^2$ / s)')
            plt.savefig(self.ANALYSIS_FOLDER + "D_constant_exp.png")

            fig1, ax1 = plt.subplots(figsize=(10,10))
            sns.regplot(x=np.reciprocal(tmpv),y=np.log(_sorted_D_constants), marker='o', ci=None, ax=ax1)
            #sns.regplot(x=np.reciprocal(tmpv),y=np.log(_sorted_D_constants2), marker='o', ci=None, ax=ax1)

            #sns.regplot(np.reciprocal(tmpv), np.log(_sorted_D_constants2), 'o-', ci=None, ax=ax1)

            if self.heater == True:
                self._label_axes(ax1,'1/T (1/K)','Log Diffusion constant ($nm^2$ / s)')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{T}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result.slope,slope_stderr=result.stderr,intercept = result.intercept, intercept_stderr=result.intercept_stderr),xy=(350,500), xycoords='figure pixels')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{T}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result2.slope,slope_stderr=result2.stderr,intercept = result2.intercept, intercept_stderr=result2.intercept_stderr),xy=(350,400), xycoords='figure pixels')
            else:
                self._label_axes(ax1,'1/V (1/V)','Log Diffusion constant ($nm^2$ / s)')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{V}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result.slope,slope_stderr=result.stderr,intercept = result.intercept, intercept_stderr=result.intercept_stderr),xy=(350,500), xycoords='figure pixels')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{V}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result2.slope,slope_stderr=result2.stderr,intercept = result2.intercept, intercept_stderr=result2.intercept_stderr),xy=(350,400), xycoords='figure pixels')

            plt.savefig(self.ANALYSIS_FOLDER + "logD_constant_lin.png")
            return fig,fig1
            
    def plot_rotated_and_translated_diffusion(self, suppress_output=False):
        """Same as plot diffusion, but seperates translated and rotated molecules."""
        tmpv, _sorted_rotated_D_constants = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.rotated_D_constants))))
        tmpv, _sorted_translated_D_constants = (list(t) for t in zip(*sorted(zip(self.voltages_temperatures, self.translated_D_constants))))
        result = linregress(np.reciprocal(tmpv), np.log(_sorted_rotated_D_constants))
        result2 = linregress(np.reciprocal(tmpv), np.log(_sorted_translated_D_constants))
        self.Ea_rotated = -result.slope
        self.Ea_err_rotated = result.stderr
        self.Ea_translated = -result2.slope
        self.Ea_err_translated = result2.stderr
        
        if not suppress_output:
            font = {
            'weight' : 'bold',
            'size'   : 22}

            mpl.rc('font', **font)
            mpl.rc('text',usetex =False)
            fig, ax = plt.subplots(figsize=(10,10))
            ax.plot(np.array(tmpv), _sorted_rotated_D_constants,'o-')
            ax.plot(np.array(tmpv), _sorted_translated_D_constants,'o-')

            if self.heater == True:
                self._label_axes(ax,'Temperature (K)','Diffusion constant ($nm^2$ / s)')
            else:
                self._label_axes(ax,'Voltage (V)','Diffusion constant ($nm^2$ / s)')

            fig, ax1 = plt.subplots(figsize=(10,10))
            sns.regplot(x=np.reciprocal(tmpv),y= np.log(_sorted_rotated_D_constants), ci=None, ax=ax1)
            sns.regplot(x=np.reciprocal(tmpv),y= np.log(_sorted_translated_D_constants), ci=None, ax=ax1)

            if self.heater == True:
                self._label_axes(ax1,'1/T (1/K)','Log Diffusion constant ($nm^2$ / s)')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{T}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result.slope,slope_stderr=result.stderr,intercept = result.intercept, intercept_stderr=result.intercept_stderr),xy=(350,500), xycoords='figure pixels')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{T}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result2.slope,slope_stderr=result2.stderr,intercept = result2.intercept, intercept_stderr=result2.intercept_stderr),xy=(350,400), xycoords='figure pixels')
            else:
                self._label_axes(ax1,'1/V (1/V)','Log Diffusion constant ($nm^2$ / s)')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{V}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result.slope,slope_stderr=result.stderr,intercept = result.intercept, intercept_stderr=result.intercept_stderr),xy=(350,500), xycoords='figure pixels')
                #ax1.annotate(r'ln(D)= ({slope:.2f} $\pm$ {slope_stderr:.2f})$\frac{{1}}{{V}}$+ ({intercept:.2f} $\pm$ {intercept_stderr:.2f})'.format(slope=result2.slope,slope_stderr=result2.stderr,intercept = result2.intercept, intercept_stderr=result2.intercept_stderr),xy=(350,400), xycoords='figure pixels')

    def _calculate_mean_axis(self, mu_hats):
        return sum(mu_hats)/len(mu_hats)
    
    def _project_to_mean_axis(self, mu_hats, mean_mu_hat):
        return [np.dot(v,mean_mu_hat) for v in mu_hats]
    
    def plot_drift_data(self):
        self.plot_drift_vectors()
        self.plot_drift_scalar()
          
    def make_gif(self):
        pass
    # I don't know why this exists

    def plot_ed(self):
        """"Plots ensemble average x velocity, y velocity, and msd as a function of voltage."""
        fig, axs = plt.subplots(3)
        t = [i for i in range(1,len(self.ed[0][0])+1)]
        vx = []
        vy = []
        for i, volt in enumerate(self.voltages_temperatures):
            slope, intercept, _, _, _ = linregress(t[:-5],self.ed[i][0][:-5])
            #print("vx={:.2f}nm/s".format(slope))
            vx.append(slope)
            slope, intercept, _, _, _ = linregress(t[:-5],self.ed[i][1][:-5])
            #print("vy={:.2f}nm/s".format(slope))
            vy.append(slope)
        mpl.rcParams.update({'font.size': 24, 'font.weight':'bold'})


        axs[0].plot(self.voltages_temperatures,vx,'o-')
        axs[0].set_title('ensemble averaged vx')
        axs[1].plot(self.voltages_temperatures,vy,'o-')
        axs[1].set_title('ensemble averaged vy')
        axs[2].plot(self.voltages_temperatures,np.array(vx)**2 + np.array(vy)**2,'o-')
        axs[2].set_title('ensemble averaged msd')

        for i in range(3):
            axs[i].set_xlabel('voltage(V)')
            axs[i].set_ylabel('velocity (nm/s)')
            if i == 2:
                axs[i].set_ylabel('velocity (nm/$s^2$)')
        plt.savefig(self.ANALYSIS_FOLDER + "ensemble averaged v.png")
    
    def plot_v_over_D(self):
        """Plots D, v_drift, and v_drift/D in3 separate plots."""
        def exponenial_func(x, a, b):
            return a * np.exp(-b / x )
        
        plt.figure(figsize=(7,5))
        popt, pcov = curve_fit(exponenial_func, self.voltages_temperatures, self.D_constants)

        xx = np.linspace(self.voltages_temperatures[0], self.voltages_temperatures[-1], 100)
        yy = exponenial_func(xx, *popt)
        plt.plot(xx, yy)
        plt.plot(self.voltages_temperatures, np.array(self.D_constants), 'o')
        plt.xlabel('$V_{SD} (V)$')
        plt.ylabel('$D (nm^2/s)$')

        plt.figure(figsize=(7,5))
        mag_displace = np.linalg.norm(self.mu_hats, 2, axis=1)
        popt1, pcov1 = curve_fit(exponenial_func, self.voltages_temperatures, mag_displace)
        yy1 = exponenial_func(xx, *popt1)
        plt.plot(xx, yy1)
        
        plt.plot(self.voltages_temperatures, mag_displace , 'o')
        plt.xlabel('$V_{SD} (V)$')
        plt.ylabel('$v_{drift} (nm/s)$')

        plt.figure(figsize=(7,5))
        yy2 = exponenial_func(xx, *popt1)/exponenial_func(xx, *popt)
        plt.plot(xx, yy2)
        plt.plot(self.voltages_temperatures, mag_displace/np.array(self.D_constants), 'o')
        plt.xlabel('$V_{SD} (V)$')
        plt.ylabel('$v_{drift}/D \ (1/nm)$')
        
    def _overlay_diffusion_grid(self, ax, plotrange = 6, theta = 0, **kwargs):
        a = 0.246
        for j, phi in enumerate(np.linspace(0,4*np.pi/3,3)):
            for i in np.arange(-plotrange,plotrange):
                x = np.linspace(-5,5,100)
                y = x*np.tan(theta+ phi) + i*a/np.cos(theta+ phi)*np.sqrt(3)/2
                ax.plot(x,y,color='silver', linestyle = '--', linewidth=1)
    
    
    def plot_scatter_rot(self,markers=['^','o','s'],palette=['#4c92c3','#ff993e','#41aa41'],sizes=[60,30,60],titles=False,temps=None):
        """
        Plots particle displacements on a triangular lattice and visualizes rotation classifications.
    
        This function generates two main visualizations:
        1. A FacetGrid of displacement scatter plots (dx vs. dy), grouped by applied voltage (VSD).
           Points are color-coded and marked based on their rotation classification:
           - 0: Translation
           - 1: Rotation
           - 2: No movement (below threshold)
        2. A line plot showing the fraction of each rotation type (translation, rotation, no movement)
           as a function of applied voltage or temperature.
    
        Parameters
        ----------
        markers : list of str, default=['^','o','s']
            List of marker styles for each rotation class (0, 1, 2) used in plots.
            
        palette : list of str, default=['#4c92c3','#ff993e','#41aa41']
            List of color hex codes used to represent each rotation class (0, 1, 2) in the scatter and line plots.
    
        sizes : list of int, default=[60,30,60]
            Marker sizes for each rotation class in the scatter plot.
    
        titles : list of str or bool, default=False
            If False, plot titles will display the `VSD` value using `self.voltages_temperatures`.
            If a list of strings is provided, these titles will be used directly for the scatter plots.
    
        temps : list-like or None, optional
            Temperature values corresponding to each `VSD` condition.
            Required if `titles` is provided and used to plot the line graph x-axis.
    
        Returns
        -------
        g : seaborn.FacetGrid
            The grid of displacement scatter plots by VSD.
    
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the rotation kind population line plot.
    
        data : pandas.DataFrame
            The concatenated DataFrame of all displacements and rotation classifications.
    
        Notes
        -----
        Particles are classified as 'No Movement' (rotation class 2) if their total displacement
        magnitude `dr` is below 0.04. """
        #titles is just to change titles for the paper so it's temperature
        sns.set(font_scale=2)        # controls default text sizes
        sns.set_style('ticks')
        try:
            data = pd.concat(self.rots, axis=0)
        except:
            data = self.rots[0]

        data.loc[data['dr'] < 0.04, 'rotated'] = 2
        # Reduce palette/size/marker lists to match the actual data
        n_classes = data['rotated'].nunique()
        # Full reference versions (DO NOT slice these later)
        full_palette = ['#ff9999', '#66b3ff', '#99ff99']
        full_markers = ['o', 's', '^']
        Names = ['Translations', 'Rotations', 'No Movement']


        g = sns.FacetGrid(data, col="VSD", col_wrap=4, xlim=(-1, 1), ylim=(-1, 1), despine=False,height=5)
        theta = 0 / 180 * np.pi
        g.map(sns.scatterplot, 'dx', 'dy', 'rotated','rotated','rotated',
            palette=palette,
            markers=markers,
            sizes=sizes,edgecolor='black',linewidth=0.2)
        for number,ax in enumerate(g.axes.flat):
            self._overlay_diffusion_grid(ax, theta=np.pi / 2)
            ax.set_xlabel("dx (nm)")
            ax.set_ylabel("dy (nm)")
            if not titles:
                ax.set_title(r'$V_{SD}=$'+f'{self.voltages_temperatures[number]:.2f}',)
            else:
                ax.set_title(titles[number],fontweight='bold')
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        Rotation_kinds = data.pivot_table(index='VSD', columns='rotated', aggfunc='size', fill_value=0)

        # Ensure all expected columns are present
        for col in [0, 1, 2]:
            if col not in Rotation_kinds.columns:
                Rotation_kinds[col] = 0

        # Sort columns to ensure consistent order
        Rotation_kinds = Rotation_kinds[[0, 1, 2]]

        Names=['Translations','Rotations','No Movement']
        totals=Rotation_kinds[0].values+Rotation_kinds[1].values+Rotation_kinds[2].values
        fig,ax=plt.subplots(figsize=(9,6))
 
        for a in range(3):
            i=a
            if not titles:
                ax.plot(Rotation_kinds[i].index, Rotation_kinds[i].values/(totals)*100,color=palette[i],marker=markers[i],
                label=Names[i],markersize=12,linewidth=6)
            else:
                ax.plot(temps, Rotation_kinds[i].values/(totals)*100,
                        color=palette[i],marker=markers[i],markersize=20,linewidth=6)
                ax.scatter(temps, Rotation_kinds[i].values/(totals)*100,
                        color=palette[i],marker=markers[i],label=Names[i],s=120)
    
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(25)
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(25)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.8)  # change width
            ax.spines[axis].set_color('black')
        ax.legend(fontsize=17,loc='upper right',
                  handletextpad=0,handlelength=1.5
                 ,borderpad=0.4)
    
        ax.tick_params(which='minor', width=0)
    
        ax.tick_params(which='major', width=0.5,)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        if titles:
                ax.set_xlabel(r'Temperature (K)')
        else:
            ax.set_xlabel(r'$V_{SD}$(V)')
        ax.set_ylim(0,100)
        ax.set_ylabel('% of Molecules')
        ax.locator_params(nbins=7,axis='x')
        fig.tight_layout()
    
        return g,fig,data
    def _overlay_diffusion_grid(self, ax, plotrange = 6, theta = 0, **kwargs):
        a = 0.246
        for j, phi in enumerate(np.linspace(0,4*np.pi/3,3)):
            for i in np.arange(-plotrange,plotrange):
                x = np.linspace(-5,5,100)
                y = x*np.tan(theta+ phi) + i*a/np.cos(theta+ phi)*np.sqrt(3)/2
                ax.plot(x,y,color='silver', linestyle = '--', linewidth=1)
    
    def plot_rotated_frac(self):
        """Plots the percentage of each kind of rotation."""
        plt.figure(figsize=(10,6))
        plt.rcParams.update({'font.size': 18})
        plt.plot(self.voltages_temperatures,np.array(self.total_translated)/self.total_molecules*100,'o-',label='translated')
        plt.plot(self.voltages_temperatures,np.array(self.total_rotated)/self.total_molecules*100,'o-',label='rotated')
        plt.plot(self.voltages_temperatures,(np.array(self.total_molecules)-np.array(self.total_moved))/self.total_molecules*100,'o-',label='no movement')
        plt.legend()
        #plt.xlim([0.5,2.0])
        plt.ylim([0, 100])
        plt.xlabel('$V_{SD}$ (V)')
        plt.ylabel('% of molecules')
        
    def plot_rotated_abs(self):
        """Plots the population of each kind of rotation."""

        plt.figure(figsize=(10,6))
        plt.rcParams.update({'font.size': 18})
        plt.plot(self.voltages_temperatures,np.array(self.total_translated),'o-',label='translated')
        plt.plot(self.voltages_temperatures,np.array(self.total_rotated),'o-',label='rotated')
        plt.plot(self.voltages_temperatures,(np.array(self.total_molecules)-np.array(self.total_moved)),'o-',label='no movement')
        plt.plot(self.voltages_temperatures,np.array(self.total_molecules),'o-',label='total')
        plt.legend()
        #plt.xlim([0.5,2.0])
        #plt.ylim([0, 200])
        plt.xlabel('$V_{SD}$ (V)')
        plt.ylabel('# of molecules')
     
    def _calculate_n_frame_displacements(self, n):
        for t3 in self.t3s:
            self.n_frame_displacements.append(self._calculate_displacements(t3,n))
            
    def plot_n_frame_scatter(self, n):
        """   
        Plots displacement scatter and density maps for particle motion over `n` frames.
    
        Parameters
        ----------
        n : int
            Number of frames over which to compute displacements.
    
        Notes
        -----
        For each voltage condition, shows a density heatmap and scatter of displacements
        (dx, dy), with contours overlaid.
        """
        self._calculate_n_frame_displacements(n)
        x, y = 1, len(self.voltages_temperatures)
        fig, ax = plt.subplots(x, y, figsize=(20,40))
        xmin, xmax, ymin, ymax = -20, 20, -20, 20
        for i, test in enumerate(self.n_frame_displacements):
            
            X, Y, Z = self._density_estimation(test.dx, test.dy)

            # Show density 
            ax[i].imshow(np.rot90(np.fliplr(Z)), cmap=plt.cm.gist_earth_r,                                                    
                      extent=[xmin, xmax, ymin, ymax])

            # Add contour lines
            ax[i].contour(X, Y, Z)                                                                           
            ax[i].plot(test.dx, test.dy, 'r.', markersize=2)    
            ax[i].set_xlim([xmin, xmax])                                                                           
            ax[i].set_ylim([ymin, ymax])
            ax[i].set_title('{:.1f} V'.format(self.voltages_temperatures[i]))

    def _density_estimation(self, m1, m2):
        xmin, xmax, ymin, ymax = -20, 20, -20, 20
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]                                                     
        positions = np.vstack([X.ravel(), Y.ravel()])                                                       
        values = np.vstack([m1, m2])                                                                        
        kernel = stats.gaussian_kde(values)                                                                 
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z
    
    def plot_n_frame_scatter_overlay(self, n, threshold = 0.0007):
        self._calculate_n_frame_displacements(n)
        fig, ax = plt.subplots(figsize=(10,10))                   
        cmap = mpl.cm.get_cmap('coolwarm', len(self.n_frame_displacements))
        colors = cmap(np.arange(0,cmap.N))

        for i, test in enumerate(self.n_frame_displacements):
            X, Y, Z = self._density_estimation(test.dx, test.dy)

            # Add contour lines
            plt.contour(X, Y, Z, [threshold], colors=colors[i].reshape(-1,4))                                                                           
            ax.plot(test.dx, test.dy, '.', markersize=5, color= colors[i], label='{:.2f} V'.format(self.voltages_temperatures[i]))    
            ax.set_xlim([-10, 10])                                                                           
            ax.set_ylim([-10, 10])                                                                           
            plt.legend(fontsize=16)
            plt.grid('on')
            ax.set_xticks(np.linspace(-20,20,5))
            ax.set_yticks(np.linspace(-20,20,5))
            ax.tick_params(labelsize=16)
            ax.set_aspect('equal', 'box')
            
    def _avg_over_n_steps(self, arr, n):
        result = []
        for i in range(len(arr),0,-n):
            if i-n < 0: break
            result.append(np.mean(arr[i-n:i]))
        return result[::-1]
    
    def plot_rocket_tracks(self, set_i=-1, image_i = -1, valid_list = [], label_particles = False):
        """
        Plot particle trajectories over a background image for a specific dataset.
    
        This function visualizes particle motion as colored tracks ("rocket trails") 
        overlaid on a single frame from a specified dataset. Each trajectory fades 
        from bright to dark using a colormap to indicate time evolution.
    
        Parameters
        ----------
        set_i : int, optional
            Index of the dataset in `self.t3s` and `self.frames` to use (default is -1, the last dataset).
        image_i : int, optional
            Index of the image/frame within the selected dataset to use as the background (default is -1, the last frame).
        valid_list : list of int, optional
            List of particle indices to include in the plot. If empty, all particles are included.
        label_particles : bool, optional
            Whether to annotate the endpoints of trajectories with particle indices (default is False).
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plotted tracks.
        """        
        x_end = self.t3s[set_i].groupby(by='particle').x.apply(list)
        y_end = self.t3s[set_i].groupby(by='particle').y.apply(list)

        if not valid_list:
            valid_list = x_end.index
        #valid_list = [2,24,8,12,19,22,7,6,5,9,13,10,1,21,16] # positive current
        #valid_list = [17,20,15,0,16,10,19,21,9,12,8,7,6,11] # negative current
        fig, ax = plt.subplots(figsize=(10,10))
        skip = 1
        for i in valid_list:
            ax.imshow(self.frames[set_i][image_i],cmap='gray')
            x_plt = self._avg_over_n_steps(x_end[i], skip)
            y_plt = self._avg_over_n_steps(y_end[i], skip)
            cmap = plt.cm.get_cmap("YlOrRd")
            colors = cmap(np.linspace(0.8,0,len(x_plt)+1))
            if label_particles:
                ax.text(x_plt[-1],y_plt[-1],f'{i}')
            for j, _ in enumerate(x_plt):
                if j == len(x_plt) + image_i: break
                ax.plot(x_plt[j:j+2], y_plt[j:j+2], color=colors[j], linewidth=3)

        ax.axis('off')
        return fig
    
    def plot_orientation_dist(self):
        """Plots the distribution of particle orientations as histograms."""
        #def draw_dividing_line(**kwargs):
        #    plt.axvline(30-15,color='r')
        #    plt.axvline(-30-15,color='r')     
        sns.set_style('ticks')
        concatted = pd.concat(self.displacements, axis=0)
        concatted['angle'] = concatted.angle/np.pi*180
        g = sns.FacetGrid(concatted, hue='orientation', col="VSD", col_wrap=4, height=5, xlim = (-300,300), despine =False)
        g.map(sns.histplot, "angle" , bins=20, kde=False)
        #g.map(draw_dividing_line)
        for ax in g.axes.flat:
            ax.tick_params(labelleft=True,labelbottom=True)
            ax.set_xticks([-180,-60,0,60,180])
            ax.set_xlabel('Angle (deg.)')
            ax.set_ylabel('Counts')
        g.fig.tight_layout()
        
    def _calculate_dist(self, X0, X1):
        '''
        This function takes a list of subject coords of parcels X1 and 
        returns the nearest distance to a list of coords of target parcels X0
        e.g. calculate_dist(road_parcels, sold_parcels)
        '''
        neigh = NearestNeighbors(n_neighbors=1, radius=10.0, algorithm = 'kd_tree')
        neigh.fit(X0)
        dist, _ = neigh.kneighbors(X1)
        return dist
    
    def _overlay_lattice_points(self, ax, a, trunc_lattice, rot, color="C0", fill = False):
            hexbin_counts = []
            total_counts = len(rot)
            
            for _, point in trunc_lattice.iterrows():
                hexagon = patches.RegularPolygon((point.x, point.y), 6, radius=a/2/np.sqrt(3), orientation=np.pi/2, 
                                          edgecolor=color, linewidth=1, linestyle='--', fill = fill, alpha=0.7)
                ax.add_patch(hexagon)
                
                
                
                if fill:
                    cp_idx = rot.apply(lambda row: hexagon.contains_point(ax.transData.transform([row.dx,row.dy])), axis=1)
                    #ax.scatter(rot[cp_idx].dx, rot[cp_idx].dy,color='r')
                    count = len(rot[cp_idx])
                    
                    entry = [point.x, point.y, count, count/total_counts]
                    hexbin_counts.append(entry)
                    colormax = 25
                    hexagon.set_facecolor(plt.cm.hot(count/colormax))
                    #ax.text(point.x, point.y, f'{count}')
            return pd.DataFrame(data = hexbin_counts, columns = ['x','y','counts','prob'])
                    
    def _calculate_lattice_points(self, a):
            plotrange = 1
            a1 = np.array([np.sqrt(3)/2*a, a/2])
            a2 = np.array([np.sqrt(3)/2*a, -a/2])
            lattice_points = []
            btw_lattice_points = []
            for i in range(-30,30):
                for j in range(-30,30):
                    lattice_points.append(i*a1 + j*a2)
                    btw_lattice_points.append((i*a1 + j*a2)/2)
            lattice_points = np.array(lattice_points) 
            lattice_df = pd.DataFrame(lattice_points)
            lattice_df = lattice_df.rename(columns={0: "x", 1: "y"})
            trunc_lattice = lattice_df[(lattice_df.x <plotrange) & (lattice_df.x >-plotrange) & (lattice_df.y <plotrange) & (lattice_df.y >-plotrange)]

            btw_lattice_df = pd.DataFrame(np.array(btw_lattice_points)).rename(columns={0: "x", 1: "y"})
            btwdf = pd.merge(btw_lattice_df, lattice_df, how='left', on=['x', 'y'], indicator=True)
            btw_lattice_df = btwdf[btwdf['_merge'] == 'left_only']
            btw_trunc_lattice = btw_lattice_df[(btw_lattice_df.x <plotrange) & (btw_lattice_df.x >-plotrange) & (btw_lattice_df.y <plotrange) & (btw_lattice_df.y >-plotrange)]
            return trunc_lattice, btw_trunc_lattice

    
    def plot_hexbin(self, *args, **kwargs):
        rot = kwargs.pop('data')
        print(rot['VSD'].iloc[0])
        
        ax = plt.gca()
        trunc_lattice, btw_trunc_lattice = self._calculate_lattice_points(self.a)
        hexbin_counts_rot = self._overlay_lattice_points(ax, self.a, self.btw_trunc_lattice, rot, color = "C1", fill=True)
        hexbin_counts_tr = self._overlay_lattice_points(ax, self.a, self.trunc_lattice, rot, color = "C0", fill=True)
        self.hexbin_counts.append(pd.concat([hexbin_counts_rot, hexbin_counts_tr], ignore_index=True))


        dist_tran = self._calculate_dist(list(zip(self.trunc_lattice.x, self.trunc_lattice.y)), list(zip(rot['dx'], rot['dy'])))
        dist_rot = self._calculate_dist(list(zip(self.btw_trunc_lattice.x, self.btw_trunc_lattice.y)), list(zip(rot['dx'], rot['dy'])))
        rot = rot.copy()
        rot['rot_ind'] = dist_tran > dist_rot
        tran_p = rot[~rot['rot_ind']]
        rot_p = rot[rot['rot_ind']]
        #plt.scatter(tran_p.dx,tran_p.dy, color = 'C0')
        #plt.scatter(rot_p.dx,rot_p.dy, color = 'C1')

        self._format_plotrange(ax)


    def plot_hexbins(self):
        concatted = pd.concat(self.rots, axis=0)
        g = sns.FacetGrid(concatted, col="VSD", col_wrap=4, height=5, despine =False)
        g.map_dataframe(self.plot_hexbin)
        g.fig.tight_layout()
    
    def _format_plotrange(self, ax, plotrange = 1):
        ax.set_xlim([-plotrange,plotrange])
        ax.set_ylim([-plotrange,plotrange])
        ax.set_xticks(np.linspace(-plotrange,plotrange, 5))
        ax.set_yticks(np.linspace(-plotrange,plotrange, 5))
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

    
    def plot_rot(self, *args, **kwargs):
        rot = kwargs.pop('data')
        rot_1 = rot[rot.orientation == 0]
        rot_2 = rot[rot.orientation == 1]
        rot_3 = rot[rot.orientation == 2]
        ax = plt.gca()
        ax.scatter(rot_1.dx,rot_1.dy, label='0 deg.')
        ax.scatter(rot_2.dx,rot_2.dy, label='120 deg.')
        ax.scatter(rot_3.dx,rot_3.dy, label='240 deg.')
        ax.legend()
        
        self._format_plotrange(ax)
         
    def plot_rots(self):
        concatted = pd.concat(self.rots, axis=0)
        g = sns.FacetGrid(concatted, col="VSD", col_wrap=4, height=5, despine =False)
        g.map_dataframe(self.plot_rot)
        g.fig.tight_layout()
class STPFitter:
    
    def __init__(self, fileranges=None, voltages=None, folder_name = None):
        
        if any((fileranges == None) or (voltages == None)):
            print('no filerange or voltages/temperatures specified')
            return
        if len(fileranges) != len(voltages):
            print('number of sets and voltages/temperatures don\'t match')
            return

        self.fileranges = fileranges
        self.voltages = voltages    
        self.SXM_PATH = [folder_name + "/Image_{0:03}.sxm".format(i) for i in fileranges]
        self.SET_NAME = "{}-{}_STP/".format(min(fileranges), max(fileranges))
        self.ANALYSIS_FOLDER = "./analysis/" + folder_name + "_" + self.SET_NAME
        self.MOVIE_FOLDER = self.ANALYSIS_FOLDER + "movies/"
        self.PARAMS_FILENAME = "params.yaml"
        if not os.path.exists(self.ANALYSIS_FOLDER):
            os.makedirs(self.ANALYSIS_FOLDER)
        if not os.path.exists(self.ANALYSIS_FOLDER):
            os.makedirs(self.MOVIE_FOLDER)
        self.frames = SXMReader(self.SXM_PATH, channel = "Bias")
        thetas = []
        theta_mag = []
    
    def analyze_STP(self, rotation=0):
        self.thetas=[]
        self.theta_mag=[]
        mpl.rc('figure',  figsize=(20, 20))
        fig, ax = plt.subplots(1,len(self.frames))
        plt.subplots_adjust(wspace=0.1, hspace=0.05) #for 25 images (wspace=-0.8, hspace=0.05) #wspace=-0.69, hspace=0.01 #wspace=-0.1, hspace=0.01
        for i, frame in enumerate(self.frames):
            frame = -frame*1000
                      
            z=np.array(frame).ravel()
            scan_px = self.frames.scan_size['pixels']['x']
            scan_size = self.frames.scan_size['real']['x']*1e9 #in nm
            m_range = np.linspace(-scan_size/2,scan_size/2,scan_px)
            x, y = np.meshgrid(m_range,m_range)
            theta = self._fit_plane(x,y,frame)
            
            # Plotting the potential and average gradient as arrow
            avg = np.average(frame)
            rot_frame = ndimage.rotate(frame, -rotation, reshape=True)
            
            ax[i].imshow(rot_frame, clim=[avg-0.5,avg+0.5])
            ax[i].text(scan_px*0.5,scan_px*5/6, "{}V".format(self.voltages[i]), ha="right", weight='bold', color="orange", fontsize =12)
            ax[i].axis('off')
            #cbar = plt.colorbar()
            #cbar.ax.set_ylabel('Voltage (mV)', rotation=270, labelpad =25)
            plt.axis('off')
            plt.grid()
            arrx = 1000*(np.cos(rotation*np.pi/180)*theta[0] - np.sin(rotation*np.pi/180)*theta[1])
            arry = 1000*(np.sin(rotation*np.pi/180)*theta[0] + np.cos(rotation*np.pi/180)*theta[1])
            ax[i].arrow(scan_px/2,scan_px/2,arrx,arry,shape='full', head_length=5, head_width=5, lw=3, color='r')


            self.thetas.append(theta)
            self.theta_mag.append(np.linalg.norm(theta[:1]))
        fig.savefig(self.ANALYSIS_FOLDER + "STP_all.png", bbox_inches='tight')
        x = self.voltages
        y = np.sign(np.array(self.thetas)[:,0])*np.array(self.theta_mag)
        self.slope, self.intercept, _, _, _ = linregress(x,y)

    
    def _fit_plane(self,x,y,frame):
        Z=np.array(frame).ravel()
        X = np.transpose(np.vstack([x.ravel(), y.ravel()]))
        X = np.append(X, np.ones([len(Z),1]), axis=1)
        theta = np.linalg.inv(np.dot(np.transpose(X),X)).dot(np.transpose(X)).dot(Z)
        return theta
    
    def plot_Efield(self):
        x = self.voltages
        y = np.sign(np.array(self.thetas)[:,0])*np.array(self.theta_mag)
        xx = np.linspace(min(self.voltages),max(self.voltages),100)
        yy = self.slope * xx + self.intercept
        fig = plt.figure(figsize=(6,6))
        plt.plot(x, y,'o-')
        plt.plot(xx, yy,'r--')
        plt.xlabel('$V_{SD}$ (V)')
        plt.ylabel('E-field (mV/nm)')
        plt.grid()
        fig.savefig(self.ANALYSIS_FOLDER + "Efield.png", bbox_inches='tight')

    
    def interp_efield(self,v):
        return self.slope * v + self.intercept