#-----------------------------------------------------------------------------80
# NOTE: DOCUMENTATION FOR THIS FILE IS STILL IN PROGRESS

import freud
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Polygon
import matplotlib.animation
from matplotlib.animation import FuncAnimation 
from IPython.display import display,HTML
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from IPython.display import clear_output
from PIL import Image
import trackpy as tp
from sxmreader import SXMReader

from scipy.ndimage import maximum_filter


# Definitions for various fitting functions
pwrlaw = lambda t, c, K, a: (K + c) * (t**a)
pwrlawD = lambda t, D, a: 4 * D * (t**a)
r2 = lambda y, ypred: np.abs(1 - np.sum((y - ypred)**2) / np.sum((y - np.mean(y))**2))
rmse = lambda y, ypred: np.sqrt(np.sum((y - ypred)**2) / len(y))
constfunc = lambda x, c: x/x*c

pwrlawdecay = lambda x, m,d,a: m * (x)**(-a)
#pwrlawdecay = lambda x, m,d,a: m * (x-d)**(-a)
pwrlawdecay2 = lambda x,m,d,a: m * (x)**(-a)+d

#expdecay = lambda x, m,a: m * np.exp(-(x) / a)
#expdecay2 = lambda x, m,d,a: m * np.exp(-(x) / a)+d
pwrlawdecay=lambda x,a,n: a+-n*np.log(x)
expdecay = lambda x, m,a: m+-x/a
def get_cffits(x, a, fit_id,upper_bounds=[np.inf,np.inf],sigma=None,r2fit=1,):
    """
    Fit correlation function data to different decay models to characterize phase behavior.
    
    This function is central to 2D phase identification, as different phases exhibit
    characteristic correlation decay:
    - Liquid: Exponential decay
    - Hexatic: Algebraic (power law) decay  
    - Crystal: Constant or very slow decay
    
    Parameters
    ----------
    x : array_like
        Distance values (typically from correlation function peaks)
    a : array_like
        Correlation values to fit (will be log-transformed internally)
    fit_id : int or None
        Which fit to return: 0=constant, 1=power law, 2=exponential
        If None, returns the best fit based on r2fit criterion
    upper_bounds : list of float, default [np.inf, np.inf]
        Upper bounds for [power_law_exponent, exponential_length_scale]
    sigma : array_like, optional
        Uncertainties for weighted fitting
    r2fit : int, default 1
        Fit selection criterion: 0=R², 1=RMSE, 2=combined
        
    Returns
    -------
    list
        [fit_function, optimal_parameters, annotation_string, best_fit_id, rmse_scores]
        
    Notes
    -----
    Fitting is performed in log space (log(a)) for better discrimination between
    exponential and power law behavior. The power law fit uses the form:
    log(correlation) = a - n*log(r), corresponding to r^(-n) decay.
    """
    y=np.log(a) #fitting to log scale
    popt_constfunc = curve_fit(constfunc, x,y,sigma=sigma,maxfev=3000, )[0]
    popt_pwrlawdecay = curve_fit(pwrlawdecay, x,y,sigma=sigma,
                                     maxfev=3000,p0=(1,0.2),
                                     bounds=((-np.inf,0),
                                     (np.inf,upper_bounds[0])))[0]
    popt_expdecay = curve_fit(expdecay, x, y,sigma=sigma,maxfev=3000,p0=(1,1),
                                      bounds=((-np.inf,-np.inf),
                                     (np.inf,upper_bounds[1])))[0]
   
    fit_ypreds = [constfunc(x, *popt_constfunc), 
                  pwrlawdecay(x, *popt_pwrlawdecay), 
                  expdecay(x, *popt_expdecay)]
    
    r2_scores = []
    rmse_scores = []
    for ypred in fit_ypreds:
        r2_scores.append(r2(y, ypred))
        rmse_scores.append(rmse(y, ypred))
    
    fit_names = ['constant fit', 'power law fit','exponential fit']
    fit_funcs = [constfunc, pwrlawdecay, expdecay]
    popts = [popt_constfunc, popt_pwrlawdecay, popt_expdecay]
    annotations = [
        r'$c\simeq ${:.2f}'.format(popts[0][-1]),
        r'$r^{{-\eta}}: $ ($\eta\simeq ${:.4f})'.format(popts[1][-1]),
        r'$\exp(-r/\xi): $($\xi\simeq ${:.4f})'.format(popts[2][-1])
    ]
    if r2fit==0:
        min_id = np.argmin(np.abs(np.array(r2_scores)))
    if r2fit==1:
        min_id = np.argmin(np.abs(np.array(rmse_scores)))
    if r2fit==2:
        min_id = np.argmin(np.abs(np.array(r2_scores))+np.abs(np.array(rmse_scores)))
        
    print(f'popt_constfunc (r2={r2_scores[0]:.5g}, rmse={rmse_scores[0]:.5g}): {popt_constfunc}')
    print(f'popt_pwrlawdecay (r2={r2_scores[1]:.5g}, rmse={rmse_scores[1]:.5g}): {popt_pwrlawdecay}')
    print(f'popt_expdecay (r2={r2_scores[2]:.5g}, rmse={rmse_scores[2]:.5g}): {popt_expdecay}\n')
    
    if fit_id is not None:
        return_fit = [fit_funcs[fit_id], popts[fit_id], annotations[fit_id],min_id,rmse_scores]
    else:
        print(f'winner: {fit_names[min_id]}\n')
        return_fit = [fit_funcs[min_id], popts[min_id], annotations[min_id],min_id,rmse_scores]        
    
    return return_fit

class imgdata:
    """
    Container class for image data and particle tracking analysis.
    
    This class handles loading and preprocessing of experimental images,
    particularly STM data, and performs automatic particle detection using trackpy.
    It serves as the primary data structure for subsequent phase analysis.
    
    Parameters
    ----------
    path : list or tuple
        Path(s) to image file(s). For SXM files, should be a single-element list/tuple
        due to SXMReader requirements
    filetype : str, default 'SXM'
        Type of image file: 'SXM' for scanning tunneling microscopy data,
        or other formats supported by PIL
    **kwargs
        Additional parameters passed to trackpy.locate() for particle detection
        
    Attributes
    ----------
    img : Image or SXMReader frame
        The loaded image data
    data : pandas.DataFrame
        Particle positions from trackpy analysis with columns ['x', 'y', ...]
    sxm : bool
        Whether the data is from an SXM file
    dims : list, optional
        Real-world dimensions [x_size, y_size] in nanometers (SXM files only)
        
    Examples
    --------
    >>> # Load STM data with particle detection
    >>> img_d = imgdata(['data.sxm'], diameter=15, minmass=1000)
    >>> print(f"Found {len(img_d.data)} particles")
    
    >>> # Load regular image
    >>> img_d = imgdata(['image.png'], filetype='PNG', diameter=10)
    """
    def __init__(self,path,filetype='SXM',**kwargs,):
    
        if filetype=='SXM':
            img = SXMReader(path, correct='lines')[0]
            self.sxm=True
            self.dims=[
                SXMReader(path).scans[0].size['real']['x']*10**9,
                SXMReader(path).scans[0].size['real']['y']*10**9]

        else:
            img = Image.open(path[0])
            img = img.resize((200,200), Image.ANTIALIAS)
            img=img.convert('L')
            img.shape=img.size
            img.ndim=2
            self.sxm=False
        self.img=img
        self.data=tp.locate(img,**kwargs,engine='python').reset_index()
        
    
def plot_boops(img_d,dpi=72,savename=False,kind=0,realdim=False,filetype='svg',anim_param=False,):
    """
    Plot bond-orientational order parameter (BOOP) to visualize local structure and defects.
    
    This function creates Voronoi tessellations and computes hexatic order parameters
    to reveal topological defects (5-7 disclinations) and local orientational order.
    Essential for identifying hexatic phases and grain boundaries in 2D crystals.
    
    Parameters
    ----------
    img_d : imgdata
        Image data container with particle positions
    dpi : int, default 72
        Figure resolution
    savename : str or False, default False
        Filename to save figure, or False for no saving
    kind : int, default 0
        Plot type: 0=coordination number/defects, 1=orientational order
    realdim : bool or list, default False
        Use real dimensions: False=pixels, True=use img_d.dims, 
        or [x_size, y_size] for custom dimensions
    filetype : str, default 'svg'
        File format for saving
    anim_param : tuple or False, default False
        (fig, ax) for animation, or False for new figure
        
    Notes
    -----
    Kind 0 (defects): Colors Voronoi cells by coordination number
    - Blue/green: 5-fold coordinated (disclination cores)
    - White: 6-fold coordinated (ideal hexagonal)  
    - Red/purple: 7-fold coordinated (disclination cores)
    
    Kind 1 (orientation): Colors by local hexatic phase angle
    - Reveals grain boundaries and orientational domains
    - Color represents angle from 0 to π radians
    
    The hexatic order parameter ψ₆ = (1/N)Σ exp(6iθⱼₖ) characterizes
    local 6-fold rotational symmetry.
    """
    
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

    box = freud.box.Box(Lx=xdim*20, Ly=ydim*20, is2D=True)
    
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    if not anim_param:
        fig, ax = plt.subplots(1,figsize=(7,7), facecolor='w', constrained_layout=True, dpi=dpi)
    else:
        fig,ax=anim_param
    if kind==0:
        cmap0 = ListedColormap(np.vstack([
            mpl.cm.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([4,3,2,1])],
            mpl.cm.get_cmap('bwr', 3)(np.arange(3)),
            mpl.cm.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([8,7,6,5])]]))
        norm0 = Normalize(vmin=1-0.5, vmax=11+0.5)
    if kind==1:
        cmap1 = ListedColormap(mpl.cm.get_cmap('hsv_r', 256)(np.linspace(0., 1., 256))[80:])    
        norm1 = Normalize(vmin=0., vmax=np.pi)    
    
   
    vor = freud.locality.Voronoi()
    fpsi = freud.order.Hexatic(k=6, weighted=False)
    ax.set_xlim(0,xdim)
    ax.set_ylim(0,ydim)
    if True:
        mxy = xydata
        vor.compute(system=(box, points))
        fpsi.compute(system=(box, points), neighbors=vor.nlist)
        fpsi_phase = np.abs(np.angle(fpsi.particle_order))
        nsides = np.array([polytope.shape[0] for polytope in vor.polytopes])
        
        patches = []
        for polytope in vor.polytopes:
            poly = Polygon(polytope[:,:2], closed=True, facecolor='r')
            patches.append(poly)
        if kind==0:
            collection0 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap0, norm=norm0, alpha=0.6)
            collection0.set_array(nsides)
            ax0 = ax.add_collection(collection0)
            ax.set_title(label=f'5 and 7-Disclinations ',fontsize=20)
        if kind==1:
            collection1 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap1, norm=norm1, alpha=0.7)
            collection1.set_array(fpsi_phase)
            ax1 = ax.add_collection(collection1)
            ax.set_title(label=f'Local Bond-Orientational Order ',fontsize=20)
        
        ax.scatter(mxy[:,0], mxy[:,1], s=1, c='k', zorder=2)
        #box.plot(ax=ax)
        if kind==0:
            cax0 = fig.add_axes([ax.get_position().x1+0.11,ax.get_position().y0+0.045, 
                             0.02, 
                             ax.get_position().height])
                             
            cbar0 = fig.colorbar(ax0, cax=cax0, ticks=np.arange(1, 12))
            cbar0.set_label(label='number of neighbors', labelpad=20., rotation=270)
        if kind==1:
            cax1 = fig.add_axes([ax.get_position().x1+0.11, 
                             ax.get_position().y0-0.02, 
                             0.02, 
                             ax.get_position().height])
            cbar1 = fig.colorbar(ax1, cax=cax1)
            cbar1.set_ticks(np.arange(0., np.pi+(np.pi/4.), np.pi/4.))
            cbar1.ax.set_yticklabels(labels=['0', 'π/4', 'π/2', '3π/4', 'π'])
            cbar1.set_label(label='orientation in radians', labelpad=10., rotation=270)
    
    if savename!=False:
        fig.savefig(savename, format=filetype,bbox_inches='tight')    
    #return ax

def plot_bocf(img_d, bins=100,  fit_id=None, upper_bounds=[np.inf,np.inf], dpi=72,savename=False,filetype='svg',start=7,realdim=True,p_dist=1,xscale_plot='linear',yscale_plot='log',r2fit=1,figsize_=(10,6),ylim=None,sample=0,plot=True,periodic=False):
    """
    Plot bond-orientational correlation function g₆(r) for 2D phase identification.
    
    This is a key diagnostic for the KTHNY theory of 2D melting. The correlation
    function measures how 6-fold orientational order persists over distance:
    - Crystal: g₆(r) → constant (long-range orientational order)
    - Hexatic: g₆(r) ~ r^(-η) (quasi-long-range, power law decay)
    - Liquid: g₆(r) ~ exp(-r/ξ) (short-range, exponential decay)
    
    Parameters
    ----------
    img_d : imgdata
        Image data container with particle positions
    bins : int, default 100
        Number of radial bins for correlation function
    fit_id : int or None, default None
        Force specific fit: 0=constant, 1=power law, 2=exponential
        If None, selects best fit automatically
    upper_bounds : list, default [np.inf, np.inf]
        Upper bounds for [power_law_exponent, correlation_length]
    dpi : int, default 72
        Figure resolution
    savename : str or False, default False
        Filename to save figure
    filetype : str, default 'svg'
        File format for saving
    start : int, default 7
        Peak index to start fitting from (ignores near-neighbor peaks)
    realdim : bool or list, default True
        Use real dimensions for scaling
    p_dist : int, default 1
        Minimum peak separation for peak finding
    xscale_plot, yscale_plot : str, default 'linear', 'log'
        Axis scaling for the plot
    r2fit : int, default 1
        Fit criterion: 0=R², 1=RMSE, 2=combined
    figsize_ : tuple, default (10,6)
        Figure size
    ylim : tuple or None, default None
        Y-axis limits
    sample : int, default 0
        Unused parameter (legacy)
    plot : bool, default True
        Whether to create plot or just return fit parameters
    periodic : bool, default False
        Use periodic boundary conditions
        
    Returns
    -------
    tuple
        (fit_parameters, figure) if plot=True
        (fit_parameters, best_fit_id, rmse_scores) if plot=False
        
    Notes
    -----
    The correlation function is computed as:
    g₆(r) = ⟨ψ₆*(r')ψ₆(r'+r)⟩ / ⟨|ψ₆|²⟩
    
    where ψ₆(r) = (1/N)Σⱼ exp(6iθⱼ) is the local hexatic order parameter.
    
    Peak fitting starts from the 'start' peak to avoid near-neighbor correlations
    that don't follow simple scaling laws.
    """
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]


    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])
    if periodic:
        box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    else:
        box = freud.box.Box(Lx=xdim*20, Ly=ydim*20, Lz=0, is2D=True)

    fpsi = freud.order.Hexatic(k=6, weighted=False)
    cf = freud.density.CorrelationFunction(bins=bins, r_max=ydim*0.499)
    fpsi.compute(system=(box, points))
    fpsi_complex = fpsi.particle_order
    cf.compute(system=(box, points), values=fpsi_complex)    
    cf_edges = cf.bin_edges[:-1] #/ sigma)
    cf_vals=np.abs(cf.correlation)
    cfpks = find_peaks(cf_vals, distance=p_dist)[0]
    idxlim = np.argwhere(cf_edges == cf_edges[cfpks][start])[0][0]-1
    #fit_func, popt, fit_text = get_cffits(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]],
    #                                      fit_id,r2fit=r2fit,upper_bounds=[0.8,(xdim+ydim)/2])
    fit_func, popt, fit_text,min_id,rmse_scores = get_cffits(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]],
                                          fit_id,r2fit=r2fit,upper_bounds=upper_bounds)


    if plot==False:
        return popt,min_id,rmse_scores
    fig, ax = plt.subplots(figsize=figsize_, facecolor='w', constrained_layout=True, dpi=dpi)

    if ylim!=None:
        ax.set_ylim(ylim)
    ax.set(xlim=(cf_edges[cfpks[0]]*0.75, cf_edges[-1]*1.1))
    ax.set(xscale=xscale_plot)
    ax.set(yscale=yscale_plot)
    ax.plot(cf_edges, cf_vals, 'o', ms=8, c='w', mec='b', zorder=2)
    ax.plot(cf_edges, cf_vals, lw=2.5, c='b', zorder=3) 


    ax.plot(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], 'x', c='r', ms=14)

    ax.plot(cf_edges[idxlim:],np.exp(fit_func(cf_edges[idxlim:], *popt)), '--', lw=2., c='k', zorder=9)

    ax.set(xlabel=r'$r$', ylabel=f'$g_{{6}}(r)$', title=f'Bond-orientational Correlation Function $g_{{6}}(r)$  |  {fit_text}')

    if savename!=False:
        fig.savefig(savename, format=filetype)
    return popt,fig

def plot_rdf(img_d, bins=85, realdim=False, step=-1,rho=1, peak_distance=1, xlim=None, dpi=72,savename=False,filetype='svg',fit_id=None,showpeak=True,r2fit=True,start=0):
    """
    Plot radial distribution function g(r) to analyze local structure and coordination.
    
    The RDF measures the probability of finding particles at distance r relative
    to a random distribution. Peak positions reveal lattice structure, while peak
    heights and widths characterize positional order and thermal fluctuations.
    
    Parameters
    ----------
    img_d : imgdata
        Image data container with particle positions
    bins : int, default 85
        Number of radial bins
    realdim : bool or list, default False
        Use real dimensions for scaling
    step : int, default -1
        Time step (unused for static images)
    rho : float, default 1
        Density for normalizing distance by particle spacing σ
    peak_distance : int, default 1
        Minimum separation for peak detection
    xlim : tuple or None, default None
        X-axis limits
    dpi : int, default 72
        Figure resolution
    savename : str or False, default False
        Filename to save figure
    filetype : str, default 'svg'
        File format for saving
    fit_id : int or None, default None
        Fit type for peak decay analysis
    showpeak : bool, default True
        Mark detected peaks with 'x' symbols
    r2fit : bool, default True
        Use R² criterion for fitting
    start : int, default 0
        Peak index to start fitting from
        
    Notes
    -----
    For a triangular lattice, peaks occur at:
    - 1st shell: r = σ (6 nearest neighbors)  
    - 2nd shell: r = √3σ (6 next-nearest neighbors)
    - 3rd shell: r = 2σ (6 third-nearest neighbors)
    
    Peak decay analysis can reveal the nature of positional correlations,
    complementing the bond-orientational analysis.
    
    The distance is normalized by σ = √(2/√3ρ) where ρ is the particle density.
    """
    sigma = np.sqrt(2. / ((3.**0.5) * rho))
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    rdf = freud.density.RDF(bins=bins, r_max=ydim*0.4999)

    rdf.compute(system=(box, points))
    rdf_edges = rdf.bin_edges[:-1] / sigma
    rdf_vals=rdf.rdf

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax.plot(rdf_edges, rdf_vals, 'o', ms=5, c='w', mec='b', zorder=2)
    ax.plot(rdf_edges, rdf_vals, lw=2.5, zorder=3, label=r'$g(r)$')
    
    rdfpks = find_peaks(rdf_vals, distance=peak_distance)[0]
    peaks3 = rdf_edges[rdfpks][:3]
    peakcolors = ['r', 'g', 'b']
    peaknames = [r'$\sigma=$', r'$\sqrt{3}\sigma=$', r'$2\sigma=$']
    peakfactors = [1., np.sqrt(3), 2.]
    for n, (peakcolor, peakname, peakfactor, xpeak) in enumerate(zip(peakcolors, peaknames, peakfactors, peaks3)):
        ax.axvline(xpeak, ls='--', lw=1.5, c=peakcolor, zorder=9, 
                   label=f'Peak {n} @ {xpeak:.5f} ({peakname}{peaks3[0]*peakfactor:.5f})')
    
    if showpeak:
        ax.plot(rdf_edges[rdfpks],rdf_vals[rdfpks],'x', c='r', ms=14)
    fit_func, popt, fit_text = get_cffits(rdf_edges[rdfpks[start:]],rdf_vals[rdfpks[start:]]-1, fit_id,r2fit=r2fit)
    ax.plot(rdf_edges[rdfpks[0]:], fit_func(rdf_edges[rdfpks[0]:], *popt)+1, '--', lw=1.5, c='k', zorder=9)
    if xlim is not None:
        ax.set(xlim=xlim)
    ax.set(xlabel=r'$r / \sigma$', ylabel=f'$g(r)$', title=f'Radial Distribution Function $g(r)$  |  {fit_text}')
    ax.legend(shadow=True)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()


def plot_ssf(img_d, bins=50, k_max=8, k_min=0, grid_size=512,dpi=72,realdim=False,savename=False,filetype='svg'):
    """
    Plot static structure factor S(k) in both 1D and 2D representations.
    
    The static structure factor reveals the reciprocal space structure and lattice
    symmetries. Bragg peaks indicate long-range positional order (crystalline phases),
    while their absence suggests liquid-like structure.
    
    Parameters
    ----------
    img_d : imgdata
        Image data container with particle positions
    bins : int, default 50
        Number of radial bins for 1D S(k)
    k_max, k_min : float, default 8, 0
        Range of k-values for 1D analysis
    grid_size : int, default 512
        Resolution of 2D diffraction pattern
    dpi : int, default 72
        Figure resolution
    realdim : bool or list, default False
        Use real dimensions for scaling
    savename : str or False, default False
        Filename to save figure
    filetype : str, default 'svg'
        File format for saving
        
    Notes
    -----
    The static structure factor is defined as:
    S(k) = (1/N)⟨|Σⱼ exp(ik·rⱼ)|²⟩
    
    For crystalline phases, S(k) shows sharp Bragg peaks at reciprocal lattice vectors.
    The 2D pattern reveals lattice symmetry:
    - Triangular lattice: 6-fold symmetric pattern
    - Square lattice: 4-fold symmetric pattern
    - Disordered: Circular/ring-like patterns
    """
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])


    box3d = freud.Box.cube(L=ydim) 
    sf = freud.diffraction.StaticStructureFactorDirect(bins=bins, k_max=k_max, k_min=k_min)
    ssf =  freud.diffraction.DiffractionPattern(grid_size=grid_size)
    sf_vals = []
    ssf_vals = []
    sf.compute(system=(box3d, points))
    sf_edges = sf.bin_edges[:-1]
    sf_vals=sf.S_k
    ssf.compute(system=(box3d, points))
    ssf_vals=ssf.to_image()
    ssfmin, ssfmax = np.min(ssf.k_values), np.max(ssf.k_values)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax[0].plot(sf_edges, sf_vals, 'o', ms=5, c='w', mec='b', zorder=2)
    ax[0].plot(sf_edges, sf_vals, lw=2., zorder=3, label=r'$S(k)$')
    ax[0].set(xlabel=r'$k$', ylabel=r'$S(k)$', title=r'Static Structure Factor')
    
    sfax = ax[1].imshow(ssf_vals, origin='lower', cmap='afmhot', extent=[ssfmin, ssfmax, ssfmin, ssfmax])
    ax[1].set(xlabel=r'$k_{x}$', ylabel=r'$k_{y}$', title=r'2D Diffraction Pattern')
    plt.colorbar(sfax, label=r'$S(\vec{k})$', shrink=0.95, pad=0.02, ax=ax[1])
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()



def plot_ssf2(img_d, k_max=8, k_min=0, grid_size=512,  dpi=72,lower=0.01,upper=1,realdim=False,savename=False,filetype='svg',):
    """
    Plot 2D static structure factor (diffraction pattern) with intensity scaling.
    
    This focused view of the diffraction pattern allows detailed analysis of
    reciprocal space structure with adjustable intensity range for better contrast.
    
    Parameters
    ----------
    img_d : imgdata
        Image data container with particle positions
    k_max, k_min : float, default 8, 0
        Range of k-values (unused for 2D pattern)
    grid_size : int, default 512
        Resolution of diffraction pattern
    dpi : int, default 72
        Figure resolution
    lower, upper : float, default 0.01, 1
        Intensity range as fraction of maximum for display scaling
    realdim : bool or list, default False
        Use real dimensions for scaling
    savename : str or False, default False
        Filename to save figure
    filetype : str, default 'svg'
        File format for saving
        
    Notes
    -----
    Intensity scaling helps visualize weak Bragg peaks that might be obscured
    by the central peak. The lower/upper parameters control the display range
    as fractions of the total intensity.
    """
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    box3d = freud.Box.cube(L=ydim) 
    ssf =  freud.diffraction.DiffractionPattern(grid_size=grid_size)
    ssf_vals = []
    ssf.compute(system=(box3d, points))
    ssf_vals=ssf.to_image(vmin=lower*ssf.N_points,vmax=upper*ssf.N_points)
    ssfmin, ssfmax = np.min(ssf.k_values), np.max(ssf.k_values)
    
    fig, ax = plt.subplots(1, 1, facecolor='w', constrained_layout=True, dpi=dpi)
    sfax = ax.imshow(ssf_vals, origin='lower', cmap='afmhot', extent=[ssfmin, ssfmax, ssfmin, ssfmax])
    ax.set(xlabel=r'$k_{x}$', ylabel=r'$k_{y}$', title=r'2D Diffraction Pattern')
    plt.colorbar(sfax, label=r'$S(\vec{k})$', shrink=0.95, pad=0.02, ax=ax)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()



def estimate_first_shell_G_robust_image(img_d, realdim=False, grid_size=512, 
                                       mask_radius=5, neighborhood_size=3, top_N=50,
                                       intensity_threshold=0.1, min_distance_pixels=10,
                                       lower=0.1, upper=0.9):
    """
    Robustly estimate the first reciprocal lattice vector G from diffraction pattern.
    
    This function performs sophisticated peak detection in reciprocal space to find
    the primary lattice vector, which is essential for translational correlation analysis.
    Multiple filtering steps ensure robust detection even in noisy data.
    
    Parameters
    ----------
    img_d : imgdata
        Image data container with particle positions
    realdim : bool or list, default False
        Use real dimensions for scaling
    grid_size : int, default 512
        Resolution of diffraction pattern
    mask_radius : int, default 5
        Radius to mask around DC component (k=0)
    neighborhood_size : int, default 3
        Size of neighborhood for local maximum detection
    top_N : int, default 50
        Maximum number of peaks to consider
    intensity_threshold : float, default 0.1
        Minimum relative intensity for peak detection
    min_distance_pixels : int, default 10
        Minimum distance between peaks in pixels
    lower, upper : float, default 0.1, 0.9
        Intensity range for diffraction pattern
        
    Returns
    -------
    tuple
        (G_vector, diffraction_image, k_values)
        G_vector : ndarray, shape (2,)
            The primary reciprocal lattice vector [Gx, Gy]
        diffraction_image : ndarray
            The 2D diffraction pattern
        k_values : ndarray
            k-space coordinate values
        
    Notes
    -----
    The algorithm:
    1. Computes 2D diffraction pattern
    2. Masks the DC component (k=0)
    3. Finds local maxima using maximum filter
    4. Applies intensity and distance thresholds
    5. Selects the peak closest to the diffraction pattern center
    
    This G vector is used for translational correlation g_T(r) = ⟨exp(iG·Δr)⟩.
    """
    # Get dimensions and scaling like other phasecheck functions
    if realdim == False:
        xdim, ydim = img_d.img.shape
        xscale = 1
        yscale = 1
    else:
        if img_d.sxm == True:
            xdim = img_d.dims[0]
            ydim = img_d.dims[1]
        else:
            xdim = realdim[0]
            ydim = realdim[1]
        xscale = xdim / img_d.img.shape[0]
        yscale = ydim / img_d.img.shape[1]

    # Get positions like other phasecheck functions
    xydata = np.array([[img_d.data['x'][i]*xscale, img_d.data['y'][i]*yscale] 
                       for i in range(img_d.data['x'].size)])
    points = np.array([[img_d.data['x'][i]*xscale, img_d.data['y'][i]*yscale, 0] 
                       for i in range(img_d.data['x'].size)])

    # Use smaller dimension for box size (like other functions)
    box_size = min(xdim, ydim)
    
    box = freud.Box.cube(box_size)
    ssf = freud.diffraction.DiffractionPattern(grid_size=grid_size)
    ssf.compute((box, points))
    image = ssf.to_image(vmin=lower * ssf.N_points, vmax=upper * ssf.N_points)
    
    # flatten channels if needed
    if image.ndim == 3:
        print(f"WARNING: image was {image.shape}, flattening to image[:,:,0]")
        image = image[:, :, 0]
    print("Final image shape:", image.shape)
    
    k_vals = ssf.k_values
    kx_grid, ky_grid = np.meshgrid(k_vals, k_vals)
    center = grid_size // 2
    Y, X = np.ogrid[:grid_size, :grid_size]
    dc_mask = (X - center)**2 + (Y - center)**2 <= mask_radius**2
    image[dc_mask] = 0
    
    local_max = maximum_filter(image, size=neighborhood_size) == image
    local_max[dc_mask] = False
    peak_coords = np.column_stack(np.nonzero(local_max))
    peak_intensities = image[peak_coords[:, 0], peak_coords[:, 1]]
    
    #  apply intensity threshold
    valid = peak_intensities >= intensity_threshold
    peak_coords = peak_coords[valid]
    peak_intensities = peak_intensities[valid]
    print("Number of detected peaks after threshold:", len(peak_coords))
    
    # Sort peaks by intensity descending
    sorted_indices = np.argsort(-peak_intensities)
    sorted_coords = peak_coords[sorted_indices]
    
    #  enforce min distance between peaks
    selected_peaks = []
    for coord in sorted_coords:
        y = int(coord[0])
        x = int(coord[1])
        if all(np.hypot(y - py, x - px) >= min_distance_pixels for py, px in selected_peaks):
            selected_peaks.append((y, x))
            if len(selected_peaks) >= top_N:
                break
    print("Peaks kept after min distance filter:", len(selected_peaks))
    
    #  find the peak closest to center
    min_distance = np.inf
    best_peak = None
    for y, x in selected_peaks:
        distance = np.hypot(y - center, x - center)
        if distance < min_distance:
            min_distance = distance
            best_peak = (y, x)
    
    y, x = best_peak
    Gx = kx_grid[y, x]
    Gy = ky_grid[y, x]
    return np.array([Gx, Gy]), image, k_vals


def compute_translational_correlation_image(positions, boxsize, G, r_max=None, 
                                           nbins=100, periodic=False):
    """
    Compute translational correlation function g_T(r) for given lattice vector G.
    
    This function calculates how translational order (characterized by G) persists
    over distance. Unlike bond-orientational correlations, this directly probes
    positional order and is essential for distinguishing crystalline from 
    non-crystalline phases.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions in real space
    boxsize : float
        System size for periodic boundary conditions
    G : ndarray, shape (2,)
        Reciprocal lattice vector from estimate_first_shell_G_robust_image
    r_max : float, optional
        Maximum distance for correlation analysis
        If None, uses half the maximum system dimension
    nbins : int, default 100
        Number of radial bins
    periodic : bool, default False
        Use periodic boundary conditions
        
    Returns
    -------
    tuple
        (r_centers, |g_T(r)|, bin_counts)
        r_centers : ndarray
            Center of each radial bin
        |g_T(r)| : ndarray
            Magnitude of translational correlation function
        bin_counts : ndarray
            Number of pairs in each bin for statistics
        
    Notes
    -----
    The translational correlation function is:
    g_T(r) = ⟨exp(iG·Δr)⟩
    
    where Δr is the separation vector between particle pairs.
    
    For crystalline phases: |g_T(r)| remains finite at large r
    For non-crystalline phases: |g_T(r)| → 0 as r → ∞
    
    The function handles both periodic and non-periodic boundary conditions,
    with non-periodic useful for finite experimental samples.
    """
    N = len(positions)
    if r_max is None:
        r_max = 0.5 * np.max(np.ptp(positions, axis=0))
    
    bin_edges = np.linspace(0, r_max, nbins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    gT_r = np.zeros(nbins, dtype=np.complex128)
    counts = np.zeros(nbins, dtype=int)
    
    if periodic:
        box = freud.box.Box(Lx=boxsize, Ly=boxsize, is2D=True)
    else:
        # For non-periodic like phasecheck.py does in plot_bocf
        box = freud.box.Box(Lx=boxsize*20, Ly=boxsize*20, is2D=True)
    
    for i in range(N):
        dr = positions[i+1:] - positions[i]
        dr_padded = np.hstack((dr, np.zeros((dr.shape[0], 1))))  # (N,2) → (N,3)
        dr_wrapped = box.wrap(dr_padded)
        dr = dr_wrapped[:, :2]  # strip z-component back off
        distances = np.linalg.norm(dr, axis=1)
        phase = np.exp(1j * (dr @ G))
        
        bin_idx = np.searchsorted(bin_edges, distances, side='right') - 1
        valid = (bin_idx >= 0) & (bin_idx < nbins)
        
        for b in np.unique(bin_idx[valid]):
            mask = bin_idx == b
            gT_r[b] += np.sum(phase[mask])
            counts[b] += np.sum(mask)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        gT_r /= counts
        gT_r[counts == 0] = 0
    
    return r_centers, np.abs(gT_r), counts


def plot_tocf(img_d, bins=100, realdim=False, fit_id=None, upper_bounds=[np.inf, np.inf],
              dpi=72, savename=False, filetype='svg', start=7, p_dist=1, 
              xscale='linear', yscale='log', r2fit=1, figsize_=(10, 6), 
              ylim=None, plot=True, periodic=False):
    """
    Plot translational order correlation function g_T(r) for crystalline phase detection.
    
    This function measures how translational order (lattice periodicity) persists over
    distance. It's the key diagnostic for distinguishing crystalline from non-crystalline
    phases in the KTHNY theory:
    - Crystal: g_T(r) → constant (long-range translational order)
    - Hexatic/Liquid: g_T(r) → 0 (no long-range translational order)
    
    Parameters
    ----------
    img_d : imgdata
        Image data container with particle positions
    bins : int, default 100
        Number of radial bins for correlation function
    realdim : bool or list, default False
        Use real dimensions for scaling
    fit_id : int or None, default None
        Force specific fit: 0=constant, 1=power law, 2=exponential
    upper_bounds : list, default [np.inf, np.inf]
        Upper bounds for fitting parameters
    dpi : int, default 72
        Figure resolution
    savename : str or False, default False
        Filename to save figure
    filetype : str, default 'svg'
        File format for saving
    start : int, default 7
        Peak index to start fitting from
    p_dist : int, default 1
        Minimum peak separation for peak finding
    xscale, yscale : str, default 'linear', 'log'
        Axis scaling for the plot
    r2fit : int, default 1
        Fit criterion: 0=R², 1=RMSE, 2=combined
    figsize_ : tuple, default (10, 6)
        Figure size
    ylim : tuple or None, default None
        Y-axis limits
    plot : bool, default True
        Whether to create plot or just return fit parameters
    periodic : bool, default False
        Use periodic boundary conditions
        
    Returns
    -------
    tuple
        (fit_parameters, best_fit_id, rmse_scores, figure) if plot=True
        (fit_parameters, best_fit_id, rmse_scores) if plot=False
        
    Notes
    -----
    The algorithm:
    1. Estimates reciprocal lattice vector G from diffraction pattern
    2. Computes g_T(r) = ⟨exp(iG·Δr)⟩ for all particle pairs
    3. Fits correlation decay to identify phase behavior
    4. Displays both correlation function and diffraction pattern with G vector
    
    The translational correlation function directly probes lattice periodicity,
    making it complementary to bond-orientational analysis. Together, these
    provide complete phase identification for 2D systems.
    """
    # Get dimensions and scaling like other phasecheck functions
    if realdim == False:
        xdim, ydim = img_d.img.shape
        xscale_coord = 1
        yscale_coord = 1
    else:
        if img_d.sxm == True:
            xdim = img_d.dims[0]
            ydim = img_d.dims[1]
        else:
            xdim = realdim[0]
            ydim = realdim[1]
        xscale_coord = xdim / img_d.img.shape[0]
        yscale_coord = ydim / img_d.img.shape[1]

    # Get positions like other phasecheck functions
    xydata = np.array([[img_d.data['x'][i]*xscale_coord, img_d.data['y'][i]*yscale_coord] 
                       for i in range(img_d.data['x'].size)])
    
    # Try multiple lower values to find G (like your original)
    lower_values = [0.5, 0.1, 0.001]
    G = None
    
    for lower_try in lower_values:
        try:
            G, image, k_vals = estimate_first_shell_G_robust_image(
                img_d, realdim=realdim, grid_size=400, 
                mask_radius=5, neighborhood_size=3, top_N=50,
                intensity_threshold=0.1, min_distance_pixels=3,
                lower=lower_try, upper=1)
            print(f"Success with lower={lower_try}")
            break
        except Exception as e:
            print(f"Failed with lower={lower_try}, error: {e}")
    
    if G is None:
        raise RuntimeError("All attempts failed to find G.")
    
    print('This is G')
    print(G)
    
    # Use smaller dimension for calculations
    box_dim = min(xdim, ydim)
    r_vals, gT_vals, counts = compute_translational_correlation_image(
        xydata, box_dim, G, r_max=box_dim/2, nbins=bins, periodic=periodic)
    
    k_min, k_max = k_vals.min(), k_vals.max()
    cf_edges = r_vals
    cf_vals = gT_vals
    
    cfpks = find_peaks(cf_vals, distance=p_dist)[0]
    if len(cfpks) <= start:
        print(f"Warning: Only found {len(cfpks)} peaks, but start={start}")
        start = max(0, len(cfpks) - 1)
    
    if len(cfpks) > start:
        idxlim = np.argwhere(cf_edges == cf_edges[cfpks][start])[0][0] - 1
        fit_func, popt, fit_text, min_id, rmse_scores = get_cffits(
            cf_edges[cfpks[start:]], cf_vals[cfpks[start:]],
            fit_id, r2fit=r2fit, upper_bounds=upper_bounds, sigma=None)
    else:
        popt, min_id, rmse_scores = None, None, None
        fit_text = "Insufficient peaks for fitting"
    
    if plot == False:
        return popt, min_id, rmse_scores
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='w', 
                                   constrained_layout=True, dpi=dpi)
    
    # Plot correlation function
    if ylim != None:
        ax1.set_ylim(ylim)
    if len(cfpks) > 0:
        ax1.set(xlim=(cf_edges[cfpks[0]]*0.75, cf_edges[-1]*1.1))
    ax1.set(xscale=xscale)
    ax1.set(yscale=yscale)
    ax1.plot(cf_edges, cf_vals, 'o', ms=8, c='w', mec='b', zorder=2)
    ax1.plot(cf_edges, cf_vals, lw=2.5, c='b', zorder=3)
    
    if len(cfpks) > start:
        ax1.plot(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], 'x', c='r', ms=14)
        ax1.plot(cf_edges[idxlim:], np.exp(fit_func(cf_edges[idxlim:], *popt)), 
                 '--', lw=2., c='k', zorder=9)
    
    ax1.set(xlabel=r'$r$', ylabel=f'$g_{{q}}(r)$', 
            title=f'Translational Correlation Function $g_{{q}}(r)$  |  {fit_text}')
    
    # Plot diffraction pattern with G vector
    ax2.imshow(image, origin='lower', extent=[k_min, k_max, k_min, k_max], cmap='afmhot')
    ax2.scatter(G[0], G[1], color='cyan', s=50, marker='x', 
                label=f'|G| ≈ {np.linalg.norm(G):.2f}')
    ax2.set(xlabel=r'$k_x$', ylabel=r'$k_y$', title='Diffraction Pattern with G vector')
    ax2.legend()
    
    if savename != False:
        fig.savefig(savename, format=filetype)
    
    return popt, min_id, rmse_scores, fig