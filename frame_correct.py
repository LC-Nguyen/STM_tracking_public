import cv2
import trackpy as tp
import numpy as np
from sxmreader import SXMReader
import matplotlib.pyplot as plt
import math
import matplotlib
import matplotlib.animation

def add_crop(frame,shift_x,shift_y,output_crop):
    """Add grey padding space of size outputcrop, and shift the frame."""
    input_crop=frame.shape[1]
    image_shift=(output_crop-input_crop)/2 #width of grey space from crop

    M = np.array([[1, 0,shift_x+image_shift],
              [0, 1, shift_y+image_shift]], dtype=np.float32)
    greypadded = cv2.warpAffine(frame, M, (output_crop, output_crop),
                        flags=cv2.INTER_LINEAR,)
    return greypadded

def mask_overlap(shift_1,shift_2,output_crop,input_crop):
    """use the add_crop generated frames, finds the overlap"""
    image_shift=(output_crop-input_crop)/2
    left=math.ceil(max(shift_1[0],shift_2[0],)+image_shift)
    right=math.floor(min(shift_1[0],shift_2[0])+input_crop+image_shift)
    up=math.floor(min(shift_1[1],shift_2[1])+image_shift+input_crop)
    down=math.ceil(max(shift_1[1],shift_2[1],)+image_shift)
    if left<0:
        print('potential image cut off. Increase padding')
        left=0
    if down<0:
        print('potential image cut off. Increase padding')
        down=0
    if right>output_crop:
        print('potential image cut off. Increase padding')
        right=output_crop
    if up>output_crop:
        print('potential image cut off. Increase padding')
        up=output_crop
    mask = np.zeros((output_crop, output_crop), dtype=np.uint8)  # Use entire image
    mask[down:up,left:right,]=1
    return mask

def shift_from_drift(frames,search_params):
    """Use trackpy to get drift of molecules to estimate shift"""
    molecule_size, min_mass, max_mass, separation, min_size,max_ecc, adaptive_stop, search_range,threshold,_=search_params
    
    f = tp.batch(frames, molecule_size, minmass=min_mass, separation=separation,threshold=threshold,engine='python')
    t = tp.link(f, search_range=search_range, adaptive_stop=adaptive_stop,memory=7,)
    drifts=tp.compute_drift(t)
    shift_y=-drifts['y']
    shift_x=-drifts['x']
    shift_y[0]=0
    shift_x[0]=0

    return shift_x,shift_y
def shift_from_phase(frame1, frame2,):
    """
    Compute the translation aligning 'image' to 'template'.
    using phase correlation.
    """

    (shift_x, shift_y), response = cv2.phaseCorrelate(frame1, frame2)
    return -shift_x,-shift_y
def shift_from_ECC(frame1,frame2,mask):
    warp_mode = cv2.MOTION_TRANSLATION
    # Initialize the warp matrix as identity.
    warp_matrix = warp_matrix = np.array([[1, 0,0], #drifts returns y,x
                        [0, 1, 0]], dtype=np.float32)
    # Set ECC termination criteria: maximum iterations and epsilon.
    number_of_iterations = 5000
    termination_eps = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    # Convert images to float32 (ECC works best with float images).
    template_f = frame1.astype(np.float32)
    image_f = frame2.astype(np.float32)
    # findTransformECC returns the enhanced correlation coefficient and warp matrix.
    cc, warp_matrix = cv2.findTransformECC(template_f, image_f, warp_matrix,
                                             warp_mode, criteria,inputMask=mask)
    shifty = warp_matrix[1, 2]
    shiftx = warp_matrix[0, 2]
    return -shiftx, -shifty
def mask_from_points(t,frame,diameter,size):
    """Creates a mask so that the image alignment doesnt look at the particles
    t is df from trackpy.link"""
    mask = np.ones((size, size), dtype=np.uint8)  # Use entire image
    t_=t[t['frame']==frame]
    x=list(t_['x'])
    y=list(t_['y'])
    for i in range(len(x)):
        cv2.circle(mask, (int(x[i]), int(y[i])), int(diameter/2), 0, thickness=-1)
    return mask
def _ecc_align_pair(i, frame_prev, frame_curr, mask_prev, mask_curr, overlap_mask):
    try:
        mask = mask_prev * mask_curr * overlap_mask
        dx, dy = shift_from_ECC(frame_prev, frame_curr, mask)

    except Exception as e:
        print(f"ECC failed at frame {i}, falling back to phase: {e}")
        dx, dy = shift_from_phase(frame_prev, frame_curr)
    return i, dx, dy
from concurrent.futures import ThreadPoolExecutor, as_completed


def Frame_correct(frames_, output_crop, search_params, input_crop=128, cumulative=None):
    molecule_size, min_mass, max_mass, separation, min_size, max_ecc, adaptive_stop, search_range, threshold, _ = search_params

    if input_crop is None:
        input_crop = frames_[0].shape[1]

    frames = [add_crop(frame, 0, 0, output_crop).astype(np.float32) for frame in frames_]

    # --- Compute drift on original frames (preserves old behavior) ---
    f = tp.batch(frames, molecule_size, minmass=min_mass, separation=separation, threshold=threshold, engine='python')
    t = tp.link(f, search_range=search_range, adaptive_stop=adaptive_stop, memory=7)

    drifts = tp.compute_drift(t)
    cumulative_x =np.array([0]+list(-drifts['x']))
    cumulative_y =np.array( [0]+list(-drifts['y']))
    # --- First correction: drift ---
    firstcorrection = [frames[0]]
    for i in range(1, len(frames)):
        firstcorrection.append(add_crop(frames[i], cumulative_x[i], cumulative_y[i], output_crop))
    # --- Second correction: phase correlation ---
    secondcorrection = [firstcorrection[0]]
    cumulative_translation = np.array([0, 0], dtype=np.float32)
    for i in range(1, len(firstcorrection)):
        shiftx, shifty = shift_from_phase(firstcorrection[i - 1], firstcorrection[i]) if cumulative is None else (0, 0)
        max_shift = 30  # pixels - adjust based on your typical drift
        if abs(shiftx) > max_shift:
            shiftx = max_shift if shiftx > 0 else -max_shift
        if abs(shifty) > max_shift:
            shifty = max_shift if shifty > 0 else -max_shift
        cumulative_translation += np.array([shiftx, shifty], dtype=np.float32)
        cumulative_x[i] += cumulative_translation[0]
        cumulative_y[i] += cumulative_translation[1]
        M = np.array([[1, 0, cumulative_translation[0]], [0, 1, cumulative_translation[1]]], dtype=np.float32)
        warped = cv2.warpAffine(firstcorrection[i], M, (output_crop, output_crop), flags=cv2.INTER_LINEAR)
        secondcorrection.append(warped)
    # --- Third correction: ECC alignment (parallel only) ---
    thirdcorrection = [secondcorrection[0]]
    cumulative_translation2 = np.array([0.0, 0.0])
    cumulative_xfinal = [cumulative_x[0]]
    cumulative_yfinal = [cumulative_y[0]]

    # Particle positions on second-corrected frames
    t = tp.batch(secondcorrection, molecule_size, minmass=min_mass, separation=separation, threshold=threshold, engine='python')
    masks = [mask_from_points(t, i, molecule_size, output_crop) for i in range(len(frames))]
    overlap_masks = {}
    for i in range(1, len(frames)):
        if cumulative is None:
            overlap_masks[(i - 1, i)] = mask_overlap((cumulative_x[i - 1], cumulative_y[i - 1]),
                                                     (cumulative_x[i], cumulative_y[i]),
                                                     output_crop, input_crop)
            
        else:
            overlap_masks[(i - 1, i)] = mask_overlap((cumulative_x[i - 1] + cumulative[0][i - 1],
                                                      cumulative_y[i - 1] + cumulative[1][i - 1]),
                                                     (cumulative_x[i] + cumulative[0][i],
                                                      cumulative_y[i] + cumulative[1][i]),
                                                     output_crop, input_crop)

    # --- Parallel ECC calls only ---
    results = [None] * (len(frames) - 1)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(1, len(frames)):
            futures.append(executor.submit(_ecc_align_pair,
                                           i,
                                           secondcorrection[i - 1],
                                           secondcorrection[i],
                                           masks[i - 1],
                                           masks[i],
                                           overlap_masks[(i - 1, i)]))
        for future in as_completed(futures):
            i, dx, dy = future.result()
            results[i - 1] = (dx, dy)

    # --- Apply ECC shifts cumulatively ---
    for i in range(1, len(frames)):
        dx, dy = results[i - 1]
        cumulative_translation2 += np.array([dx, dy], dtype=np.float32)
        cumulative_xfinal.append(cumulative_x[i] + cumulative_translation2[0])
        cumulative_yfinal.append(cumulative_y[i] + cumulative_translation2[1])
        M2 = np.array([[1, 0, cumulative_xfinal[i]],
                       [0, 1, cumulative_yfinal[i]]], dtype=np.float32)
        warped2 = cv2.warpAffine(frames[i], M2, (output_crop, output_crop), flags=cv2.INTER_LINEAR)
        thirdcorrection.append(warped2)

    if cumulative is None:
        cumulative = [np.zeros_like(cumulative_x), np.zeros_like(cumulative_y)]
    final_x = cumulative_xfinal + np.array(cumulative[0])
    final_y = cumulative_yfinal + np.array(cumulative[1])
    
    return firstcorrection, secondcorrection, thirdcorrection, (final_x, final_y)

def Frame_correct_loop(frames_,search_params,output_crop=200,steps=1,adjust=[[],[]]):
    """
    Iteratively corrects drift in a sequence of image frames using pairwise alignment.

    Parameters
    ----------
    frames_ : list of ndarray
        List of grayscale image frames (2D numpy arrays) to be aligned.
    search_params : dict or any
        Parameters used by trackpy.
        molecule_size, min_mass, max_mass, separation, min_size,max_ecc, adaptive_stop, search_range,threshold,threshold
    output_crop : int, optional
        Size of the output image after padding. The aligned frame is centered in a grey-padded canvas
        of shape (output_crop, output_crop). Default is 200.
    steps : int, optional
        Number of iterative alignment passes to perform. Default is 1.
    adjust : list of lists, optional
        Manual adjustments to apply after alignment.
        - adjust[0] should be a list of frame indices to adjust.
        - adjust[1] should be a list of (dx, dy) shifts to apply to the corresponding frames.

    Returns
    -------
    frames : list of ndarray
        List of drift-corrected, grey-padded frames as float32 arrays.
    """
    input_crop=frames_[0].shape[1]
    cumulative=None
    frames=frames_
    for i in range(steps):
        frames_1,frames_2,frames,cumulative=Frame_correct(frames,output_crop,search_params,input_crop=input_crop,cumulative=cumulative)

        for j in range(len(frames)):
            frames[j]=add_crop(frames_[j],cumulative[0][j],cumulative[1][j],output_crop).astype(np.float32)
    for index,n in enumerate(adjust[0]):
        frames=manual_adjust(frames,n,adjust[1][index])
    return frames
def manual_adjust(frames_,frame_after_jump,jump_vector):
    frames=frames_.copy()
    output_crop=frames[frame_after_jump].shape[1]
    for i in range(frame_after_jump,len(frames)):
        frames[i]=add_crop(frames[i],-jump_vector[0],-jump_vector[1],output_crop)

    return frames
import matplotlib
import matplotlib.pyplot as plt

def measure_shift(frames, jumped_frame):
    # Save original backend
    original_backend = matplotlib.get_backend()
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')

    clicked_points = []
    scatter_artists = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Press Enter to confirm, Esc to cancel, Z to undo')
    axes[0].imshow(frames[jumped_frame - 1], cmap='gray', origin='upper')
    axes[1].imshow(frames[jumped_frame], cmap='gray', origin='upper')
    axes[0].set_title(f"Frame {jumped_frame - 1} — Click first point")
    axes[1].set_title(f"Frame {jumped_frame} — Click second point")

    confirmed = {'done': False, 'canceled': False}

    def on_click(event):
        if event.inaxes and len(clicked_points) < 2:
            point = (event.xdata, event.ydata)
            clicked_points.append(point)
            ax_idx = 0 if event.inaxes == axes[0] else 1
            artist = axes[ax_idx].scatter(*point, color='red', s=60, zorder=5)
            scatter_artists.append(artist)
            fig.canvas.draw()
            print(f"Clicked point {len(clicked_points)}: ({point[0]:.2f}, {point[1]:.2f})")

    def on_key(event):
        key = event.key.lower()
        if key == 'enter':
            if len(clicked_points) == 2:
                confirmed['done'] = True
                print("Selection confirmed. Closing window.")
                fig.canvas.stop_event_loop()
                plt.close(fig)
            else:
                print(f"Please select 2 points before confirming (currently selected {len(clicked_points)}).")
        elif key == 'escape':
            confirmed['canceled'] = True
            print("Selection canceled. Closing window.")
            fig.canvas.stop_event_loop()
            plt.close(fig)
        elif key == 'z':
            if clicked_points:
                removed = clicked_points.pop()
                artist = scatter_artists.pop()
                artist.remove()
                fig.canvas.draw()
                print(f"Removed last point: ({removed[0]:.2f}, {removed[1]:.2f})")
            else:
                print("No points to undo.")

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
    fig.canvas.start_event_loop()

    # Restore original backend
    matplotlib.use(original_backend)
    plt.switch_backend(original_backend)

    if confirmed['canceled']:
        print("Operation canceled by user.")
        return None
    if len(clicked_points) != 2:
        print("Incomplete selection (less than 2 points). Returning None.")
        return None

    dx = clicked_points[1][0] - clicked_points[0][0]
    dy = clicked_points[1][1] - clicked_points[0][1]
    print(f"Shift: Δx = {dx:.2f}, Δy = {dy:.2f}")
    return dx, dy

