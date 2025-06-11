import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def get_data():
    """
    Load hand bone dataset from 'data/handdata.mat'.

    Returns
    -------
    images : list of np.ndarray
        List of grayscale bone images, each as a 2D array (height × width).
        These are transposed upon loading to match orientation of landmarks and aligned.

    masks : list of np.ndarray
        List of binary contour segmentation masks for each image, of the same shape as the corresponding image.
        These masks contain values like 0 (background) and 10 (bone).

    landmarks : list of np.ndarray
        List of raw landmark coordinates, each as a 2D array of shape (2, n_landmarks).
        Matching coordinates for images and masks.

    aligned : np.ndarray
        Aligned landmark coordinates for all samples, shape (n_samples, 2, n_landmarks), preprocessed for PCA.

    Notes
    -----
    - Uses HDF5 file format (`.mat`) via `h5py` to access MATLAB-style structured data.
    - Transposes the images and masks upon loading to ensure correct orientation.
    - Assumes file structure: 'images', 'landmarks', 'masks', and 'aligned' datasets exist in the file.
    - Must ensure the file `data/handdata.mat` is present and accessible.
    """

    with h5py.File('data/handdata.mat', 'r') as f:
        images = []
        landmarks = []
        masks = []
        aligned = f['aligned'][:]

        for i in range(f['images'].shape[0]):
            img_ref = f['images'][i, 0]
            lm_ref = f['landmarks'][i, 0]
            mask_ref = f['masks'][i, 0]

            images.append(f[img_ref][:].T)
            landmarks.append(f[lm_ref][:].T)
            masks.append(f[mask_ref][:].T)

    print(f'images: list of {len(images)} images, each as a 2D array with varying sizes')
    print(f'masks: list of {len(masks)} contour segmentation masks, each as a 2D array with varying sizes - corresponding to images')
    print(f'landmarks: list of {len(landmarks)} raw landmark coordinates, each as a 2D array of shape (2, n_landmarks)')
    print(f'aligned: np.ndarray of shape {aligned.shape} containing aligned landmark coordinates for all samples, preprocessed for PCA')

    return images, masks, landmarks, aligned


def plot_shape(generated_shape, mean_shape=None):
    """
    Plot a 2D shape defined by concatenated x and y coordinates.

    Parameters
    ----------
    generated_shape : np.ndarray
        A 1D array of length 2N, where the first N values represent the x-coordinates
        and the last N values represent the y-coordinates of the generated shape.

    mean_shape : np.ndarray, optional
        A 1D array of length 2N representing the mean shape to overlay for comparison.
        If provided, it will be plotted in addition to the generated shape.

    Notes
    -----
    - The shapes are plotted using `plt.plot` with connecting lines between landmarks.
    - Uses `plt.axis('equal')` to preserve aspect ratio.
    """

    N = mean_shape.size // 2
    x = generated_shape[:N]
    y = generated_shape[N:]

    plt.figure()
    if mean_shape is not None:
        plt.plot(mean_shape[:N], mean_shape[N:], '.-')
    plt.plot(x, y, '.-', label='Generated Shape')
    plt.axis('equal')
    plt.legend()
    plt.show()


def plot_convolutions(images, convolutions, kernels, cmap='gray'):
    """
    Plots original images alongside their corresponding convolution outputs.

    Parameters:
    -----------
    images : list of ndarray
        List of input images to display.

    convolutions : list of list of ndarray
        Precomputed convolution results for each image.
        Each item in the outer list corresponds to one image and contains
        a list of 2D arrays (one per kernel).

    kernels : list, dict, or None
        - If a list: used as titles for the convolution columns.
        - If a dict: keys are used as titles (assumes values were used to generate the convolutions).
        - If None: numerical indices (0, 1, 2, ...) are used as titles.

    cmap : str, optional (default='gray')
        Colormap used for displaying both original and convolved images.

    Notes:
    ------
    - The plot will have one row per input image.
    - The first column in each row is the original image.
    - The following columns are the convolution outputs, one per kernel.
    - Assumes that `len(convolutions[i]) == len(kernels)` (if kernels is not None).
    """

    assert len(images) == len(convolutions)
    if kernels is None:
        n_kernels = len(convolutions[0])
    else:
        n_kernels = len(kernels)
    fig, axes = plt.subplots(len(images), n_kernels + 1, figsize=(2*n_kernels, 8))
    for i, img in enumerate(images):
        axes[i, 0].axis('off')
        axes[i, 0].imshow(img, cmap=cmap)
        if isinstance(kernels, dict):
            names = kernels.keys()
        elif isinstance(kernels, list):
            names = kernels
        else:
            names = range(n_kernels)

        for j, name in enumerate(names):
            ax = axes[i, j+1]
            ax.imshow(convolutions[i][j], cmap=cmap)
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f'Image {i+1}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_prediction_triplets(img, pred, gt, triplets_per_row=3, cmap='gray'):
    """
    Plot triplets of images: [Input | Prediction A | Prediction B], side by side.

    Each row displays `triplets_per_row` triplets, totaling `triplets_per_row * 3` images per row.

    Parameters
    ----------
    img : list or np.ndarray
        List or array of input images (grayscale), shape (N, H, W) or (N, 1, H, W).

    pred : list or np.ndarray
        Predicted masks from model A, same length and shape as `I_test`.

    gt : list or np.ndarray
        Predicted masks from model B, same length and shape as `I_test`.

    triplets_per_row : int, optional
        Number of [Input | Pred A | Pred B] groups to display per row. Default is 3.

    cmap : str, optional
        Colormap used for displaying images. Default is 'gray'.

    Raises
    ------
    AssertionError
        If the lengths of the input lists/arrays are not equal.

    Notes
    -----
    This function is useful for comparing multiple model predictions visually.
    It disables axis ticks and lays out the results using `matplotlib.pyplot`.

    """
    assert len(img) == len(pred) == len(gt), "All input arrays must have the same length"

    # remove singleton channel dimension if present (N, 1, H, W) -> (N, H, W)
    if  isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[1] == 1:
        img = img[:, 0]
    if isinstance(pred, np.ndarray) and pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if isinstance(gt, np.ndarray) and gt.ndim == 4 and gt.shape[1] == 1:
        gt = gt[:, 0]

    N = len(img)
    cols = triplets_per_row * 3
    rows = math.ceil(N / triplets_per_row)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 2))
    if axes.ndim == 1:
        axes = axes.reshape((rows, cols))

    for i in range(N):
        row = i // triplets_per_row
        base_col = (i % triplets_per_row) * 3

        axes[row, base_col].imshow(img[i], cmap=cmap)
        axes[row, base_col].set_title("Input")
        axes[row, base_col].axis('off')

        axes[row, base_col + 1].imshow(pred[i], cmap=cmap)
        axes[row, base_col + 1].set_title("Prediction")
        axes[row, base_col + 1].axis('off')

        axes[row, base_col + 2].imshow(gt[i], cmap=cmap)
        axes[row, base_col + 2].set_title("Ground Truth")
        axes[row, base_col + 2].axis('off')

    # hide unused axes (if N is not a multiple of triplets_per_row)
    for j in range(N, rows * triplets_per_row):
        row = j // triplets_per_row
        base_col = (j % triplets_per_row) * 3
        for k in range(3):
            axes[row, base_col + k].axis('off')

    plt.tight_layout()
    plt.show()

def show_feature_importance(clf, labels):
    """
    Display a bar chart of feature importances from a trained Random Forest classifier.

    Parameters
    ----------
    clf : RandomForestClassifier
        A fitted Random Forest model with `feature_importances_` attribute.

    labels : list of str
        List of feature names corresponding to the input features used to train the model.
        The length must match the number of features in the classifier.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [labels[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
    plt.xticks(range(len(sorted_importances)), sorted_names, rotation=90, fontsize=6)
    plt.ylabel('Relative Importance')
    plt.title('RF Feature Importances')
    plt.tight_layout()
    plt.show()


def evaluate_binary_segmentation(predictions, ground_truths, show_confusion_matrix=True, threshold=0):
    """
    Evaluate binary segmentation over a list of predicted and ground truth masks.

    Computes precision, recall, F1 score, Dice coefficient, and displays the confusion matrix.

    Parameters
    ----------
    predictions : list of np.ndarray
        List of predicted binary masks. Non-zero values are treated as positive.
    ground_truths : list of np.ndarray
        List of ground truth binary masks. Non-zero values are treated as positive.
    cm_percent : bool
        Show confusion matrix in percent. Default is True.
    show_confusion_matrix : bool, optional
        Whether to display the confusion matrix plot (default is True).

    Returns
    -------
    metrics : dict
        Dictionary containing precision, recall, F1 score, Dice coefficient, and confusion matrix.
    """
    # flatten and stack all predictions and ground truths
    pred_bin = np.hstack([(p > threshold).astype(int).ravel() for p in predictions])
    gt_bin = np.hstack([(g > 0).astype(int).ravel() for g in ground_truths])

    # Compute metrics
    precision = precision_score(gt_bin, pred_bin)
    recall = recall_score(gt_bin, pred_bin)
    f1 = f1_score(gt_bin, pred_bin)
    dice = f1  # dice = F1 for binary classification

    cm = confusion_matrix(gt_bin, pred_bin, normalize='true')
    cm_plot = cm

    # visualization
    if show_confusion_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_plot, display_labels=["Background", "Object"])
        disp.plot(values_format='.2%')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice': dice,
        'confusion_matrix': cm
    }

def optimize(f, mi, ma, draw_function=None, n=500, max_iter=int(1e6), random_state=None, min_iterations=20):
    """
    Differential Evolution–like optimizer for continuous parameter optimization.

    Parameters
    ----------
    f : callable
        Cost function to minimize. Takes a 1D NumPy array `p` of shape (d,) and returns a scalar.
    mi : array_like, shape (d,)
        Lower bounds for each parameter.
    ma : array_like, shape (d,)
        Upper bounds for each parameter.
    draw_function : callable, optional
        A function draw_function(pop, best_index) for visualizing the current population.
    n : int, optional
        Number of candidate solutions (population size). Default is 100.
    max_iter : int, optional
        Maximum number of iterations before termination.
    random_state : int, np.random.Generator, or None
        If int, used to seed a new NumPy RNG. If Generator, used directly. If None, use default RNG.

    Returns
    -------
    best_params : np.ndarray, shape (d,)
        Best found parameter vector minimizing the objective function.

    Notes
    -----
    - This is a simplified version of a differential evolution optimizer.
    - It uses mutation of the form: a + F * (c - b), with F=0.85.
    - Worst candidates are periodically reset toward the current best with noise.
    - Stopping criteria are based on cost stagnation and iteration limit.
    """

    mi, ma = mi.ravel(), ma.ravel()
    assert mi.shape == ma.shape, "Bounds must have the same shape"
    dim = mi.size

    # Initialize RNG
    if isinstance(random_state, int):
        rng = np.random.default_rng(seed=random_state)
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng()

    # Initialize population
    pop = (rng.standard_normal((dim, n)) / 6) * (ma - mi)[:, None] + ((ma + mi) / 2)[:, None]
    pop = np.clip(pop, mi[:, None], ma[:, None])

    costs = np.array([f(pop[:, i]) for i in range(n)])

    stable_iters = 0
    last_costs = np.full(10000, np.inf)
    itr = 0

    while itr < max_iter and stable_iters < 1e6:
        itr += 1

        # Mutation
        idx = rng.integers(0, n, 3)
        candidate = pop[:, idx[0]] + 0.85 * (pop[:, idx[2]] - pop[:, idx[1]])
        candidate = np.clip(candidate, mi, ma)

        cost_candidate = f(candidate)

        if cost_candidate < costs[idx[0]]:
            pop[:, idx[0]] = candidate
            costs[idx[0]] = cost_candidate
            stable_iters = 0
        else:
            stable_iters += 1

        best_cost = np.min(costs)
        last_costs[itr % last_costs.size] = best_cost

        if itr > min_iterations:
            window = last_costs[max(0, itr - 50):itr]
            window = window[np.isfinite(window)]
            if window.size > 10 and np.std(window) < 1e-6:
                break

        if draw_function is not None and itr % 10 == 0:
            draw_function(pop, np.argmin(costs))

        worst_indices = np.argsort(costs)[int(0.8 * n):]
        best_particle = pop[:, np.argmin(costs)][:, None]
        noise = 0.005 * (ma - mi)[:, None] * rng.standard_normal((dim, len(worst_indices)))
        pop[:, worst_indices] = np.clip(best_particle + noise, mi[:, None], ma[:, None])

    return pop[:, np.argmin(costs)]

def plot_fitted_shapes(shapes, segmentations, gt_landmarks=None, points_per_shape=None,
                       figsize=(8, 12), color='r', gt_color='g'):
    """
    Plot fitted shapes over corresponding segmentation masks, 4 per row,
    with optional ground truth landmarks overlay.

    Parameters
    ----------
    shapes : list of np.ndarray
        List of shape vectors, each of shape (2N,).
    segmentations : list of np.ndarray
        List of 2D segmentation masks (H×W), same length as `shapes`.
    points_per_shape : int, optional
        Number of (x, y) points per shape (N). If None, inferred from shape size.
    gt_landmarks : list of np.ndarray, optional
        List of ground truth landmarks, each of shape (2, N), where 2 rows = [x, y].
    figsize : tuple, optional
        Size of the entire figure. Default is (14, 14).
    color : str, optional
        Color for fitted shape points. Default is 'r' (red).
    gt_color : str, optional
        Color for ground truth landmark points. Default is 'g' (green).
    """
    assert len(shapes) == len(segmentations), "Mismatch: shapes and segmentations must be the same length."
    if gt_landmarks is not None:
        assert len(gt_landmarks) == len(shapes), "Mismatch: gt_landmarks must match shapes in length."

    n_shapes = len(shapes)
    if n_shapes == 0:
        print("No shapes to display.")
        return

    if points_per_shape is None:
        points_per_shape = shapes[0].shape[0] // 2

    n_cols = 4
    n_rows = math.ceil(n_shapes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx in range(n_shapes):
        ax = axes[idx]
        seg = segmentations[idx]
        shape = shapes[idx]
        x = shape[:points_per_shape]
        y = shape[points_per_shape:]

        ax.imshow(seg, cmap='gray')
        ax.scatter(x, y, c=color, s=5, label='Fitted')

        if gt_landmarks is not None:
            gt = gt_landmarks[idx]  # shape: (2, N)
            x_gt, y_gt = gt[0], gt[1]
            ax.scatter(x_gt, y_gt, c=gt_color, s=5, marker='x', label='GT')

        ax.set_title(f"Shape {idx}")
        ax.axis('off')
        ax.set_aspect('equal')

    # Hide unused subplots
    for j in range(n_shapes, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()