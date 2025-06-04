from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from matplotlib.colors import to_hex
import pandas as pd
import numpy as np
import matplotlib
import itertools
import math
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

p0 = [48, 0.00001, 0, 0]
bounds=[[38, 0, 0, -10],[48, 10, np.inf, 10]]
maxfev = 10000000
nb_days_per_year = 365.25

def sigmoid(x, b, k, a, c):
    """
    Sigmoid function.
    :param x: 1D array data.
    :param b: first hyperparameter.
    :param k: second hyperparameter.
    :param a: third hyperparameter.
    :return: 1D array y data.
    """
    return (b) / (1 + np.exp((k * (x - a)))) + c

def get_median_after_x_days(list_timestamps, list_values, x_to_predict=183):

    predicted_values = []

    for (timestamps, values) in zip(list_timestamps, list_values):
        
        bounds[1][2] = timestamps[-1]

        popt, pcov, infodict, i1, i2 = curve_fit(
            sigmoid,
            timestamps,
            values,
            bounds=bounds,
            p0=p0,
            full_output=True,
            maxfev=maxfev
        )

        y_pred = sigmoid(x_to_predict, *popt)
        if y_pred < 0: y_pred = 0
        predicted_values.append(y_pred)

    return np.percentile(predicted_values, 90)

def get_D50(id_patient, timestamps, values):
    """
    Get timestamp where ALSFRS-R value is equal to 24
    :param timestamps: 1D array of timestamps in days.
    :param values: 1D array of ALSFRS-R values
    :return: timestamp where ALSFRS-R score is equal to 24.
    """

    if values[0] < 24: return 0, pd.NA

    for timestamp, value in zip(timestamps, values):
        if value == 24: return timestamp/nb_days_per_year, pd.NA

    bounds[1][2] = timestamps[-1]

    popt, pcov, infodict, i1, i2 = curve_fit(
        sigmoid,
        timestamps,
        values,
        bounds=bounds,
        p0=p0,
        full_output=True,
        maxfev=maxfev
        )
    
    b, k, a, c = popt[0], popt[1], popt[2], popt[3]
    residuals = infodict["fvec"]
    error = sum(np.power(residuals,2))

    def solve_for_x(b, k, a, c, y=24):
        term = (b / (y - c)) - 1
        x = (np.log(term) / k) + a
        return x
    
    x_solution = solve_for_x(*popt, y=24)
    x_solution_in_years = x_solution/nb_days_per_year

    if id_patient==297:
        print("------------------------------")
        print("timestamps=", timestamps)
        print("values=", values)
        print("bounds=", bounds)
        print("(b,k,a,c)=", (b,k,a,c))

    return x_solution_in_years, error

def get_percentage_decrease_after_x_days(timestamps, values, x_to_predict=183):
    
    """
    Get percentage of decrease after x days since first ALSFSR-R score.
    :param timestamps: 1D array of timestamps in days.
    :param values: 1D array of ALSFRS-R values.
    :param x_to_predict: Number of days elapsed to get predicted value.
    :return: Predicted Percentage of decrease.
    """

    bounds[1][2] = timestamps[-1]

    popt, pcov, infodict, i1, i2 = curve_fit(
    sigmoid, 
    timestamps, 
    values,
    p0, 
    bounds=bounds, 
    full_output=True,
    maxfev=maxfev
    )

    y_init = sigmoid(0, *popt)
    y_pred = sigmoid(x_to_predict, *popt)
    if y_pred < 0: y_pred = 0
    percentage = ((y_init - y_pred)/y_init) * 100
    return percentage

def get_alsfrsr_score_after_x_days(timestamps, values, x_to_predict=183):
    """
    Get ALSFSR-R score after x days since first record.
    :param timestamps: 1D array of timestamps in days.
    :param values: 1D array of ALSFRS-R values
    :param x_to_predict: Number of days elapsed to get predicted value.
    :return: Predicted ALSFRS-R score.
    """

    bounds[1][2] = timestamps[-1]

    popt, pcov, infodict, i1, i2 = curve_fit(
        sigmoid, 
        timestamps, 
        values, 
        p0, 
        bounds=bounds, 
        full_output=True,
        maxfev=maxfev
        )
    
    y_pred = sigmoid(x_to_predict, *popt)
    if y_pred < 0: y_pred = 0
    return y_pred

def closest_timestamp_index(timestamps, target, threshold):
    """
    Finds the index of the closest timestamp in the list to the given target timestamp.
    If the difference is greater than the threshold, returns None.
    
    :param timestamps: List of timestamps (elapsed days), first timestamp starts at 0
    :param target: The target timestamp to compare
    :param threshold: The maximum allowed difference
    :return: The index of the closest timestamp or None if the difference exceeds the threshold
    """
    closest_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target))
    closest_diff = abs(timestamps[closest_index] - target)

    return closest_index if closest_diff <= threshold else pd.NA

def get_label_colors(labels):

    """
    Return a list that contains a distinct color for each distinct label.

    :param labels: 1D array of labels
    :return: List of colors
    """

    # Get the colormap
    colors = matplotlib.colormaps.get_cmap('tab20')
    # Generate colors by evenly spacing them across the colormap for each unique cluster label
    label_colors = [to_hex(colors(i / len(labels))) for i in range(len(labels))]
    #label_colors = [to_hex(colors(i/len(labels))) for i in range(len(labels))]
    return label_colors

import matplotlib
from matplotlib.colors import to_hex

def get_label_colors(labels):
    """
    Return a fixed list of colors for a given set of labels, with predefined palettes
    for 2 to 10 clusters. Falls back to tab20 if more than 10 clusters.

    :param labels: 1D array-like of labels (must be sortable)
    :return: List of hex color strings corresponding to each label
    """
    # Sorted unique labels to assign fixed colors in a consistent order
    unique_labels = sorted(set(labels))
    num_clusters = len(unique_labels)

    # Define fixed color palettes for 2 to 10 clusters
    predefined_palettes = {
        2: ['#1f77b4', '#ff7f0e'],
        3: ['#1f77b4', '#ff7f0e', '#2ca02c'],
        4: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        5: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        6: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        7: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
        8: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
        9: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'],
        10:['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }

    # Use predefined palette if number of clusters is between 2 and 10
    if 2 <= num_clusters <= 10:
        palette = predefined_palettes[num_clusters]
    else:
        # Fallback to a generic colormap (e.g., tab20)
        cmap = matplotlib.colormaps.get_cmap('tab20')
        palette = [to_hex(cmap(i % 20)) for i in range(num_clusters)]

    # Map each unique label to a color
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Return color list corresponding to input label order
    return [label_to_color[label] for label in labels]


def find_medoid(cluster_points):
    """Finds the medoid of a given cluster of points."""
    distances = cdist(cluster_points, cluster_points, metric='euclidean')  # Compute pairwise distances
    total_distances = distances.sum(axis=1)  # Sum of distances for each point
    medoid_index = np.argmin(total_distances)  # Index of the medoid
    return medoid_index  # Return the medoid

def get_matrix_pairs(df_pairs):

    """
    Get matrix of pairs of individuals for corresponding measure.
    :param df_pairs:
    """

    unique_patients = pd.unique(df_pairs[['ID1_PATIENT', 'ID2_PATIENT']].values.ravel('K'))
    df_measures_matrix = pd.DataFrame(np.nan, index=unique_patients, columns=unique_patients)
    for _, row in df_pairs.iterrows():
        df_measures_matrix.loc[row['ID1_PATIENT'], row['ID1_PATIENT']] = 0
        df_measures_matrix.loc[row['ID2_PATIENT'], row['ID2_PATIENT']] = 0
        df_measures_matrix.loc[row['ID1_PATIENT'], row['ID2_PATIENT']] = row['MEASURE']
        df_measures_matrix.loc[row['ID2_PATIENT'], row['ID1_PATIENT']] = row['MEASURE']

    return df_measures_matrix

def get_percentile(A, p):

    """
    :param A: Array of k dimensions
    :param p: Percentile
    :return: the pth percentile from A
    """

    array = A.flatten()
    array = np.sort(array)
    return np.percentile(array, p)

def get_euclidean_cost_matrix(A, B):
    """
    :param A: 2d array
    :param B: 2d array
    :return: Cost matrix
    """

    C = np.zeros([len(A), len(B)])
    for i in range(0,len(A)):
        for j in range(0,len(B)):
            C[i,j] = euclidean(A[i], B[j])
    return C

def get_HATS_cost_matrix(sv1, sv2, st1, st2, pe=1, pt=1):

    """
    :param sv1: 2d array    (sequence of vectors)
    :param sv2: 2d array    (sequence of vectors)
    :param st1: 1d array    (sequence of values that correspond to the number of days elapsed for each vector since onset)
    :param st2: 1d array    (sequence of values that correspond to the number of days elapsed for each vector since onset)
    :return: Cost matrix
    """

    C_date = np.zeros([len(sv1), len(sv2)])
    C_event = np.zeros([len(sv1), len(sv2)])
    for i in range(0,len(sv1)):
        for j in range(0,len(sv2)):
            C_date[i,j] = (st1[i] - st2[j])**2
            C_event[i,j] = euclidean(sv1[i], sv2[j])

    return pe * C_event + pt * C_date, C_event, C_date

def get_quartiles(values):
    q1 = np.percentile(values, 25)
    q2 = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    return q1, q2, q3

def interpolate_sequence(timestamps, values, interval_in_days=90, nb_intervals=5):

    """
    Align given sequence with 90 days regularity
    """

    interp_function = interp1d(timestamps, values, kind='linear', fill_value=0, bounds_error=False)
    interpolated_timestamps = np.linspace(0, (nb_intervals) * interval_in_days, nb_intervals+1)
    interpolated_values = [np.round(interpolated_value, 1) for interpolated_value in interp_function(interpolated_timestamps)]

    return interpolated_timestamps, interpolated_values

def align_sequences(values1, timestamps1, values2, timestamps2):

    """
    Align pair of sequences with 90 days regularity
    """

    interp_function1 = interp1d(timestamps1, values1, kind='linear', fill_value=0, bounds_error=False)
    interp_function2 = interp1d(timestamps2, values2, kind='linear', fill_value=0, bounds_error=False)
    num_intervals = math.ceil(max(timestamps1[-1], timestamps2[-1])/90)
    timestamps_common = np.linspace(0, (num_intervals) * 90, num_intervals+1)

    interp_values1 = [x if x >= 0 else 0 for x in interp_function1(timestamps_common)]
    interp_values2 = [x if x >= 0 else 0 for x in interp_function2(timestamps_common)]

    return timestamps_common, interp_values1, interp_values2

def get_slope(values, timestamps):
    return (values[-1]-values[0])/(timestamps[-1]-timestamps[0])

def get_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2):
    slope_p1 = get_slope(values_p1, timestamps_p1)
    slope_p2 = get_slope(values_p1, timestamps_p2)
    slope_difference = np.abs(slope_p1-slope_p2)
    return slope_difference

def get_variation(values):
    return ((values[-1]-values[0])/values[0])

def get_auc(values, timestamps):

    auc_list = []

    for i in range(0, len(values)):
        if i+1 < len(values):
            ts_diff = timestamps[i+1] - timestamps[i]
            val_diff = np.abs(values[i+1] - values[i])
            value = min(values[i], values[i+1])

            auc = (ts_diff * value) + ((ts_diff * val_diff)/2)
            auc_list.append(auc)

    return sum(auc_list)

def get_r2(s1_values, s2_values):

    is_inverse_best = False

    x = s1_values
    y = s2_values
    min_xval = int(min(x))
    max_xval = math.ceil(max(x))
    trend_y_values = np.linspace(min_xval, max_xval, num=len(x))
    r_squared1 = r2_score(trend_y_values, sorted(y))

    x = s2_values
    y = s1_values
    min_xval = int(min(x))
    max_xval = math.ceil(max(x))
    trend_y_values = np.linspace(min_xval, max_xval, num=len(x))
    r_squared2 = r2_score(trend_y_values, sorted(y))

    if r_squared2 > r_squared1:
        is_inverse_best = True

    return is_inverse_best, max(r_squared1, r_squared2)

def get_angle(s1_values, s2_values):
    x = s1_values
    y = s2_values
    coefficients = np.polyfit(x=x, y=y, deg=1)
    trend_y_values = np.polyval(coefficients, x)
    adjacent_val = np.abs(trend_y_values[-1] - trend_y_values[0])
    axiside_val = max(x) - min(x)
    hypothenus_val = np.sqrt(np.power(adjacent_val, 2) + np.power(axiside_val, 2))
    angle = math.degrees(math.asin(adjacent_val/hypothenus_val))

    return angle

def get_pairs(df):
    ids = np.array(df["ID_PATIENT"])
    pairs = list(itertools.combinations(ids, 2))
    pairs_df = pd.DataFrame(pairs, columns=['ID1_PATIENT', 'ID2_PATIENT'])
    return pairs_df

def get_highest_consecutive_slope(values, timestamps):
    highest_slope = np.finfo(np.float32).max
    for i, (value, timestamp) in enumerate(zip(values, timestamps)):
        if (i + 1) < len(values):
            consecutive_slope = (values[i+1] - values[i])/(timestamps[i+1] - timestamps[i])
            if consecutive_slope < highest_slope:
                highest_slope = consecutive_slope
    return highest_slope

def get_highest_consecutive_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2):
    highest_consecutive_slope_p1 = get_highest_consecutive_slope(values_p1, timestamps_p1)
    highest_consecutive_slope_p2 = get_highest_consecutive_slope(values_p2, timestamps_p2)
    return np.abs(highest_consecutive_slope_p1 - highest_consecutive_slope_p2)

def get_highest_consecutive_value_difference(values):
    highest_consecutive_value_difference = np.finfo(np.float32).min
    for i, value in enumerate(values):
        if (i + 1) < len(values):
            consecutive_value_difference = np.abs(values[i] - values[i+1])
            if consecutive_value_difference > highest_consecutive_value_difference:
                highest_consecutive_value_difference = consecutive_value_difference
    return highest_consecutive_value_difference

def get_highest_consecutive_timestamp_difference(timestamps):
    highest_consecutive_timestamp_difference = np.finfo(np.float32).min
    for i, value in enumerate(timestamps):
        if (i + 1) < len(timestamps):
            consecutive_timestamp_difference = np.abs(timestamps[i] - timestamps[i+1])
            if consecutive_timestamp_difference > highest_consecutive_timestamp_difference:
                highest_consecutive_timestamp_difference = consecutive_timestamp_difference
    return highest_consecutive_timestamp_difference