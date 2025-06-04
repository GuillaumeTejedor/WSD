import pandas as pd
import numpy as np
from rdp import rdp
from utils import *

def get_ft_activation_dict():

    """
    Fonction that return dictionary with features to compute for each pair of data points.
    :return: dictionary
    """

    ft_activation_dict = {
        "duration": True,
        "auc_diff": False,
        "mean_diff": False,
        "lv": False,
        "fv": True,
        "percentage_change_M6": True,
        "percentage_change_M12": True,
        "ALS_score_M6":False,
        "ALS_score_M12":True,
        "D50":True,
        "variation_diff": False,
        "nb_breakpoint_diff": False,
        "r2": False,
        "angle": False,
        "slope": True,
        "slope_M3": False,
        "slope_M6": False,
        "slope_M3M6": False,
        "hcs": True,
        "hcs_M3": False,
        "hcs_M6": False,
        "hcs_M3M6": False
    }

    return ft_activation_dict

def add_ft(df_patients):

    """
    Compute features for each patient.
    :param df_patients: Dataframe where each row represent a patient.
    :return: Dataframe with patients and computed features.
    """

    duration_values = []
    slope_values = []
    hcs_values = []
    first_values = []
    last_values = []
    pc_changes_M6 = []
    pc_changes_M12 = []
    ALS_scores_M12 = []
    ALS_scores_M6 = []
    D50_values = []
    slopes_M3 = []
    slopes_M3M6 = []

    three_month_limit = 91
    six_month_limit = 182
    print(df_patients["List(VALUE)"])

    ft_activation_dict = get_ft_activation_dict()
    
    median = get_median_after_x_days(
        np.array(df_patients["List(TIMESTAMP)"]), 
        np.array(df_patients["List(VALUE)"]), 
        x_to_predict=183
    )
    print("median=", median)

    for _, row in df_patients.iterrows():

        id = row["ID_PATIENT"]
        timestamps = row["List(TIMESTAMP)"]
        values = row["List(VALUE)"]
    
        if ft_activation_dict["duration"]: 
            duration_values.append(timestamps[-1])
        if ft_activation_dict["slope"]: 
            slope_values.append(get_slope(values, timestamps))
        if ft_activation_dict["fv"]: 
            first_values.append(values[0])
        if ft_activation_dict["lv"]: 
            last_values.append(values[-1])
        if ft_activation_dict["hcs"]: 
            hcs_values.append(get_highest_consecutive_slope(values, timestamps))
        if ft_activation_dict["percentage_change_M6"]:
            pc_change = get_percentage_decrease_after_x_days(timestamps, values, x_to_predict=183)
            #percentage = ((values[0] - median)/values[0]) * 100
            pc_changes_M6.append(pc_change)
        if ft_activation_dict["percentage_change_M12"]:
            pc_change = get_percentage_decrease_after_x_days(timestamps, values, x_to_predict=365)
            pc_changes_M12.append(pc_change)
        if ft_activation_dict["ALS_score_M12"]:
            ALS_score_M12 = get_alsfrsr_score_after_x_days(timestamps, values, x_to_predict=365)
            ALS_scores_M12.append(ALS_score_M12)
        if ft_activation_dict["ALS_score_M6"]:
            ALS_score_M6 = get_alsfrsr_score_after_x_days(timestamps, values, x_to_predict=183)
            ALS_scores_M6.append(ALS_score_M6)
        if ft_activation_dict["D50"]:
            d50_value, _ = get_D50(id, timestamps, values)
            D50_values.append(d50_value)        
        if ft_activation_dict["slope_M3"]:
            timestamps_M3 = [timestamp for timestamp in timestamps if timestamp <= three_month_limit]
            values_M3 = [value for timestamp, value in zip(timestamps, values) if timestamp <= three_month_limit]
            if len(timestamps_M3) >= 2: slopes_M3.append(get_slope(values_M3, timestamps_M3))
            else: slopes_M3.append(pd.NA)
        if ft_activation_dict["slope_M3M6"]:
            timestamps_M3M6 = [timestamp for timestamp in timestamps if timestamp > three_month_limit and timestamp <= six_month_limit]
            values_M3M6 = [value for timestamp, value in zip(timestamps, values) if timestamp > three_month_limit and timestamp <= six_month_limit]
            if len(timestamps_M3M6) >= 2: slopes_M3M6.append(get_slope(values_M3M6, timestamps_M3M6))
            else: slopes_M3M6.append(pd.NA)

    df_patients_features = df_patients.copy()
    if ft_activation_dict["duration"]:
        df_patients_features["DURATION"] = duration_values
    if ft_activation_dict["slope"]:
        df_patients_features["SLOPE"] = slope_values
    if ft_activation_dict["fv"]:
        df_patients_features["FV"] = first_values
    if ft_activation_dict["lv"]:
        df_patients_features["LV"] = last_values
    if ft_activation_dict["hcs"]:
        df_patients_features["HCS"] = hcs_values
    if ft_activation_dict["percentage_change_M6"]:
        df_patients_features["PC_CHANGE_M6"] = pc_changes_M6
    if ft_activation_dict["percentage_change_M12"]:
        df_patients_features["PC_CHANGE_M12"] = pc_changes_M12
    if ft_activation_dict["ALS_score_M12"]:
        df_patients_features["ALS_SCORE_M12"] = ALS_scores_M12
    if ft_activation_dict["ALS_score_M6"]:
        df_patients_features["ALS_SCORE_M6"] = ALS_scores_M6
    if ft_activation_dict["D50"]:
        df_patients_features["D50"] = D50_values
    if ft_activation_dict["slope_M3"]:
        df_patients_features["SLOPE_M3"] = slopes_M3
    if ft_activation_dict["slope_M3M6"]:
        df_patients_features["SLOPE_M3M6"] = slopes_M3M6

    return df_patients_features

def add_pairs_ft(df_patients):

    """
    Compute features for each pair of data points.
    :param df_patients: Dataframe where each row represent a pair of data points.
    :return: Dataframe with pairs and computed features.
    """

    ft_activation_dict = get_ft_activation_dict()

    duration_differences = []
    slope_differences = []
    hcs_differences = []
    r2_values = []
    auc_differences = []
    mean_differences = []
    variation_differences = []
    fv_differences = []
    lv_differences = []
    nb_breakpoint_differences = []
    angle_values = []

    slope_differences_6M = []
    hcs_differences_6M = []
    slope_differences_3M = []
    hcs_differences_3M = []
    slope_differences_3M6M = []
    hcs_differences_3M6M = []

    for row in df_patients.values:

        id1 = row[0]
        id2 = row[1]

        timestamps_p1 = row[2]
        values_p1 = row[3]
        timestamps_p2 = row[4]
        values_p2 = row[5]
        
        timestamps_common, new_values_first_patient, new_values_second_patient = align_sequences(values_p1, timestamps_p1, values_p2, timestamps_p2)

        # -- COMPUTE DURATION DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["duration"]: duration_differences.append(np.abs(timestamps_p1[-1] - timestamps_p2[-1]))

        # -- COMPUTE SLOPE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["slope"]: slope_differences.append(get_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2))

        # -- COMPUTE HIGHEST CONSECUTIVE SLOPE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["hcs"]: hcs_differences.append(get_highest_consecutive_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2))
        
        # -- COMPUTE AUC DIFFERENCE BETWEEN TWO SYNCHRONOUS SEQUENCES OF VALUES OF SAME LENGTH
        if ft_activation_dict["auc_diff"]:
            auc1 = get_auc(new_values_first_patient, timestamps_common)
            auc2 = get_auc(new_values_second_patient, timestamps_common)
            auc_differences.append(np.abs(auc1 - auc2))

        # -- COMPUTE MEAN DIFFERENCE BETWEEN TWO SEQUENCES OF VALUES
        if ft_activation_dict["mean_diff"]:
            mean_differences.append(np.abs(np.mean(values_p1) - np.mean(values_p2)))

        # -- COMPUTE FIRST VALUE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["fv"]:
            fv_differences.append(np.abs(values_p1[0] - values_p2[0]))

        # -- COMPUTE LAST VALUE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["lv"]:
            lv_differences.append(np.abs(values_p1[-1] - values_p2[-1]))

        # -- COMPUTE VARIATION DIFFERENCE BETWEEN TWO SEQUENCES OF VALUES
        if ft_activation_dict["variation_diff"]:
            var1 = get_variation(values_p1)
            var2 = get_variation(values_p2)
            variation_differences.append(np.abs(var1-var2))

        # -- COMPUTE NUMBER OF BREAKPOINTS DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["nb_breakpoint_diff"]:
            simplified1 = rdp([(v, t) for v, t in zip(values_p1, timestamps_p1)], epsilon=1.35)
            simplified2 = rdp([(v, t) for v, t in zip(values_p2, timestamps_p2)], epsilon=1.35)
            original2 = [(v, t) for v, t in zip(values_p2, timestamps_p2)]
            nb_breakpoints1 = len(simplified1)
            nb_breakpoints2 = len(simplified2)
            nb_breakpoint_differences.append(np.abs(nb_breakpoints1-nb_breakpoints2))

        # -- COMPUTE R-SQUARED BETWEEN TWO SYNCHRONOUS SEQUENCES OF VALUES OF SAME LENGTH
        if ft_activation_dict["r2"]:
            is_inverse_best, r2 = get_r2(new_values_first_patient, new_values_second_patient)
            r2_values.append(r2)

        # -- COMPUTE ANGLE OF PREDICTED TREND BETWEEN TWO SYNCHRONOUS SEQUENCES OF VALUES OF SAME LENGTH
        if ft_activation_dict["angle"]:
            if is_inverse_best == True: x, y = new_values_second_patient, new_values_first_patient
            else: x, y = new_values_first_patient, new_values_second_patient
            angle = get_angle(x, y)
            angle_values.append(angle)


        # -- COMPUTE SLOPE AND DURATION AT 6 MONTH INTERVAL
        six_month_limit = 182

        if ft_activation_dict["slope_M6"] or ft_activation_dict["hcs_M6"]:
            timestamps_6M_p1 = [timestamp for timestamp in timestamps_p1 if timestamp <= six_month_limit]
            timestamps_6M_p2 = [timestamp for timestamp in timestamps_p2 if timestamp <= six_month_limit]
            values_6M_p1 = [value for timestamp, value in zip(timestamps_p1, values_p1) if timestamp <= six_month_limit]
            values_6M_p2 = [value for timestamp, value in zip(timestamps_p2, values_p2) if timestamp <= six_month_limit]

            if len(timestamps_6M_p1) >= 2 and len(timestamps_6M_p2) >= 2:
                if ft_activation_dict["slope_M6"]: slope_differences_6M.append(get_slope_difference(values_6M_p1, timestamps_6M_p1, values_6M_p2, timestamps_6M_p2))
                if ft_activation_dict["hcs_M6"]: hcs_differences_6M.append(get_highest_consecutive_slope_difference(values_6M_p1, timestamps_6M_p1, values_6M_p2, timestamps_6M_p2))    
            else:
                slope_differences_6M.append(pd.NA)
                hcs_differences_6M.append(pd.NA)

        # -- COMPUTE SLOPE AND DURATION AT 3 MONTH INTERVAL
        three_month_limit = 91

        if ft_activation_dict["slope_M3"] or ft_activation_dict["hcs_M3"]:
            timestamps_3M_p1 = [timestamp for timestamp in timestamps_p1 if timestamp <= three_month_limit]
            timestamps_3M_p2 = [timestamp for timestamp in timestamps_p2 if timestamp <= three_month_limit]
            values_3M_p1 = [value for timestamp, value in zip(timestamps_p1, values_p1) if timestamp <= three_month_limit]
            values_3M_p2 = [value for timestamp, value in zip(timestamps_p2, values_p2) if timestamp <= three_month_limit]
            
            if len(timestamps_3M_p1) >= 2 and len(timestamps_3M_p2) >= 2:
                if ft_activation_dict["slope_M3"]: slope_differences_3M.append(get_slope_difference(values_3M_p1, timestamps_3M_p1, values_3M_p2, timestamps_3M_p2))
                if ft_activation_dict["hcs_M3"]: hcs_differences_3M.append(get_highest_consecutive_slope_difference(values_3M_p1, timestamps_3M_p1, values_3M_p2, timestamps_3M_p2))    
            else:
                slope_differences_3M.append(pd.NA)
                hcs_differences_3M.append(pd.NA)

        # -- COMPUTE SLOPE AND DURATION BETWEEN 3-6 MONTH INTERVAL
        if ft_activation_dict["slope_M3M6"] or ft_activation_dict["hcs_M3M6"]:
            timestamps_3M6M_p1 = [timestamp for timestamp in timestamps_p1 if timestamp > three_month_limit and timestamp <= six_month_limit]
            timestamps_3M6M_p2 = [timestamp for timestamp in timestamps_p2 if timestamp > three_month_limit and timestamp <= six_month_limit]
            values_3M6M_p1 = [value for timestamp, value in zip(timestamps_p1, values_p1) if timestamp > three_month_limit and timestamp <= six_month_limit]
            values_3M6M_p2 = [value for timestamp, value in zip(timestamps_p2, values_p2) if timestamp > three_month_limit and timestamp <= six_month_limit]
    
            if len(timestamps_3M_p1) >= 2 and len(timestamps_3M_p2) >= 2:
                if ft_activation_dict["slope_M3M6"]: slope_differences_3M6M.append(get_slope_difference(values_3M6M_p1, timestamps_3M6M_p1, values_3M6M_p2, timestamps_3M6M_p2))
                if ft_activation_dict["hcs_M3M6"]: hcs_differences_3M6M.append(get_highest_consecutive_slope_difference(values_3M6M_p1, timestamps_3M6M_p1, values_3M6M_p2, timestamps_3M6M_p2))    
            else:
                slope_differences_3M6M.append(pd.NA)
                hcs_differences_3M6M.append(pd.NA)

    df_pairs_with_features = df_patients.copy()

    if ft_activation_dict["duration"]:
        df_pairs_with_features["DURATION_DIFF"] = duration_differences

    if ft_activation_dict["slope"]:
        df_pairs_with_features["SLOPE_DIFF"] = slope_differences
    if ft_activation_dict["slope_M3"]:
        df_pairs_with_features["SLOPE_DIFF_3M"] = slope_differences_3M
    if ft_activation_dict["slope_M6"]:
        df_pairs_with_features["SLOPE_DIFF_6M"] = slope_differences_6M
    if ft_activation_dict["slope_M3M6"]:
        df_pairs_with_features["SLOPE_DIFF_3M6M"] = slope_differences_3M6M

    if ft_activation_dict["fv"]:
        df_pairs_with_features["FV_DIFF"] = fv_differences
    if ft_activation_dict["lv"]:
        df_pairs_with_features["LV_DIFF"] = lv_differences

    if ft_activation_dict["hcs"]:
        df_pairs_with_features["HCS_DIFF"] = hcs_differences
    if ft_activation_dict["hcs_M3"]:
        df_pairs_with_features["HCS_DIFF_3M"] = hcs_differences_3M
    if ft_activation_dict["hcs_M6"]: 
        df_pairs_with_features["HCS_DIFF_6M"] = hcs_differences_6M
    if ft_activation_dict["hcs_M3M6"]: 
        df_pairs_with_features["HCS_DIFF_3M6M"] = hcs_differences_3M6M

    if ft_activation_dict["auc_diff"]:
        df_pairs_with_features["AUC_DIFF"] = auc_differences
    if ft_activation_dict["mean_diff"]: 
        df_pairs_with_features["MEAN_DIFF"] = mean_differences
    
    if ft_activation_dict["variation_diff"]: 
        df_pairs_with_features["VARIATION_DIFF"] = variation_differences
    if ft_activation_dict["nb_breakpoint_diff"]:
        df_pairs_with_features["NB_BREAKPOINT_DIFF"] = nb_breakpoint_differences
    if ft_activation_dict["r2"]:
        df_pairs_with_features["R2"] = r2_values
    if ft_activation_dict["angle"]:
        df_pairs_with_features['ANGLE_VALUES'] = angle_values

    return df_pairs_with_features