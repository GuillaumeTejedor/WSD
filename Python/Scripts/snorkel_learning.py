from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from utils import get_quartiles
import numpy as np

TOGETHER = 0
SEPARATED = 1
ABSTAIN = -1

def label_pairs(df_train, lfs):

    """
    Train Snorkel on pairs of patients.

    :param df_train: Training dataframe of pairs that contains features to train Snorkel.
    :param lfs: List of labeling functions.
    :return: Training dataframe with predicted labels and dataframe with predicted label given by each function.
    """

    global df_global
    df_global = df_train.copy()
    df_train_no_missing = df_global.fillna(value=-1, inplace=False)
    # Apply labeling functions to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = np.array(applier.apply(df_train_no_missing))
    # Train the label model and compute the training labels
    label_model = LabelModel()
    label_model.fit(L_train, seed=42)

    df_train["LABEL"] = label_model.predict(L=L_train, tie_break_policy="abstain")
    cond_proba = label_model.get_conditional_probs()

    
    return df_train, cond_proba

@labeling_function()
def duration_diff(x):

    """
    Compute global duration difference between two individuals.
    :param x: duration difference variable.
    :return: Predicted label.
    """

    feature = x.DURATION_DIFF
    q1, _, q3 = get_quartiles(df_global.DURATION_DIFF)

    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED
        
@labeling_function()
def slope_diff(x):

    """
    Compute global slope difference between two individuals.
    :param x: Slope difference variable.
    :return: Predicted label.
    """

    feature = x.SLOPE_DIFF
    q1, _, q3 = get_quartiles(df_global.SLOPE_DIFF)
    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED
    
@labeling_function()
def hcs_diff(x):

    """
    Compute Highest consecutive slope difference between two individuals.
    :param x: Highest consecutive slope difference variable.
    :return: Predicted label.
    """

    feature = x.HCS_DIFF
    q1, _, q3 = get_quartiles(df_global.HCS_DIFF)
    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED

@labeling_function()
def fv_diff(x):

    """
    Compute first value difference between two individuals.
    :param x: First value difference variable.
    :return: Predicted label.
    """

    feature = x.FV_DIFF
    
    q1, _, q3 = get_quartiles(df_global.FV_DIFF)
    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED
    
@labeling_function()
def pc_change_M6_diff(x):

    """
    Compute percentage change after 6 month difference between two individuals.
    :param x: First value difference variable.
    :return: Predicted label.
    """
    feature = x.PC_CHANGE_M6_DIFF
    q1, _, q3 = get_quartiles(df_global.PC_CHANGE_M6_DIFF)
    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED

@labeling_function()
def pc_change_M12_diff(x):

    """
    Compute percentage change after 12 months difference between two individuals.
    :param x: First value difference variable.
    :return: Predicted label.
    """
    feature = x.PC_CHANGE_M12_DIFF
    q1, _, q3 = get_quartiles(df_global.PC_CHANGE_M12_DIFF)
    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED
    
@labeling_function()
def als_score_M12_diff(x):

    """
    Compute ALSFRS-R score after 1 year difference between two individuals.
    :param x: First value difference variable.
    :return: Predicted label.
    """
    feature = x.ALS_SCORE_M12_DIFF
    q1, _, q3 = get_quartiles(df_global.ALS_SCORE_M12_DIFF)
    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED
    
@labeling_function()
def D50_diff(x):
    feature = x.D50_DIFF

    if feature == -1:
        return ABSTAIN

    q1, _, q3 = get_quartiles(df_global.D50_DIFF.dropna())
    if feature <= q1:
        return TOGETHER
    if feature > q1 and feature < q3:
        return ABSTAIN
    if feature >= q3:
        return SEPARATED
    
@labeling_function()
def abstain(x):
    """
    Return abstain value.
    :param x: First value difference variable.
    :return: Predicted label.
    """
    return ABSTAIN