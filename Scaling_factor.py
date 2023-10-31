import itertools
import math
import pickle
import time
from pathlib import Path
import sys
import logging
import scipy

import copy

import fire
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from result_caching import store
from scipy.optimize import curve_fit
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from brainio_base.assemblies import NeuronRecordingAssembly
from brainio.fetch import StimulusSetLoader
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.benchmarks._neural_common import average_repetition
from brainio_base.stimuli import StimulusSet

_logger = logging.getLogger(__name__)
################################################


####################
# Find scaling factor
###################

def save_scaling_factor_neurons(num_primate_it_neurons):
    primate_hvm_avg_performance = get_primate_it_num_img_num_neurons_performance(number_images=PRIMATE_IT_MAX_NUM_IMAGES, number_neurons=num_primate_it_neurons)
    scaling_factor_dict = {}
    for brain_model in BRAIN_MODELS:
        brain_model_image_locked_df = get_brain_model_performance_for_num_images(brain_model=brain_model, number_images=PRIMATE_IT_MAX_NUM_IMAGES)
        matching_neuron_number_in_brain_model = find_matching_number_neurons_in_brain_model(brain_model_df=brain_model_image_locked_df, primate_it_performance=primate_hvm_avg_performance)
        scaling_factor_dict[brain_model] = matching_neuron_number_in_brain_model
    save_dictionary(dictionary=scaling_factor_dict, pkl_name=f'Deep_nets_crossdomain_performance_scaling_factors_penultimate_layer_{num_primate_it_neurons}_neuron_match.pkl')


def get_primate_it_num_img_num_neurons_performance(number_images, number_neurons):
    primate_df = open_dataframe('Crossdomain_performance_hvm_hvm_averaged_performance.csv')
    primate_df = primate_df[(primate_df['#Neurons'] == number_neurons) & (primate_df['#Images training'] == number_images)]
    primate_it_performance = primate_df['Accuracy test data'].item()
    return primate_it_performance


def get_brain_model_performance_for_num_images(brain_model, number_images):
    # Load brain model performance and lock the number of images
    brain_model_df = pd.read_csv(f'Deep_nets_performance_hvm_hvm_{brain_model}_averaged_performance_penultimate_layer.csv')
    brain_model_image_locked_df = brain_model_df[brain_model_df['#Images training'] == number_images]
    return brain_model_image_locked_df

def find_matching_number_neurons_in_brain_model_via_interpolation(brain_model_upper_bound, brain_model_lower_bound, primate_it_performance):
    interpolation_neuron_value_1 = brain_model_upper_bound['#Neurons'].values[0]
    interpolation_neuron_value_2 = brain_model_lower_bound['#Neurons'].values[-1]
    interpolation_performance_value_1 = brain_model_upper_bound['Accuracy test data'].values[0]
    interpolation_performance_value_2 = brain_model_lower_bound['Accuracy test data'].values[-1]
    interpolation_function = scipy.interpolate.interp1d(
        x=[interpolation_performance_value_1, interpolation_performance_value_2],
        y=[interpolation_neuron_value_1, interpolation_neuron_value_2], kind='linear')
    interpolated_neuron = math.ceil(interpolation_function(x=primate_it_performance))
    return interpolated_neuron


def find_matching_number_neurons_in_brain_model(brain_model_df, primate_it_performance):
    # Find the number of neurons that have bigger and smaller performance than hvm to interpolate
    brain_model_performance_bigger_than_it = brain_model_df[brain_model_df['Accuracy test data'] >= primate_it_performance]
    brain_model_performance_smaller_than_it = brain_model_df[brain_model_df['Accuracy test data'] <= primate_it_performance]
    matching_number_neurons_in_brain_model = find_matching_number_neurons_in_brain_model_via_interpolation(brain_model_upper_bound=brain_model_performance_bigger_than_it,
                                                                                                           brain_model_lower_bound=brain_model_performance_smaller_than_it,
                                                                                                           primate_it_performance=primate_it_performance)
    return matching_number_neurons_in_brain_model



