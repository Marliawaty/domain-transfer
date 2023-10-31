import copy
import itertools
import math
import pickle
from pathlib import Path

import pandas as pd
import scipy.interpolate
from matplotlib import pyplot as plt
import pylab as pl
import seaborn as sns
import numpy as np
from matplotlib import image
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, friedmanchisquare
import scikit_posthocs as sp


import sys
import logging

##################################

# Fixed variables
MAX_NUM_NEURONS_IMAGES_INDEX = -1
SPLIT_NUM = 100
PRIMATE_IT_MAX_NUM_IMAGES = 100
MAX_NUM_NEURONS_PRIMATE_IT = 71
PRIMATE_IT_NEURONS_LIST = [10, 20, 30, 40, 50, 71, 100]

TICKS_SIZE = 50
TICKS_WIDTH = 2
TICKS_LENGTH = 6
FONT_AXIS_LABEL_SIZE = 40
FIG_SIZE = (30, 15)
FIG_SIZE_SQUARE = (30, 30)
MARKER_SIZE = 30

CROSSDOMAINS = ['hvm', 'convex_hull', 'outline', 'skeleton', 'silhouette', 'cartoon', 'line_drawing', 'mosaic', 'painting', 'sketch', 'cococolor', 'cocogray', 'tdw']
CROSSDOMAINS_NO_HVM = ['convex_hull', 'outline', 'skeleton', 'silhouette', 'cartoon', 'line_drawing', 'mosaic', 'painting', 'sketch', 'cococolor', 'cocogray', 'tdw']
OODs = ['convex_hull', 'outline', 'silhouette', 'cartoon', 'line_drawing', 'mosaic', 'painting', 'sketch']

HUMAN_PERFORMANCE = [0.86831, 0.6038, 0.6958, 0.54305, 0.788, 0.92693, 0.94775, 0.76399, 0.9402, 0.9505, 1, 1, 1]   #old
HUMAN_PERFORMANCES_OODs = [0.587, 0.706, 0.788, 0.9342, 0.9503, 0.7626, 0.9434, 0.9502]

BRAIN_MODELS = ['alexnet', 'custom_model_cv_18_dagger_408', 'CORnet-S', 'voneresnet-50-non_stochastic', 'resnet50-barlow', ]
MAJAJ_HONG_NEURONS = 600
MAJAJ_HONG_IMAGES = 4608

BRAIN_MODEL_COLOR = 'darkblue'
PRIMATE_IT_COLOR = 'darkgreen'
BRAIN_MODEL_COLOR_ARRAY = ['blue', 'darkblue', 'royalblue', 'steelblue', 'deepskyblue']


##############################################################

def get_averaged_performance_primate_it():
    for crossdomain in CROSSDOMAINS:
        crossdomain_df = pd.read_csv(f'Crossdomain_performance_hvm_{crossdomain}_averaged_performance.csv')
        if crossdomain == 'hvm':
            concat_df = crossdomain_df.iloc[[MAX_NUM_NEURONS_IMAGES_INDEX]]
            concat_df['Crossdomain'] = crossdomain
        else:
            crossdomain_last_row = crossdomain_df.iloc[[MAX_NUM_NEURONS_IMAGES_INDEX]]
            crossdomain_last_row['Crossdomain'] = crossdomain
            concat_df = pd.concat([concat_df, crossdomain_last_row], ignore_index=True)
#    save_dataframe(dataframe=concat_df, csv_dataframe_name='Crossdomain_performance_hvm_all_domains_averaged.csv')
    return concat_df


def plot_figure_2():
    performance_df = get_averaged_performance_primate_it()
    hvm_performance = performance_df[performance_df['Crossdomain'] == 'hvm']['Accuracy test data'].values
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    performance_df.plot.bar(x='Crossdomain', y='Accuracy test data',
                                       yerr=performance_df['Std test data'].T.values,
                                       rot=45, ax=ax, legend=False, color=PRIMATE_IT_COLOR)
    plt.tick_params(axis='both', which='major', labelsize=TICKS_SIZE)
    plt.ylabel('Accuracy [2AFC]', fontsize=FONT_AXIS_LABEL_SIZE)
    plt.axhline(y=hvm_performance, linestyle='dashed', color='gray')
    plt.axhline(y=.5, linestyle='dashed', color='black')
    plt.ylim(0.45, 0.9)
    save_fig('figure_2_final.svg')
    plt.show()


    def plot_figure_3b():
    primate_it_df = pd.read_csv('Crossdomain_performance_hvm_all_domains_averaged.csv')
    all_brain_model_df = open_dataframe('Deep_nets_performance_hvm_all_domains_all_brain_models_averaged.csv')

    for brain_model in BRAIN_MODELS:   
        brain_model_df = all_brain_model_df[all_brain_model_df['Brain model name'] == brain_model]
        x_values = primate_it_df['Accuracy test data'].values
        y_values = brain_model_df['Accuracy test data'].values
        x_err_interval = [primate_it_df['Std test data'], primate_it_df['Std test data']]
        y_err_interval = [brain_model_df['Std test data'], brain_model_df['Std test data']]
        fig, ax = pl.subplots(figsize=FIG_SIZE_SQUARE)
        lines = {'linestyle': 'None'}
        plt.rc('lines', **lines)
        pl.plot(x_values, y_values, 'o', markersize=MARKER_SIZE)
        pl.axline((0.5, 0.5), (0.85, 0.85), linestyle='--', color='black')
        el = pl.errorbar(x_values, y_values, xerr=x_err_interval, yerr=y_err_interval, fmt='b',
                         ecolor=[PRIMATE_IT_COLOR, BRAIN_MODEL_COLOR])
        # Add crossdomain labels
        for crossdomain, x, y in zip(CROSSDOMAINS, x_values, y_values):
            label = crossdomain

            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='right')  # horizontal alignment can be left, right or center
        elines = el.get_children()
        elines[1].set_color(PRIMATE_IT_COLOR)
        elines[2].set_color(BRAIN_MODEL_COLOR)
        plt.xlabel('Primate IT accuracy', fontsize=FONT_AXIS_LABEL_SIZE)
        plt.ylabel('ANN accuracy', fontsize=FONT_AXIS_LABEL_SIZE)
        plt.ylim(0.5, 0.85)
        plt.xlim(0.5, 0.85)
        plt.tick_params(axis='both', which='major', labelsize=TICKS_SIZE, width=TICKS_WIDTH, length=TICKS_LENGTH)
        plt.title(f'Performance for primate IT and {brain_model}', fontsize=FONT_AXIS_LABEL_SIZE)
        save_fig(f'figure_3b_{brain_model}_final.svg')
        pl.show()

def prep_data_for_3c():
    brain_model_avg = pd.read_csv('Deep_nets_performance_hvm_all_domains_all_brain_models_averaged.csv')   
    brain_model_avg_no_hvm = brain_model_avg[brain_model_avg['Crossdomain'] != 'hvm']
    brain_model_avg_pro_model = brain_model_avg_no_hvm.groupby('Brain model name').mean('Accuracy test data')
    brain_model_avg_pro_model.reset_index(inplace=True)
    brain_model_avg_pro_model = brain_model_avg_pro_model.rename(columns={'index': 'Brain model name'})
    brain_model_mean = brain_model_avg_pro_model['Accuracy test data'].mean()
    brain_model_std = brain_model_avg_pro_model['Accuracy test data'].std()
    primate_it = pd.read_csv('Crossdomain_performance_hvm_all_domains_averaged.csv')    
    primate_it_no_hvm = primate_it[primate_it['Crossdomain'] != 'hvm']
    primate_it_avg = primate_it_no_hvm['Accuracy test data'].mean()
    primate_it_std = primate_it_no_hvm['Accuracy test data'].std()
    figure_3c_df = brain_model_avg_pro_model.append({
        'Brain model name': 'Primate IT',
        'Accuracy test data': primate_it_avg.item(),
        'Std test data': primate_it_std.item()
    }, ignore_index=True)
    figure_3c_df = figure_3c_df.append({
        'Brain model name': 'all brain models',
        'Accuracy test data': brain_model_mean.item(),
        'Std test data': brain_model_std.item()
    }, ignore_index=True)

    return figure_3c_df[['Brain model name', 'Accuracy test data', 'Std test data']] #TODO: what kind of df is this? Add to description


def plot_figure_3c():
    data = prep_data_for_3c()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    data.plot.bar(x='Brain model name', y='Accuracy test data',
                                       yerr=data['Std test data'].T.values,
                                       rot=90, ax=ax, legend=False, color=BRAIN_MODEL_COLOR)
    plt.tick_params(axis='both', which='major', labelsize=TICKS_SIZE, width=TICKS_WIDTH, length=TICKS_LENGTH)
    plt.ylabel('Out of domain accuracy', fontsize=50)
    primate_it_avg_accuracy = data[data['Brain model name'] == 'Primate IT']['Accuracy test data'].values
    plt.axhline(y=primate_it_avg_accuracy, linestyle='dashed', color=PRIMATE_IT_COLOR)
    plt.ylim(0.45, 0.7)
    save_fig('figure_3c_poster_new_models.svg')
    plt.show()


    
def get_dataframe_for_figure_3d():
    # Create dataframe with distance for each split for each brain model and domain
    distance_df = pd.DataFrame(columns=['Brain model name', 'Crossdomain', 'Mean distance primate IT', 'Std distance primate IT'])
    for brain_model in BRAIN_MODELS:
        for crossdomain in CROSSDOMAINS_NO_HVM:
            brain_model_df = pd.read_csv(f'Deep_nets_performance_hvm_{crossdomain}_{brain_model}_scaling_factor_penultimate_layer_all_splits.csv')
            distance_list_brain_model_crossdomain = []
            for split in np.arange(SPLIT_NUM):
                primate_it_df = open_dataframe(f'Crossdomain_performance_hvm_{crossdomain}_split_{split}.csv')
                primate_it_accuracy = primate_it_df.iloc[[MAX_NUM_NEURONS_IMAGES_INDEX]]['Accuracy test data'].values
                brain_model_accuracy = brain_model_df.iloc[[split]]['Accuracy test data'].values
                delta_split = brain_model_accuracy - primate_it_accuracy
                distance_list_brain_model_crossdomain.append(delta_split)
            mean_distance_crossdomain = np.mean(distance_list_brain_model_crossdomain)
            std_distance_crossdomain = np.std(distance_list_brain_model_crossdomain)

            distance_df = distance_df.append({
                'Brain model name': brain_model,
                'Crossdomain': crossdomain,
                'Mean distance primate IT': mean_distance_crossdomain,
                'Std distance primate IT': std_distance_crossdomain
            }, ignore_index=True)
    save_dataframe(dataframe=distance_df, csv_dataframe_name='Deep_nets_performance_hvm_all_domains_all_brain_models_distance_avg_over_100_splits.csv') #TODO: add to folder


def plot_figure_3d():
    dataframe = open_dataframe(csv_filename='Deep_nets_performance_hvm_all_domains_all_brain_models_distance_avg_over_100_splits.csv')
    mean_distance_df = dataframe.groupby('Crossdomain', as_index=False).mean()
    colors_cmap = plt.cm.get_cmap('Blues')
    colors = [colors_cmap(.4), colors_cmap(.55), colors_cmap(.7), colors_cmap(.85), colors_cmap(.9)]
    # Plot each brain models' mean accuracy and the avgerage over brain models
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(x='Crossdomain', y='Mean distance primate IT',data=dataframe, hue='Brain model name', ax=ax1, legend=True, s=500)
    for crossdomain_index in np.arange(len(mean_distance_df)):
        crossdomains = mean_distance_df['Crossdomain'].values
        mean_values = mean_distance_df['Mean distance primate IT'].values
        plt.scatter(x=crossdomains[crossdomain_index], y=mean_values[crossdomain_index], color='black', marker='_', s=2500, linewidths=10)
    plt.ylabel('Accuracy delta ANN - primate IT', fontsize=FONT_AXIS_LABEL_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICKS_SIZE, width=TICKS_WIDTH, length=TICKS_LENGTH)
    save_fig('figure_3d.svg')
    plt.show()


    
