import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_dae_columns(df):
    cols = df.columns.values
    cols[2] = 'Speaker_Sex'
    cols[4] = 'Noise_Type'
    cols[5] = 'Input_SNR'
    cols[6] = 'Input_SDR'
    cols[7] = 'Output_SDR'
    
    df.columns = cols


def load_dataframes(models):
    for model in models:
        if 'in_set' in model:
            model['in_set_df'] = pd.read_csv(model['in_set'])
            if model['name'] == 'DAE':
                format_dae_columns(model['in_set_df'])
        if 'out_of_set' in model:
            model['out_of_set_df'] = pd.read_csv(model['out_of_set'])
            if model['name'] == 'DAE':
                format_dae_columns(model['out_of_set_df'])


def error_on_the_mean(x):
    return np.std(x)/np.sqrt(len(x))


def summarize_sdr(df, groupby, bins=None):
    df = df.copy(deep=False)
    df['SDR_Improvement'] = df['Output_SDR'] - df['Input_SDR']
    agg_dict = {'SDR_Improvement': ['size', 'mean', error_on_the_mean, 'std'],
                'Input_SDR': ['mean', error_on_the_mean],
                'Output_SDR': ['mean', error_on_the_mean]}
    if 'Input_SNR_Bin' in groupby:
        if bins is None:
            raise ValueError('Must provide bins for digitization!')
        df['Input_SNR_Bin'] = pd.np.digitize(df['Input_SNR'], bins=bins)
        agg_dict['Input_SDR'] = ['mean', error_on_the_mean]
    return df.groupby(groupby).agg(agg_dict)


def make_sdr_delta_versus_noise_source_plot(models, df_base_name):
    df_name = '{}_df'.format(df_base_name)
    mean_multiindex_name = ('SDR_Improvement', 'mean')
    eotm_multiindex_name = ('SDR_Improvement', 'error_on_the_mean')
    
    all_groups = {}
    all_colors = {}
    all_names = []
    all_label_df = None
    for model in models:
        if df_name in model:
            df = model[df_name]
            all_names.append(model['name'])
            #groups = summarize_sdr(df, ['Noise_Type', 'Speaker_Sex'])
            groups = summarize_sdr(df, ['Noise_Type'])
            all_groups[model['name']] = groups[mean_multiindex_name].reset_index().merge(groups[eotm_multiindex_name].reset_index(), how='outer')
            if 'color' in model:
                all_colors[model['name']] = model['color']
            if all_label_df is None:
                all_label_df = all_groups[model['name']]
            else:
                all_label_df = all_label_df.merge(all_groups[model['name']], how='outer')
    
    labels = all_label_df['Noise_Type'].unique()
    n_groups = len(labels)
    del all_label_df
    
    # create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
    
    offset = 0
    all_rects = []
    for entry_name in all_names:
        color = 'b'
        groups = all_groups[entry_name]
        if entry_name in all_colors:
            color = all_colors[entry_name]
        
        #male_means = groups[groups['Speaker_Sex'] == 'M']
        #male_means = male_means[male_means['Noise_Type'] == labels].fillna(0)
        male_means = groups[groups['Noise_Type'] == labels].fillna(0)
        male_errors = male_means[eotm_multiindex_name].values
        male_means = male_means[mean_multiindex_name].values
        rects = plt.bar(index + offset*bar_width, male_means,
                        width=bar_width,
                        alpha=opacity,
                        color=color,
                        label=entry_name,
                        yerr=male_errors)
        
        offset += 1
    
    for i in range(len(labels)):
        labels[i] = labels[i].replace('_', ' ')
    plt.xticks(index + (len(all_names)/2 - 0.5)*bar_width, labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(75)
        tick.set_fontsize(12)
    plt.xlabel('Noise Type')
    plt.ylabel('SDR Improvement')
    plt.title('SDR Improvement Versus Noise Type', fontsize=20)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    
    ylim = [-0.5, ax.get_ylim()[1]]
    #ylim[0] = -0.5
    ax.set_ylim(ylim)
    #plt.axis([0, 11, -.5, 16])
    plt.legend(fontsize=12, edgecolor='black')
    plt.tight_layout()
    
    plt.savefig('{}_sdr_delta_versus_noise_type.pdf'.format(df_base_name), format='pdf')


def make_sdr_delta_versus_input_snr_plot(models, df_base_name, bins):
    df_name = '{}_df'.format(df_base_name)
    mean_multiindex_name = ('SDR_Improvement', 'mean')
    eotm_multiindex_name = ('SDR_Improvement', 'error_on_the_mean')
    
    all_groups = {}
    all_colors = {}
    all_names = []
    all_label_df = None
    for model in models:
        if df_name in model:
            df = model[df_name]
            all_names.append(model['name'])
            #groups = summarize_sdr(df, ['Input_SNR_Bin', 'Speaker_Sex'])
            groups = summarize_sdr(df, ['Input_SNR_Bin'], bins)
            all_groups[model['name']] = groups[mean_multiindex_name].reset_index().merge(groups[eotm_multiindex_name].reset_index(), how='outer')
            if 'color' in model:
                all_colors[model['name']] = model['color']
            if all_label_df is None:
                all_label_df = all_groups[model['name']]
            else:
                all_label_df = all_label_df.merge(all_groups[model['name']], how='outer')
    
    labels = all_label_df['Input_SNR_Bin'].unique()
    n_groups = len(labels)
    del all_label_df
    
    # create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
    
    offset = 0
    all_rects = []
    for entry_name in all_names:
        color = 'b'
        groups = all_groups[entry_name]
        if entry_name in all_colors:
            color = all_colors[entry_name]
        
        #male_means = groups[groups['Speaker_Sex'] == 'M']
        #male_means = male_means[male_means['Input_SNR_Bin'] == labels].fillna(0)
        male_means = groups[groups['Input_SNR_Bin'] == labels].fillna(0)
        male_errors = male_means[eotm_multiindex_name].values
        male_means = male_means[mean_multiindex_name].values
        rects = plt.bar(index + offset*bar_width, male_means,
                        width=bar_width,
                        alpha=opacity,
                        color=color,
                        label=entry_name,
                        yerr=male_errors)
        
        offset += 1
    
    print_labels = []
    for i in range(len(labels)):
        #print_labels.append('[{}, {})'.format(i - 5, i - 4))
        print_labels.append('{}'.format((i - 5 + i - 4)/2))
    plt.xticks(index + (len(all_names)/2 - 0.5)*bar_width, print_labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_fontsize(12)
    #plt.xlabel('Input SNR Range')
    plt.xlabel('Input SNR')
    plt.ylabel('SDR Improvement')
    plt.title('SDR Improvement Versus Input SNR', fontsize=20)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    
    ylim = [-0.5, ax.get_ylim()[1]]
    #ylim[0] = -0.5
    ax.set_ylim(ylim)
    #plt.axis([0, 11, -.5, 16])
    plt.legend(fontsize=12, edgecolor='black')
    plt.tight_layout()
    
    plt.savefig('{}_sdr_delta_versus_input_snr.pdf'.format(df_base_name), format='pdf')


def main():
    models = [
        {
            'name': 'SNMF',
            'in_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/in_sample_test_sdr_summary.csv',
            'out_of_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/out_of_sample_test_sdr_summary.csv',
            'color': '#98C1D9'
        },
        #{
        #    'name': 'DAE',
        #    'out_of_set': '/data/fs4/home/pgamble/Magnolia/Denoising/Autoencoder/Final Results/eval_test_A.csv',
        #    'color': '#E0FBFC'
        #},
        #{
        #    'name': 'Chimera MI',
        #    'in_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_in_sample_test_sdr_summary.csv',
        #    'out_of_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_out_of_sample_test_sdr_summary.csv',
        #    'color': '#3D5A80'
        #},
        {
            'name': 'DC',#'Chimera DC',
            'in_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_in_sample_test_sdr_summary.csv',
            'out_of_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_out_of_sample_test_sdr_summary.csv',
            'color': '#0C0A3E'
        },
        {
            'name': 'SCE',
            'in_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/in_sample_test_sdr_summary.csv',
            'out_of_set': '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/out_of_sample_test_sdr_summary.csv',
            'color': '#A4303F'
        },
    ]
    bins = np.linspace(-5, 5, 11)
    bins[-1] = 1.02*bins[-1]
    
    load_dataframes(models)
    
    make_sdr_delta_versus_input_snr_plot(models, 'out_of_set', bins)
    make_sdr_delta_versus_noise_source_plot(models, 'out_of_set')
    #make_sdr_delta_versus_sex_plot(models, 'out_of_set')
    

if __name__ == '__main__':
    main()