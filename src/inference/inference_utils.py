import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def plot_bias_corr(data):
    df = data[['headline_bias','abstract_bias','text_bias','quoting_ratio']]
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


def plot_corr(data):
    df = data[['text_bias','abstract_bias','headline_bias','length','quoting_ratio']]
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

    plt.savefig('corr.png',dpi=400)


def agg_years_moths(data):
    data['published'] = pd.to_datetime(data['published'], format = '%Y-%m')
    df_years = data.groupby('Year',as_index=False).agg({'text_bias':'mean'})
    df_months = data.groupby('published',as_index=False).agg({'text_bias':'mean'})

    return df_years,df_months

def plot_in_time(data,level='years'):
    name = data['subdomain'].iloc[0]

    df_years,df_months = agg_years_moths(data)
    _, ax = plt.subplots(figsize = (10, 5))
    time_data = None
    x_axis = None
    
    if level == 'years':
        time_data = df_years
        x_axis = 'Year'

    elif level == 'months':
        time_data = df_months
        x_axis = 'published'
     
    else:
        print("Error.")
        return None
    
    sns.lineplot(ax = ax, x=x_axis, y='text_bias', data=time_data).set_title(name + '\'s media bias development over ' + level)


def strip_commentary(data):
    num_of_lost = len(data['section'] == 'komentare') + len(data['section']=='fejeton')

    data = data[data['section'] != 'komentare']
    data = data[data['section'] != 'fejeton']

    print(num_of_lost, " of articles removed.\n")

    return data

def over_time_corr(data):
    df_years,_ = agg_years_moths(data)

    plt.figure(figsize=(6, 4))
    heatmap = sns.heatmap(df_years.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


def section_bias(data):
    df_sections = data.groupby('section',as_index=False).agg({'text_bias':'mean'})

    return df_sections.sort_values(by='text_bias')