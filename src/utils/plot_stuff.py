from altair import Chart, Row, Color, X, Y, Row, Column
import pandas as pd

def line_xy(data, x_label="Time", y_label="Value"):
    df = pd.DataFrame(data, columns=[x_label,y_label])
    return (Chart(df)
        .encode(x=x_label, y=(y_label))
        .mark_line())


def line_x_y(x, y, x_label="Time", y_label="Value"):
    data = np.concatenate((x,y)).reshape(2,-1).T
    df = pd.DataFrame(data, columns=[x_label,y_label])
    return (Chart(df)
        .encode(x=x_label, y=(y_label))
        .mark_line())

def plot_signals(sigs, time=None):
    """
    sigs: channels x samples
    """
    df = pd.DataFrame(data=sigs.T)
    if time is not None:
        df['time'] = time
    else:
        df['time'] = df.index
    df_long = pd.melt(df,
                 id_vars=('time',),
                 var_name='signal',
                 value_name='magnitude')
    sig_chart = (Chart(df_long)
                 .encode(X('time'),
                        Y('magnitude'),
                        Row('signal'))
                 .mark_line())
    return sig_chart

def plot_sig_recon(sigs, recons, time=None):
    """
    sigs: channels x samples
    """
    long_dfs = []
    for sig_name, sig_array in zip(('sig', 'recon'), (sigs, recons)):
        df = pd.DataFrame(data=sig_array.T)
        if time is not None:
            sdf['time'] = time
        else:
            df['time'] = df.index
        df_long = pd.melt(df,
                     id_vars=('time',),
                     var_name='signal',
                     value_name='magnitude')
        df_long['src'] = sig_name
        long_dfs.append(df_long)  
    combined_dfs = pd.concat(long_dfs,ignore_index=True)
    combined_dfs.src = pd.Categorical(combined_dfs.src, categories=('sig','recon'),ordered=True)
    sig_chart = (Chart(combined_dfs)
                 .encode(X('time'),
                        Y('magnitude'),
                        Row('signal'),
                        Column('src:O', scale=Scale(domain=('sig','recon'), type='ordinal')))
                 .mark_point())
    return sig_chart