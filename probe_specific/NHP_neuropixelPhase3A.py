import numpy as np
import pandas as pd
import scipy.io as io


def column_split_npx(data, axis=0):
    npx_dict = io.loadmat('/auto/data/code/KiloSort/chanMap_neuropixPhase3A.mat')
    npx_df = pd.DataFrame(npx_dict['chanMap'], columns=['chanMap'])
    npx_df['xcoords'] = npx_dict['xcoords']
    npx_df['ycoords'] = npx_dict['ycoords']
    x_vals = np.sort(npx_df.xcoords.unique())
    col1 = npx_df.index[npx_df['xcoords'] == x_vals[0]].tolist()
    col2 = npx_df.index[npx_df['xcoords'] == x_vals[1]].tolist()
    col3 = npx_df.index[npx_df['xcoords'] == x_vals[2]].tolist()
    col4 = npx_df.index[npx_df['xcoords'] == x_vals[3]].tolist()

    col1_data = np.take(data, indices=col1, axis=axis)
    col2_data = np.take(data, indices=col2, axis=axis)
    col3_data = np.take(data, indices=col3, axis=axis)
    col4_data = np.take(data, indices=col4, axis=axis)

    return col1_data, col2_data, col3_data, col4_data
