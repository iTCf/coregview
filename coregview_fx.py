def make_bip_coords(seeg_coords, columns_mean=['x', 'y', 'z']):
    import re
    import numpy as np
    import pandas as pd
    from natsort import natsort_keygen

    seeg_coords = seeg_coords.sort_values(by='name', key=natsort_keygen()) # new
    names = seeg_coords['name'].values
    new_names = []
    new_coords = []

    for ix_ch, ch in enumerate(names):
        if ch is not names[-1]:
            next_name = names[ix_ch+1]

            match = re.match(r"([a-z]+)(\')?([0-9]+)", ch, re.I)
            if match:
                items_ch = match.groups()
                if items_ch[1] == None:
                    items_ch = [items_ch[0], items_ch[2]]
                else:
                    items_ch = [items_ch[0]+items_ch[1], items_ch[2]]

            match = re.match(r"([a-z]+)(\')?([0-9]+)", next_name, re.I)
            if match:
                items_next_ch = match.groups()
                if items_next_ch[1] == None:
                    items_next_ch = [items_next_ch[0], items_next_ch[2]]
                else:
                    items_next_ch = [items_next_ch[0]+items_next_ch[1], items_next_ch[2]]

            if (items_ch[0] == items_next_ch[0]) and (int(items_ch[1])
                                                      == int(items_next_ch[1])
                                                      - 1):
                new_names.append('%s-%s' % (ch, re.findall(r'\d+', next_name)[0]))

                # cols_x_mean = seeg_coords.columns[1:-1]
                cols_x_mean = [c for c in seeg_coords.columns if any(i in c for i in columns_mean)]
                old_coord_ch1 = seeg_coords.iloc[ix_ch][cols_x_mean].values
                old_coord_ch2 = seeg_coords.iloc[ix_ch+1][cols_x_mean].values
                new_coord = np.mean([old_coord_ch1, old_coord_ch2], axis=0)
                new_coords.append(new_coord)
            del items_ch, items_next_ch

    new_coords = np.array(new_coords, dtype=float)

    bip_coords = pd.DataFrame(data=new_coords, columns=cols_x_mean)
    bip_coords['label'] = new_names

    bip_coords = bip_coords[['label']+cols_x_mean]
    bip_coords = bip_coords.round(decimals=2)
    return bip_coords


def read_run_json(fname):
    import json
    with open(fname) as f:
        run_info = json.load(f)
    return run_info


def import_raw(dir_data, subj, task, run, kind='ieeg'):
    import mne
    import pandas as pd
    import numpy as np
    import os.path as op

    factor = 1 if kind == 'eeg' else 1e-6
    fname = op.join(dir_data, subj, kind, f'{subj}_task-{task}_{run}_raw.npy')
    d = np.load(fname)*factor

    fname_chans = op.join(dir_data, subj, kind, f'{subj}_task-{task}_{run}_channels.tsv')
    chans = pd.read_csv(fname_chans, sep='\t')

    info = mne.create_info(chans['name'].tolist(), sfreq=1000, ch_types='seeg')
    raw = mne.io.RawArray(d, info)
    return raw


def load_pickle(fname):
    import pickle
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    pkl_file = open(fname, 'rb')
    variable = pickle.load(pkl_file)
    pkl_file.close()
    return variable


def find_closest_vert(surf_coords, surf):
    import numpy as np
    dist_all = np.sqrt(np.sum((surf - surf_coords) ** 2, axis=1))
    min_dist = dist_all[np.argmin(dist_all)]
    return dist_all.argmin()