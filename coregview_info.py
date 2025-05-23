import numpy as np
import os.path as op

# CONFIG
dir_base = '/home/eze/mounts/temp2share/EBRAINS_bids/ebrains_eegseeg'
dir_data = op.join(dir_base, 'derivatives', 'epochs')
dir_analysis = ''

# DEFS
dir_resources = op.join(op.dirname(__file__), 'resources')
chans = ['e%s' % (i + 1) for i in range(256)]
egi_outside_chans = ['e%i' % ch for ch in np.sort([241, 244, 248, 252, 253, 242, 245, 249, 254, 243, 246, 250, 255, 247, 251, 256, 73, 82,
                             91, 92, 102, 93, 103, 111, 104, 112, 120, 113, 121, 133, 122, 134, 145, 135, 146, 147,
                             156, 157, 165, 166, 167, 174, 175, 176, 187, 188, 189, 199, 200, 201, 208, 209, 216, 217,
                             218, 225, 227, 228, 229, 226, 230, 234, 238, 239, 235, 231, 232, 233, 236, 237, 240])]

ch185 = [ch for ch in chans if ch not in egi_outside_chans]
ix185 = [ix for ix, ch in enumerate(chans) if ch not in egi_outside_chans ]
