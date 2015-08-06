# coding: utf-8
import cPickle as pickle
import lasagne as nn

import sys
dump_path = sys.argv[1]
model_data = pickle.load(open(dump_path, 'r'))

print "PARAMS LOADED FROM %s" % dump_path

param_values = nn.layers.get_all_param_values(model_data['l_out'])

params_dump_path = dump_path.replace('.pkl', '_PARAMSDUMP.pkl')

with open(params_dump_path, 'w') as f:
    pickle.dump(param_values, f, pickle.HIGHEST_PROTOCOL)

print "PARAMS DUMPED TO %s" % params_dump_path
