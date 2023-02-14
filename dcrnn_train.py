from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor
import os, sys
sys.path.append(os.getcwd())
from core.Map import NetMap
from core.generate_adjmat_train_data import gen_adjmat_byedge

## DCRNN train script
def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        junction_list = \
        ['229','499','332','334',\
        'cluster_2059459190_2059459387_423609690_429990307_455858124_5692002934_8446346736','429989387','cluster_2021192413_428994253_55750386','565876699',\
                '140', 'cluster_565877085_565877100', 'cluster_1496058258_420887788']

        # junction_list = ['332']
        map_obj = NetMap('real_data/colorado_global_routing.xml',junction_list)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        adj_mx= gen_adjmat_byedge(map_obj, graph_pkl_filename)
        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='DCRNN/data/model/dcrnn_corolado.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
