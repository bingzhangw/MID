from mid import MID
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import pdb

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='')
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["dataset"] = args.dataset[:]
    config["exp_name"] = args.config.split("/")[-1].split(".")[0] + '_' + args.dataset[:]
    #pdb.set_trace()
    config = EasyDict(config)
    agent = MID(config)

    # keyattr = ["lr", "data_dir", "epochs", "dataset", "batch_size","diffnet"]
    # keys = {}
    # for k,v in config.items():
    #     if k in keyattr:
    #         keys[k] = v
    #
    # pprint(keys)

    sampling = "ddpm"
    steps = 5

    if config["eval_mode"]:
        agent.eval(sampling, 100//steps)
    else:
        agent.train()





if __name__ == '__main__':
    main()
