# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os.path

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Train logger
    logger_files = []
    logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_100pos_exp2_batch100k_skipempty.json")
    logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_1000pos_exp2_batch100k_skipempty.json")
    logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_-1pos_negFalse_exp1_batch100k_skipempty.json")

    for logger_file in logger_files:
        exp_name = os.path.basename(logger_file)
        with open(logger_file) as ifp:
            lines = ifp.readlines()
        #        data = json.load(ifp)
        dict_list = [json.loads(x.strip("DLLL ").strip()) for x in lines]

        # get per epoch loss:
        loss_dict = {}
        for l in dict_list:
            if not 'step' in l.keys() or len(l['step']) != 2 or not 'loss' in l['data'].keys():
                continue
            epoch = l['step'][0]
            loss = l['data']['loss']
            if not epoch in loss_dict.keys():
                loss_dict[epoch] = []
            loss_dict[epoch].append(loss)
        print("loaded")
        tr_loss = []
        epochs = []
        for k in loss_dict.keys():
            epochs.append(k)
            tr_loss.append(np.mean(loss_dict[k]))
        plt.plot(epochs, tr_loss, label = exp_name)
    plt.legend()
    plt.show()

    # Test Logger

    print("Json read")