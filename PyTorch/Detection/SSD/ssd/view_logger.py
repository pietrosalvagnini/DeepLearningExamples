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

    mode = "validation" #"train" #"test"
    # Train logger
    if mode == "train":
        logger_files = []
        #logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_100pos_exp2_batch100k_skipempty.json")
        #logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_1000pos_exp2_batch100k_skipempty.json")
        #logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_-1pos_negFalse_exp1_batch100k_skipempty.json")

        logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_pos-1_negratio0_noskipempty_exp1.json")
        logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_pos-1_negratio0.01_noskipempty_exp1")
        logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_pos-1_negratio0.05_noskipempty_exp1")
        logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_pos-1_negratio0.1_noskipempty_exp1")
        logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_pos1000_negratio0_noskipempty_exp1.json")
        logger_files.append("/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/train_pos100_negratio0_noskipempty_exp1")

        for logger_file in logger_files:
            exp_name = os.path.basename(logger_file)
            with open(logger_file) as ifp:
                lines = ifp.readlines()
            #        data = json.load(ifp)
            dict_list = [json.loads(x.strip("DLLL ").strip()) for x in lines]

            # get per epoch loss:
            loss_dict = {}
            val_loss = {}
            for l in dict_list:
                if not 'step' in l.keys() or len(l['step']) != 2 or not ('loss' in l['data'].keys() or 'validation accuracy' in l['data'].keys()):
                    continue
                epoch = l['step'][0]
                if 'loss' in l['data'].keys():
                    loss = l['data']['loss']
                    if not epoch in loss_dict.keys():
                        loss_dict[epoch] = []
                    loss_dict[epoch].append(loss)
                if 'validation accuracy'in l['data'].keys():
                    val_loss[epoch] = l['data']['validation accuracy']
            print("loaded")
            tr_loss = []
            epochs = []
            for k in loss_dict.keys():
                epochs.append(k)
                tr_loss.append(np.mean(loss_dict[k]))
            plt.plot(epochs, tr_loss, label = exp_name)
            print(val_loss)
        plt.legend()
        plt.show()

    # Test Logger
    elif mode == "validation":
        print("Json read")
        #logger_file  = "/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/train_1000pos_exp2_batch100k_skipempty/validation_out.log"
        #logger_file = "/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/train_100pos_exp2_batch100k_skipempty/validation_out.log"
        #logger_file = "/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/train_-1pos_negFalse_exp1/validation_out.log"

        base_path = "/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/validation"
        log_files= sorted([x for x in os.listdir(base_path) if x.endswith(".log")])
        for logger_file_i in log_files:
            logger_file= os.path.join(base_path, logger_file_i)
            with open(logger_file) as ifp:
                lines = [x.strip()  for x in ifp.readlines()]
            print("read")
            epochs = []
            maps = []
            for x in lines:
                if x.startswith("===EPOCH"):
                    epoch = int(x.split("===EPOCH:")[1].split("===")[0])
                    epochs.append(epoch)
                if x.startswith("Model precision"):
                    map = float(x.split("Model precision ")[1].split(" mAP")[0])
                    maps.append(map)
            plt.plot(epochs[:len(maps)], maps, label = f"{logger_file_i}")
            print(f"{logger_file_i}: max = {np.max(maps)}, epoch = {epochs[np.argmax(maps)]}")
        plt.legend()
        plt.show()

    else:
        base_folder = "/mnt/poranonna/ssd/storage/shared/pietro/demo/gtc_demo/data/log/tests"
        files = sorted(os.listdir(base_folder))
        for ifile in files:
            filepath = os.path.join(base_folder, ifile)
            if not "_epoch64" in ifile or not "_noskipempty.log" in ifile:
                continue
            with open(filepath) as ifp:
                lines = [x.strip() for x in ifp.readlines()]
            print(f"{ifile}: {lines[-13]}")

