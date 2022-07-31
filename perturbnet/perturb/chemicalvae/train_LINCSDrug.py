#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import anndata as ad
import torch
import pandas as pd
import numpy as np
import random

import time
from chemicalvae.chemicalVAE import *

if __name__ == "__main__":


    reload_model = False
    path_save = "model"
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok=True)

    path_data = ""
    path_lincs_onehot = ""
    path_chem_onehot = ""

    path_chem_model = ""

    ########################
    ## Data preparation
    ########################
    # with smiles
    idx_to_train = np.load(os.path.join(path_lincs_onehot,
                                        "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_processed_IndicesWithCanonicalSmiles.npy"))

    input_ltpm_label = pd.read_csv(
        os.path.join(path_data, "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_processed_PerturbMeta.csv"))
    perturb_with_onehot_overall = np.array(list(input_ltpm_label["canonical_smiles"]))

    input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]

    indicesWithoutRes = list(input_ltpm_label["canonical_smiles"] != "restricted")
    input_ltpm_label = input_ltpm_label.iloc[indicesWithoutRes, :]

    # with onehot data
    data_lincs_onehot = np.load(os.path.join(path_lincs_onehot,
                                             "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_processed_UniqueCanonicalSmilesOneHot.npy"))
    trt_list = np.load(os.path.join(path_lincs_onehot,
                                    "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_processed_UniqueCanonicalSmilesOneHotSmiles.npy"))

    list_canonSmiles = list(input_ltpm_label["canonical_smiles"])
    indicesWithOnehot = np.in1d(list_canonSmiles, trt_list)

    perturb_with_onehot = perturb_with_onehot_overall[idx_to_train][indicesWithoutRes][indicesWithOnehot]

    # removed perturbations
    removed_all_pers = np.load(os.path.join(path_lincs_onehot, "LINCS_2000RemovedPerturbs.npy"))

    kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
    perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

    # library for perturbations
    perturbToOnehot = {}
    for i in range(trt_list.shape[0]):
        perturbToOnehot[trt_list[i]] = i

    # onehot data
    data_chem_onehot = np.load(os.path.join(path_chem_onehot, "OnehotData.npy"))

    ##################
    # Model
    ##################
    # train_loader = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(data_test, batch_size=128, shuffle=True)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ChemicalVAE(n_char=data_chem_onehot.shape[2], max_len=data_chem_onehot.shape[1]).to(device)
    if reload_model:
        model.load_state_dict(torch.load(path_chem_model, map_location=device))

    model_train = ChemicalVAETrain(chemvae_model=model, device=device)
    model_train.train_np(epochs=60, data_vae_onehot=data_chem_onehot, perturb_with_onehot=perturb_with_onehot_kept,
                       model_save_per_epochs=10, path_save=path_save)
