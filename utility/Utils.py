import scipy.sparse as sp
import numpy as np

def get_setting(datasetname, key, Transmatrix, device):
    if(datasetname == "Movielens"):
        if(key == "user"):
            setting_umu = {'T': 16, 'device': device, 'TransM': Transmatrix[0]}
            return [settings_umu]
        elif(key == "movie"):
            settings_mum = {'T': 16, 'device': device, 'TransM': Transmatrix[0]}
            settings_mgm = {'T': 4, 'device': device, 'TransM': Transmatrix[1]}
            return [settings_mum,settings_mgm]
    elif(datasetname == "Amazon"):
        if(key == "user"):
            settings_uiu = {'T': 32, 'device': device, 'TransM': Transmatrix[0]}
            return [settings_uiu]
        elif(key == "item"):
            settings_iui = {'T': 32, 'device': device, 'TransM': Transmatrix[0]}
            settings_ici = {'T': 16, 'device': device, 'TransM': Transmatrix[1]}
            settings_ibi = {'T': 16, 'device': device, 'TransM': Transmatrix[2]}
            return [settings_iui,settings_ici,settings_ibi]
    elif (datasetname == "Yelp"):
        if (key == "user"):
            settings_ubu = {'T': 8, 'device': device, 'TransM': Transmatrix[0]}
            settings_ucu = {'T': 2, 'device': device, 'TransM': Transmatrix[1]}
            return [settings_ubu, settings_ucu]
            # return [settings_ubu]
        elif (key == "business"):
            settings_bub = {'T': 8, 'device': device, 'TransM': Transmatrix[0]}
            settings_bcb = {'T': 8, 'device': device, 'TransM': Transmatrix[1]}
            settings_bib = {'T': 8, 'device': device, 'TransM': Transmatrix[2]}
            return [settings_bub, settings_bcb,settings_bib]
    elif (datasetname == "Dbbook"):
        if (key == "user"):
            settings_ubu = {'T': 28, 'device': device, 'TransM': Transmatrix[0]}
            return [settings_ubu]
        elif (key == "book"):
            settings_bub = {'T': 28, 'device': device, 'TransM': Transmatrix[0]}
            settings_bab = {'T': 12, 'device': device, 'TransM': Transmatrix[1]}
            return [settings_bub, settings_bab]
    elif (datasetname == "LastFM"):
        if (key == "user"):
            settings_uau = {'T': 24, 'device': device, 'TransM': Transmatrix[0]}

            return [settings_uau]
        elif (key == "artist"):
            settings_aua = {'T': 32, 'device': device, 'TransM': Transmatrix[0]}
            settings_ata = {'T': 8, 'device': device, 'TransM': Transmatrix[1]}
            return [settings_aua, settings_ata]
    else:
        print("Available datasets: Movielens, amazon, Yelp.")
        raise NotImplementedError



def get_hete_adjs(g, meta_paths, device):
    hete_adjs = {}
    for i in range(len(meta_paths)):
        for value in meta_paths[i]:
            adj_matrixb = g.adj_external(transpose=False, ctx=device, scipy_fmt="coo", etype=value)

            adj_matrixb = adj_matrixb.tocsc()
            hete_adjs.update({value:adj_matrixb })
    return hete_adjs

def get_transition(given_hete_adjs, metapath_info):
    # transition
    hete_adj_dict_tmp = {}
    for key in given_hete_adjs.keys():
        deg = given_hete_adjs[key].sum(1)
        hete_adj_dict_tmp[key] = given_hete_adjs[key] / (np.where(deg > 0, deg, 1))
    homo_adj_list = []
    for i in range(len(metapath_info)):
        adj = hete_adj_dict_tmp[metapath_info[i][0]]#
        for etype in metapath_info[i][1:]:
            adj = adj.dot(hete_adj_dict_tmp[etype])
        homo_adj_list.append(sp.csc_matrix(adj))
    return homo_adj_list
