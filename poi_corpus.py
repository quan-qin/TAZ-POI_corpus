from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from math import sin, asin, sqrt, cos, radians
import itertools


def geo_dist(lng1, lat1, lng2, lat2) -> float:
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis1 = 2 * asin(sqrt(a)) * 6371 * 1000
    return round(dis1, 2)  # haversine distance (m)


def greedy_walk(proto_corpus: pd.DataFrame) -> List[Tuple[Any, Any, Any]]:
    """
    ref. DOI: 10.1080/13658816.2016.1244608
    :param proto_corpus
    :return: geo-corpus
    """
    # Step0.determining the order of TAZs sequence
    taz_set = {}  # {TAZid: (TAZx, TAZy), ..., TAZid: (TAZx, TAZy)] the wait-to-insert set of TAZs (x is lng, y is lat)
    proto_corpus_set = proto_corpus.drop_duplicates(subset=['TAZ_ID'], keep='first')
    taz_id_ls = proto_corpus_set['TAZ_ID'].tolist()
    taz_lon_ls = proto_corpus_set['taz_lon'].tolist()
    taz_lat_ls = proto_corpus_set['taz_lat'].tolist()
    for i in range(len(proto_corpus_set)):
        taz_set[taz_id_ls[i]] = (taz_lon_ls[i], taz_lat_ls[i])
    taz_pair_set = list(itertools.combinations(taz_set, 2))
    taz_dist_set = []
    for _pair in taz_pair_set:
        taz_dist_set.append((geo_dist(taz_set[_pair[0]][0], taz_set[_pair[0]][1],
                                      taz_set[_pair[1]][0], taz_set[_pair[1]][1]),
                             _pair[0], _pair[1]))
    end_a_id = taz_dist_set[int(np.argmax([_i[0] for _i in taz_dist_set]))][1]
    end_b_id = taz_dist_set[int(np.argmax([_i[0] for _i in taz_dist_set]))][2]
    taz_seq_order = [end_a_id, end_b_id]  # record the TAZ sequence's order
    taz_seq_dist = geo_dist(taz_set[end_a_id][0], taz_set[end_a_id][1],
                            taz_set[end_b_id][0], taz_set[end_b_id][1])
    for _taz, _coord in taz_set.items():  # wait-to-insert POIid traversing
        t_seq_dist = []  # [(_t, seq_dist), ...]
        for _t in range(len(taz_seq_order) - 1):  # insert location traversing
            t_taz_seq_order = taz_seq_order[:]
            t_taz_seq_order.insert(_t + 1, _taz)
            # dist_lr is the dist between the left & right locations of the current insert point
            dist_lr = geo_dist(taz_set[t_taz_seq_order[_t]][0], taz_set[t_taz_seq_order[_t]][1],
                               taz_set[t_taz_seq_order[_t + 2]][0], taz_set[t_taz_seq_order[_t + 2]][1])
            # dist_l/r refer to the dist between the insert point and the POI to its left/right
            dist_l = geo_dist(taz_set[t_taz_seq_order[_t]][0], taz_set[t_taz_seq_order[_t]][1],
                              taz_set[t_taz_seq_order[_t + 1]][0], taz_set[t_taz_seq_order[_t + 1]][1])
            dist_r = geo_dist(taz_set[t_taz_seq_order[_t + 1]][0], taz_set[t_taz_seq_order[_t + 1]][1],
                              taz_set[t_taz_seq_order[_t + 2]][0], taz_set[t_taz_seq_order[_t + 2]][1])
            _seq_dist = taz_seq_dist - dist_lr + dist_r + dist_l
            t_seq_dist.append((_t + 1, _seq_dist))  # record the sequence geo-dist in each insert moment _t
        min_dist_t = t_seq_dist[int(np.argmin([_i[1] for _i in t_seq_dist]))][0]
        seq_dist = t_seq_dist[int(np.argmin([_i[1] for _i in t_seq_dist]))][1]
        if _taz not in taz_seq_order:
            taz_seq_order.insert(min_dist_t, _taz)
        t_taz_seq_order = taz_seq_order[:]
        
    # Step1.determining the order of POIs sequence
    taz_poi_set = {}  # {TAZid:[POIid, ..., POIid], ...} recording the POIs sequence in each TAZ
    for _taz_id in sorted(set(proto_corpus['TAZ_ID'].tolist()), key=proto_corpus['TAZ_ID'].tolist().index):
        # traversing each taz
        poi_set = {}  # {POIid: (POIx, POIy), ...] the wait-to-insert set of POIs (x is lng, y is lat)
        proto_corpus1 = proto_corpus.groupby(['TAZ_ID']).get_group(_taz_id).reset_index(drop=True)
        poi_num = len(proto_corpus1)
        if poi_num >= 10:
            for j in range(poi_num):
                poi_set[proto_corpus1['POI_ID'][j]] = (proto_corpus1['WGS84_lon'][j], proto_corpus1['WGS84_lat'][j])
            poi_pair_set = list(itertools.combinations(poi_set, 2))  # [((POIid, POIid), (POIid, POIid)) ...]
            dist_set = []  # [(dist, A_id, B_id), (), ...] dist is the distance between a POI pair, e.g. POI_A & POI_B
            for _i in poi_pair_set:
                dist_set.append((geo_dist(poi_set[_i[0]][0], poi_set[_i[0]][1],
                                          poi_set[_i[1]][0], poi_set[_i[1]][1]),
                                 _i[0], _i[1]))
            end_a_id = dist_set[int(np.argmax([_i[0] for _i in dist_set]))][1]  # the ID of a endpoint POI
            end_b_id = dist_set[int(np.argmax([_i[0] for _i in dist_set]))][2]
            poi_seq_order = [end_a_id, end_b_id]  # initializing the POIid sequence order of i-th TAZ
            seq_dist = geo_dist(poi_set[end_a_id][0], poi_set[end_a_id][1],
                                poi_set[end_b_id][0], poi_set[end_b_id][1])  # initializing the geo-dist of POI seq
            for _poi, _coord in poi_set.items():  # wait-to-insert POIid traversing
                t_seq_dist = []  # [(_t, seq_dist), ...]
                for _t in range(len(poi_seq_order) - 1):  # insert location traversing
                    t_poi_seq_order = poi_seq_order[:]
                    t_poi_seq_order.insert(_t + 1, _poi)
                    # dist_lr is the dist between the left & right locations of the current insert point
                    dist_lr = geo_dist(poi_set[t_poi_seq_order[_t]][0], poi_set[t_poi_seq_order[_t]][1],
                                       poi_set[t_poi_seq_order[_t + 2]][0], poi_set[t_poi_seq_order[_t + 2]][1])
                    # dist_l/r refer to the dist between the insert point and the POI to its left/right
                    dist_l = geo_dist(poi_set[t_poi_seq_order[_t]][0], poi_set[t_poi_seq_order[_t]][1],
                                      poi_set[t_poi_seq_order[_t + 1]][0], poi_set[t_poi_seq_order[_t + 1]][1])
                    dist_r = geo_dist(poi_set[t_poi_seq_order[_t + 1]][0], poi_set[t_poi_seq_order[_t + 1]][1],
                                      poi_set[t_poi_seq_order[_t + 2]][0], poi_set[t_poi_seq_order[_t + 2]][1])
                    _seq_dist = seq_dist - dist_lr + dist_r + dist_l
                    t_seq_dist.append((_t + 1, _seq_dist))  # record the sequence geo-dist in each insert moment _t
                min_dist_t = t_seq_dist[int(np.argmin([_i[1] for _i in t_seq_dist]))][0]
                seq_dist = t_seq_dist[int(np.argmin([_i[1] for _i in t_seq_dist]))][1]
                if _poi not in poi_seq_order:
                    poi_seq_order.insert(min_dist_t, _poi)
                t_poi_seq_order = poi_seq_order[:]
        else:
            poi_seq_order = []
        taz_poi_set[_taz_id] = poi_seq_order
        print(f'>> TAZ {_taz_id} traversing is completed.')
        
    # Step2.formatting
    poi_corpus = []  # [(taz_id, [POI, POI, ...], TrueLabel), (taz_id, [...], TrueLabel), ...] TAZ-POI corpus
    for _taz_id in taz_seq_order:
        if _taz_id in taz_poi_set.keys() and len(taz_poi_set[_taz_id]) != 0:
            poi_corpus.append((_taz_id, taz_poi_set[_taz_id]))
    poi_id2label_dict = {}
    poi_id_ls = proto_corpus['POI_ID'].tolist()
    poi_type_ls = proto_corpus['mid_type'].tolist()
    for i in range(len(proto_corpus)):
        poi_id2label_dict[poi_id_ls[i]] = poi_type_ls[i]
    for _taz_idx, (_taz_id, _poi_seq) in enumerate(poi_corpus):
        for _poi_idx, _poi in enumerate(_poi_seq):
            poi_corpus[_taz_idx][1][_poi_idx] = poi_id2label_dict[_poi]

    taz_id2label_dict = proto_corpus_set.set_index(['TAZ_ID'])['TAZ_type_Level1'].to_dict()  # Level1 | Level2
    poi_corpus = [(taz_id, poi_ls, taz_id2label_dict[taz_id]) for taz_id, poi_ls in poi_corpus]
    
    return poi_corpus


def main():
    proto_corpus = pd.read_csv('POI_sample.csv')
    poi_corpus = greedy_walk(proto_corpus)
    poi_corpus = np.array(poi_corpus, dtype=object)
    np.save('poi_corpus.npy', poi_corpus, allow_pickle=True)
    return poi_corpus


if __name__ == '__main__':
    main()