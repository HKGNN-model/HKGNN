import copy
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as scio
import os
import json
import time
from datetime import datetime
import geohash
from itertools import combinations

from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
import torch

# import utils.util as util

def load_city_data(city):
    path = "./data/raw/follow_mat/" + city + "/"
    check_in = pd.read_csv(path + "checkins_" + city + "_follow_mat.csv")
    poi_data = pd.read_csv(path + "POI_" + city + "_follow_mat.csv")

    print(city + " check_in number: ", len(check_in))
    users = set(list(check_in['userid']))
    print(city + " user number: ", len(users))
    venues = set(list(check_in['Venue_id']))
    print(city + " venues number: ", len(venues))

    friend_old = np.load(path + "friend_old.npy")
    friend_new = np.load(path + "friend_new.npy")
    friend = np.vstack([friend_old, friend_new])
    
    return check_in, poi_data, users, venues, friend

def process_city_data_recode(city, check_in, poi_data, users, venues, friend):
    '''
    @param city: city name
    @param check_in: check_in data
    @param poi_data: poi data
    @param users: users set
    @param venues: venues set
    @param friend: friendship data

    @return: check_in: recoded check_in data with new user index and venue index, time index and geohash index
    @return friend_list_index: recoded friendship data with new user index
    @return venues_dic: dictionary of venues and their index
    @return time_hour_dic: dictionary of hours(48*7) and their index(start and end)
    @return time_month_dic: dictionary of months(12) and their index(start and end)
    @return time_year_dic: dictionary of years(3) and their index(start and end)
    @return geo_relations: dictionary of geohash relations, containing all (4,5,6) geohash relation
    @return current_index: current index of all vertices
    @return user_index: dictionary of the start and end index of the users
    @return poi_index: dictionary of the start and end index of the venues
    '''

    # process friendship
    friend = pd.DataFrame(friend)
    friend.columns = ['user1', 'user2'] # rename columns to user1 and user2
    friend.drop_duplicates(subset=['user1', 'user2'], keep='first', inplace=True) # drop duplicate rows
    friend = friend.reset_index(drop=True) # reset index after drop 
 
    friend_list = []
    for index, row in friend.iterrows():
        if row['user1'] != row['user2'] and row['user1'] in users and row['user2'] in users:
            friend_list.append([row['user1'], row['user2']]) # if the friendship is valid, add it to the list
    print(city + " friendship number: ", len(friend_list))

    # Sort the users in the list
    users_sort = list(sorted(list(users)))
    # Create a dictionary of the users and their index
    users_dic = {}
    for i in range(len(users_sort)):
        users_dic[users_sort[i]] = i

    # Create a dictionary of the start and end index of the users
    user_index = {}
    user_index['start'], user_index['end'] = 0, len(users_sort) - 1

    # recode venues
    venues_sort = list(sorted(list(venues)))
    venues_dic = {}
    for i in range(len(venues_sort)):
        venues_dic[venues_sort[i]] = i + len(users_sort)

    poi_index = {}
    poi_index['start'], poi_index['end'] = len(users_sort), len(users_sort) + len(venues_sort) - 1

    current_index = len(users_sort) + len(venues_sort) - 1

    # recode friendship
    friend_list_index = []
    for i in range(len(friend_list)):
        friend_list_index.append((users_dic[friend_list[i][0]], users_dic[friend_list[i][1]]))

    poi_data['venues_index'] = None
    poi_data_venue_index = []
    for index, row in poi_data.iterrows():
        poi_data_venue_index.append(venues_dic[row['Venue_id']])
    poi_data['venues_index'] = poi_data_venue_index

    # recode check_in 
    poi_data = poi_data.sort_values('Venue_id', ascending=True, inplace=False)
    poi_data = poi_data.reset_index(drop=True)
    poi_lat_lon = {}  # {poi_index:(lat,lon)}
    for index, row in poi_data.iterrows():
        poi_lat_lon[row['venues_index']] = (row['latitude'], row['longitude'])

    # process check_in location into 
    poi_location = {}
    all_geo_4, all_geo_5, all_geo_6 = {}, {}, {}

    geo_relations = {}
    
    count = 0 # count the number of hyperedges
    for poi in poi_lat_lon.keys():
        location = poi_lat_lon[poi]
        geohash_4, geohash_5, geohash_6 = transform_location(location)
        # encode geohash vertices
        if geohash_4 not in all_geo_4:
            all_geo_4[geohash_4] = len(all_geo_4) + 1
        if geohash_5 not in all_geo_5:
            all_geo_5[geohash_5] = len(all_geo_5) + 1
        if geohash_6 not in all_geo_6:
            all_geo_6[geohash_6] = len(all_geo_6) + 1

        poi_location[poi] = {}
        poi_location[poi]['geohash_4'] = geohash_4
        poi_location[poi]['geohash_5'] = geohash_5
        poi_location[poi]['geohash_6'] = geohash_6

        # combine three geohash vertices into one hyperedge
        geo_relations[count] = [geohash_4, geohash_5, geohash_6]
        count += 1

    for geo_4 in all_geo_4.keys():
        all_geo_4[geo_4] = all_geo_4[geo_4] + current_index
    current_index += len(all_geo_4)

    for geo_5 in all_geo_5.keys():
        all_geo_5[geo_5] = all_geo_5[geo_5] + current_index
    current_index += len(all_geo_5)

    for geo_6 in all_geo_6.keys():
        all_geo_6[geo_6] = all_geo_6[geo_6] + current_index
    current_index += len(all_geo_6)

    for poi in poi_location.keys():
        poi_location[poi]['geohash_4'] = all_geo_4[poi_location[poi]['geohash_4']]
        poi_location[poi]['geohash_5'] = all_geo_5[poi_location[poi]['geohash_5']]
        poi_location[poi]['geohash_6'] = all_geo_6[poi_location[poi]['geohash_6']]

    for geo in geo_relations.keys():
        geo_relations[geo][0] = all_geo_4[geo_relations[geo][0]]
        geo_relations[geo][1] = all_geo_5[geo_relations[geo][1]]
        geo_relations[geo][2] = all_geo_6[geo_relations[geo][2]]


    check_in = check_in.sort_values('Venue_id', ascending=True, inplace=False)
    check_in = check_in.reset_index(drop=True)
    check_in['users_index'] = None
    check_in['venues_index'] = None

    check_in_users_index = []
    check_in_venues_index = []

    check_in_geohash_4 = []
    check_in_geohash_5 = []
    check_in_geohash_6 = []

    for _, row in check_in.iterrows():
        check_in_users_index.append(users_dic[row['userid']])
        check_in_venues_index.append(venues_dic[row['Venue_id']])
        # check_in_lat_lon.append(poi_lat_lon[venues_dic[row['Venue_id']]])
        check_in_geohash_4.append(poi_location[venues_dic[row['Venue_id']]]['geohash_4'])
        check_in_geohash_5.append(poi_location[venues_dic[row['Venue_id']]]['geohash_5'])
        check_in_geohash_6.append(poi_location[venues_dic[row['Venue_id']]]['geohash_6'])


    check_in['users_index'] = check_in_users_index
    check_in['venues_index'] = check_in_venues_index
    # check_in['lat_lon'] = check_in_lat_lon
    check_in['geohash_4'] = check_in_geohash_4
    check_in['geohash_5'] = check_in_geohash_5
    check_in['geohash_6'] = check_in_geohash_6

    check_in['local_year'] = None
    check_in['local_month'] = None
    check_in['hour_period'] = None
    check_in['local_time'] = None
    check_in_local_time = []
    check_in_local_year = []
    check_in_local_month = []
    check_in_hour_period = []

    for _, row in check_in.iterrows():
        time_1 = row['utc_time'][:-10] + row['utc_time'][-4:]
        timezone_offset = row['Timezone_offset']
        struct_time = time.mktime(time.strptime(time_1, "%a %b %d  %H:%M:%S %Y")) + timezone_offset * 60
        localtime = time.localtime(struct_time)  # 返回元组
        check_in_local_time.append(localtime)
        check_in_local_year.append(localtime[0])
        check_in_local_month.append(localtime[1])
        check_in_hour_period.append(time_partition(localtime[6], localtime[3], localtime[4]))


    time_hour_dic = {}
    tot_hour_period = max(check_in_hour_period)

    for i in range(len(check_in_hour_period)):
        if check_in_hour_period[i] not in time_hour_dic:
            time_hour_dic[check_in_hour_period[i]] = check_in_hour_period[i] + current_index
        check_in_hour_period[i] = time_hour_dic[check_in_hour_period[i]]

    current_index += tot_hour_period

    time_month_dic = {}
    tot_month = max(check_in_local_month)

    for i in range(len(check_in_local_month)):
        if check_in_local_month[i] not in time_month_dic:
            time_month_dic[check_in_local_month[i]] = check_in_local_month[i] + current_index
        check_in_local_month[i] = time_month_dic[check_in_local_month[i]]

    current_index += tot_month

    time_year_dic = {}
    tot_year = max(check_in_local_year) - min(check_in_local_year) + 1
    min_year = min(check_in_local_year)
    for i in range(len(check_in_local_year)):
        if check_in_local_year[i] not in time_year_dic:
            time_year_dic[check_in_local_year[i]] = check_in_local_year[i] + current_index - min_year
        check_in_local_year[i] = time_year_dic[check_in_local_year[i]]

    check_in['local_year'] = check_in_local_year
    check_in['local_month'] = check_in_local_month
    check_in['hour_period'] = check_in_hour_period
    check_in['local_time'] = check_in_local_time

    check_in.sort_values('local_time', ascending=True, inplace=True)

    check_in = check_in.drop(columns=['userid', 'Venue_id', 'utc_time', 'Timezone_offset', 'local_time'])
    return check_in, friend_list_index, venues_dic, time_hour_dic, time_month_dic, time_year_dic, \
            geo_relations, current_index, user_index, poi_index

def process_city_data(city, check_in, poi_data, users, venues, friend):
    # process friendship
    friend = pd.DataFrame(friend)
    friend.columns = ['user1', 'user2']  # rename columns to user1 and user2
    friend.drop_duplicates(subset=['user1', 'user2'], keep='first', inplace=True)  # drop duplicate rows
    friend = friend.reset_index(drop=True)  # reset index after drop

    friend_list = []
    for index, row in friend.iterrows():
        if row['user1'] != row['user2'] and row['user1'] in users and row['user2'] in users:
            friend_list.append([row['user1'], row['user2']])  # if the friendship is valid, add it to the list
    print(city + " friendship number: ", len(friend_list))

    # Sort the users in the list
    users_sort = list(sorted(list(users)))
    # Create a dictionary of the users and their index
    users_dic = {}
    for i in range(len(users_sort)):
        users_dic[users_sort[i]] = i

    # Create a dictionary of the start and end index of the users
    user_index = {}
    user_index['start'], user_index['end'] = 0, len(users_sort) - 1

    # recode venues
    venues_sort = list(sorted(list(venues)))
    venues_dic = {}
    for i in range(len(venues_sort)):
        venues_dic[venues_sort[i]] = i + len(users_sort)

    poi_index = {}
    poi_index['start'], poi_index['end'] = len(users_sort), len(users_sort) + len(venues_sort) - 1

    current_index = len(users_sort) + len(venues_sort) - 1

    # recode friendship
    friend_list_index = []
    for i in range(len(friend_list)):
        friend_list_index.append((users_dic[friend_list[i][0]], users_dic[friend_list[i][1]]))

    poi_data['venues_index'] = None
    poi_data_venue_index = []
    for index, row in poi_data.iterrows():
        poi_data_venue_index.append(venues_dic[row['Venue_id']])
    poi_data['venues_index'] = poi_data_venue_index

    # recode check_in
    poi_data = poi_data.sort_values('Venue_id', ascending=True, inplace=False)
    poi_data = poi_data.reset_index(drop=True)
    poi_lat_lon = {}  # {poi_index:(lat,lon)}
    for index, row in poi_data.iterrows():
        poi_lat_lon[row['venues_index']] = [row['latitude'], row['longitude']]

    # process check_in location into
    poi_location = {}

    check_in = check_in.sort_values('Venue_id', ascending=True, inplace=False)
    check_in = check_in.reset_index(drop=True)
    check_in['users_index'] = None
    check_in['venues_index'] = None

    check_in_users_index = []
    check_in_venues_index = []

    check_in_lat_lon = []


    for _, row in check_in.iterrows():
        check_in_users_index.append(users_dic[row['userid']])
        check_in_venues_index.append(venues_dic[row['Venue_id']])
        check_in_lat_lon.append(poi_lat_lon[venues_dic[row['Venue_id']]])


    check_in['users_index'] = check_in_users_index
    check_in['venues_index'] = check_in_venues_index
    check_in['lat_lon'] = check_in_lat_lon

    check_in_time = []
    check_in_local_time = []

    for index, row in check_in.iterrows():
        time_1 = row['utc_time'][:-10] + row['utc_time'][-4:]
        timezone_offset = row['Timezone_offset']
        # struct_time = time.mktime(time.strptime(time_1, "%a %b %d  %H:%M:%S %Y")) + timezone_offset * 60
        struct_time = (datetime.strptime(time_1, "%a %b %d  %H:%M:%S %Y") - datetime(1970, 1, 1)).total_seconds()
        local_time = time.localtime(struct_time)
        struct_time = time.mktime(time.strptime(time_1, "%a %b %d  %H:%M:%S %Y")) + timezone_offset * 60
        check_in_time.append(struct_time)
        check_in_local_time.append(local_time)

    check_in['time'] = check_in_time
    check_in['local_time'] = check_in_local_time
    check_in.sort_values('local_time', ascending=True, inplace=True)

    check_in = check_in.drop(columns=['userid', 'Venue_id', 'utc_time', 'Timezone_offset', 'local_time'])
    return check_in, friend_list_index, user_index, poi_index



def time_partition(day, hour, min):
    # days [0-6] per 1 day, hours [0-23] per 1 hour, minutes [0-1] per 30 minutes
    # the time will be partied into 7 * 24 * 2 index
    if 0 <= min < 30:
        return day * 48 + (hour + 1) * 2 - 1
    else:
        return day * 48 + (hour + 1) * 2

def transform_location(location):
    lat, lng = location[0], location[1]
    hash_4 = geohash.encode(lat, lng, precision=4)
    hash_5 = geohash.encode(lat, lng, precision=5)
    hash_6 = geohash.encode(lat, lng, precision=6)
    return hash_4, hash_5, hash_6

def load_extra_poi_info(city, venues, venues_dic, time_hour_dic, time_month_dic, current_index):
    base_dir = './data/raw/Venue_detail/'
    path = base_dir + city + '/'
    file_names = os.listdir(path)
    file_names = zip(file_names, range(len(file_names)))
    file_names = dict(file_names)
    all_side_info = {}

    all_contact = {} # contain all types of contact
    all_category_level_one = {} # contain all types of category level one
    all_category_level_two = {} # contain all types of category level two

    all_tip_count = [] # contain all tip count
    all_price_tier = {} # contain all price tier
    all_like_count = [] # contain all like count
    all_rating = [] # contain all rating
    all_photos_count = [] # contain all photos count

    tip_count_cut_num = 6 # cut the tip count into 6 bins
    like_count_cut_num = 6 # cut the like count into 6 bins
    rating_cut_num = 6 # cut the rating into 6 bins
    photos_count_cut_num = 6 # cut the photos count into 6 bins

    for file_name in file_names:
        # load extra poi info from json file
        with open(path + file_name, 'r') as f:
            extra_poi_info = json.load(f)
            side_info = {} # store the side info of each venue

            id = extra_poi_info['id']
            if id not in venues or id in all_side_info:
                # if the venue is not in the check-in data or the venue has been processed
                continue

            contacts_load = extra_poi_info['contact']
            contacts_processed = set()

            for contact in contacts_load.keys():
                if contact == 'facebookUsername' or contact == 'facebookName' or contact == 'formattedPhone':
                    # ignore these three types of contact as they are identical
                    continue
                if contact not in all_contact:
                    # add new contact type and assign a new index
                    all_contact[contact] = len(all_contact) + 1
                contacts_processed.add(all_contact[contact])

            if len(contacts_processed) == 0:
                contacts_processed.add(0) # 5 is the index of 'None'

            categories = extra_poi_info['categories']
            category_level_one_processed = set()
            category_level_two_processed = set()

            for category_info in categories:
                category_level_one = category_info['icon']['prefix'].split('/')[5]
                if category_level_one not in all_category_level_one:
                    all_category_level_one[category_level_one] = len(all_category_level_one) + 1
                category_level_one_processed.add(all_category_level_one[category_level_one])

                category_level_two = category_info['name']
                if category_level_two not in all_category_level_two:
                    all_category_level_two[category_level_two] = {}
                    all_category_level_two[category_level_two]['index'] = len(all_category_level_two)
                    all_category_level_two[category_level_two]['parent'] = all_category_level_one[
                        category_level_one]
                category_level_two_processed.add(all_category_level_two[category_level_two]['index'])

            tip_count = extra_poi_info['stats']['tipCount']
            all_tip_count.append(tip_count)
            # tip_count_processed = process_tip_count(tip_count)

            price_tier = extra_poi_info['price']['tier'] if 'price' in extra_poi_info.keys() else None
            if price_tier not in all_price_tier:
                all_price_tier[price_tier] = 1
            else:
                all_price_tier[price_tier] += 1
            price_tier_processed = 1 if price_tier is None else price_tier + 1

            like_count = extra_poi_info['likes']['count'] if 'likes' in extra_poi_info.keys() else None
            if like_count is None or like_count == -1:
                like_count = 0
            all_like_count.append(like_count)

            rating = extra_poi_info['rating'] if 'rating' in extra_poi_info.keys() else None
            all_rating.append(rating)

            photos_count = extra_poi_info['photos']['count']
            all_photos_count.append(photos_count)


            side_info['contact'] = contacts_processed
            side_info['category_level_one'] = category_level_one_processed
            side_info['category_level_two'] = category_level_two_processed
            side_info['tip_count'] = tip_count
            side_info['price_tier'] = price_tier_processed
            side_info['like_count'] = like_count
            side_info['rating'] = rating
            side_info['photos_count'] = photos_count
            all_side_info[venues_dic[id]] = side_info

    all_tip_count = np.array(all_tip_count)
    all_like_count = np.array(all_like_count)
    all_rating = np.array(all_rating)
    all_photos_count = np.array(all_photos_count)

    tip_count_cut = cut_all_count(all_tip_count, cut_num=tip_count_cut_num)
    like_count_cut = cut_all_count(all_like_count, cut_num=like_count_cut_num)
    rating_cut = cut_all_count(all_rating[all_rating != None], cut_num=rating_cut_num)
    photos_count_cut = cut_all_count(all_photos_count, cut_num=photos_count_cut_num)

    # transform the count into bins index
    for key in all_side_info.keys():
        all_side_info[key]['tip_count'] = process_counts(all_side_info[key]['tip_count'], tip_count_cut)
        all_side_info[key]['like_count'] = process_counts(all_side_info[key]['like_count'], like_count_cut)
        if all_side_info[key]['rating'] is None:
            all_side_info[key]['rating'] = 1
        else:
            all_side_info[key]['rating'] = process_counts(all_side_info[key]['rating'], rating_cut) + 1
        all_side_info[key]['photos_count'] = process_counts(all_side_info[key]['photos_count'], photos_count_cut)

    # recode side info

    # print(current_index)
    for key in all_side_info.keys():
        all_side_info[key]['contact'] = {current_index + val + 1 for val in all_side_info[key]['contact']}

    current_index += len(all_contact) + 1

    for key in all_category_level_one:
        all_category_level_one[key] = current_index + all_category_level_one[key]

    for key in all_category_level_two:
        all_category_level_two[key]['parent'] = current_index + all_category_level_two[key]['parent']

    for key in all_side_info.keys():
        all_side_info[key]['category_level_one'] = {current_index + val for val in
                                                    all_side_info[key]['category_level_one']}

    current_index += len(all_category_level_one)

    for key in all_category_level_two:
        all_category_level_two[key]['index'] = current_index + all_category_level_two[key]['index']

    for key in all_side_info.keys():
        all_side_info[key]['category_level_two'] = {current_index + val for val in
                                                    all_side_info[key]['category_level_two']}

    current_index += len(all_category_level_two)

    for key in all_side_info.keys():
        all_side_info[key]['tip_count'] = current_index + all_side_info[key]['tip_count']

    current_index += tip_count_cut_num

    for key in all_side_info.keys():
        all_side_info[key]['price_tier'] = current_index + all_side_info[key]['price_tier']

    current_index += 5  # 5 price_tier in tot(1 is None)

    for key in all_side_info.keys():
        all_side_info[key]['like_count'] = current_index + all_side_info[key]['like_count']

    current_index += like_count_cut_num

    for key in all_side_info.keys():
        all_side_info[key]['rating'] = current_index + all_side_info[key]['rating']

    current_index += rating_cut_num

    for key in all_side_info.keys():
        all_side_info[key]['photos_count'] = current_index + all_side_info[key]['photos_count']

    current_index += photos_count_cut_num

    print("Total vertex number", current_index)

    print(city + ' side information number: ', len(all_side_info))
    return all_side_info, all_category_level_two, current_index

def cut_all_count(all_count, cut_num):
    all_count = np.sort(all_count)
    cut_len = len(all_count) // cut_num
    cuts = []
    total = 0
    for count in all_count:
        total += 1
        if total == cut_len:
            cuts.append(count)
            total = 0
    return cuts

def process_counts(count, cuts):
    index = 1
    for cut in cuts:
        if count > cut:
            index += 1
        else:
            break
    return index

def extract_relations(city, check_ins: pd.DataFrame, friend,
                      time_hour_dic, time_month_dic, time_year_dic, poi_details, all_categories, args):
    relations = {}
    hyper_edges = {}
    total_relation = 0

    check_in_relation = {}
    friendship_relation = {}
    categories_relation = {}
    time_relation = {}
    contact_relation = {}
    poi_category_one_relation = {}
    poi_category_two_relation = {}
    poi_counts_relation = {}
    poi_price_relation = {}

    check_ins = np.array(check_ins)
    for i in range(len(check_ins)):
        check_in_relation[i] = check_ins[i]
    relations['check_in'] = check_in_relation
    # hyper_edges['check_in'] = check_in_relation
    # type 1 : check in
    total_relation += len(check_in_relation)

    for i in range(len(friend)):
        friendship_relation[i] = friend[i]
    if args.ablation is False or (args.ablation is True and 'social' not in args.ablation_list):
        relations['friendship'] = friendship_relation
        hyper_edges['friendship'] = friendship_relation
    # type 2 : friendship
    total_relation += len(friendship_relation)

    count = 0
    for key in all_categories.keys():
        category_level2 = all_categories[key]['index']
        category_level1 = all_categories[key]['parent']
        categories_relation[count] = [category_level1, category_level2]
        count += 1
    if args.ablation is False or (args.ablation is True and 'side_info' not in args.ablation_list):
        relations['category'] = categories_relation
        if args.use_kg:
            hyper_edges['category'] = categories_relation
    # type 3 : category
    total_relation += count

    count = 0
    for year in time_year_dic.keys():
        year_node = time_year_dic[year]
        for month in time_month_dic.keys():
            month_node = time_month_dic[month]
            for period in time_hour_dic.keys():
                time_relation[count] = [year_node, month_node, time_hour_dic[period]]
                count += 1
    if args.ablation is False or (args.ablation is True and 'side_info' not in args.ablation_list):
        relations['time'] = time_relation
        if args.use_kg:
            hyper_edges['time'] = time_relation
    total_relation += count
    # type 4 : time:  all possible (year, month, period) combinations

    count = 0
    count_contact = 0
    count_category_one = 0
    count_category_two = 0
    for poi in poi_details.keys():
        contacts = poi_details[poi]['contact']
        category_level_one = poi_details[poi]['category_level_one']
        category_level_two = poi_details[poi]['category_level_two']
        tip_count = poi_details[poi]['tip_count']
        price_tier = poi_details[poi]['price_tier']
        like_count = poi_details[poi]['like_count']
        rating = poi_details[poi]['rating']
        photos_count = poi_details[poi]['photos_count']

        contact_relation[count_contact] = [poi] + [contact for contact in contacts]
        count_contact += 1

        if len(category_level_one) > 0:
            poi_category_one_relation[count_category_one] = [poi] + [category for category in category_level_one]
            count_category_one += 1

        if len(category_level_two) > 0:
            poi_category_two_relation[count_category_two] = [poi] + [category for category in category_level_two]
            count_category_two += 1

        poi_counts_relation[count] = [poi, tip_count, like_count, rating, photos_count]
        poi_price_relation[count] = [poi, price_tier]
        count += 1
    if args.ablation is False or (args.ablation is True and 'side_info' not in args.ablation_list):
        relations['contact'] = contact_relation
        relations['poi_category_one'] = poi_category_one_relation
        relations['poi_category_two'] = poi_category_two_relation
        relations['poi_counts'] = poi_counts_relation
        relations['poi_price'] = poi_price_relation
        if args.use_kg:
            hyper_edges['contact'] = contact_relation
            hyper_edges['poi_category_one'] = poi_category_one_relation
            hyper_edges['poi_category_two'] = poi_category_two_relation
            hyper_edges['poi_counts'] = poi_counts_relation
            hyper_edges['poi_price'] = poi_price_relation

    total_relation += count * 2 + count_contact + count_category_one + count_category_two
    return relations, hyper_edges, total_relation



def load_data(city, args):


    def load_all(city, args):
        check_in, poi_data, users, venues, friend = load_city_data(city)

        check_in_processed, friend_processed, venues_dic, time_hour_dic, \
            time_month_dic,time_year_dic, geo_relations, current_index, user_index, poi_index = process_city_data_recode(city, check_in, poi_data, users, venues, friend)

        poi_details, all_categories, total_nodes = load_extra_poi_info(city, venues, venues_dic, time_hour_dic, time_month_dic, current_index)
        relations, hyper_edges, total_relation_num = extract_relations(city, check_in_processed, friend_processed,
                                      time_hour_dic, time_month_dic,time_year_dic, poi_details, all_categories, args)
        if args.ablation is False or (args.ablation is True and'side_info' not in args.ablation_list):
            relations['poi_geo'] = geo_relations
            if args.use_kg:
                hyper_edges['poi_geo'] = geo_relations

        total_relation_num += len(geo_relations)
        total_edge_num = total_relation_num

        print('Total edge number: ', total_edge_num)
        print('Total relation number:{}'.format(total_relation_num))
        print('load success')
        return relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes

    #load from file
    try:
        # raise Exception
        relations = pickle.load(open('./data/processed/'+city+'/relations.pkl', 'rb'))
        hyper_edges = pickle.load(open('./data/processed/'+city+'/hyper_edges.pkl', 'rb'))
        total_relation_num = pickle.load(open('./data/processed/'+city+'/total_relation_num.pkl', 'rb'))
        total_edge_num = pickle.load(open('./data/processed/'+city+'/total_edge_num.pkl', 'rb'))
        user_index = pickle.load(open('./data/processed/'+city+'/user_index.pkl', 'rb'))
        poi_index = pickle.load(open('./data/processed/'+city+'/poi_index.pkl', 'rb'))
        total_nodes = pickle.load(open('./data/processed/'+city+'/total_nodes.pkl', 'rb'))
        print('load data from file success')
        return relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes
    except:
        # print('load failed')
        print('load data from file failed, load from raw data')
        relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes = load_all(city, args)
        pickle.dump(relations, open('./data/processed/'+city+'/relations.pkl', 'wb'))
        pickle.dump(hyper_edges, open('./data/processed/'+city+'/hyper_edges.pkl', 'wb'))
        pickle.dump(total_relation_num, open('./data/processed/'+city+'/total_relation_num.pkl', 'wb'))
        pickle.dump(total_edge_num, open('./data/processed/'+city+'/total_edge_num.pkl', 'wb'))
        pickle.dump(user_index, open('./data/processed/'+city+'/user_index.pkl', 'wb'))
        pickle.dump(poi_index, open('./data/processed/'+city+'/poi_index.pkl', 'wb'))
        pickle.dump(total_nodes, open('./data/processed/'+city+'/total_nodes.pkl', 'wb'))
        print('load data from raw data success')
        return relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes

def ConstructV2V(edge_index):
    # Assume edge_index = [V;E], sorted
    """
    For each he, clique-expansion. Note that we DONT allow duplicated edges.
    Instead, we record its corresponding weights.
    We default no self loops so far.
    """
    
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1
            else:
                edge_weight_dict[comb] += 1

# # Now, translate dict to edge_index and norm
#
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1
        
    return new_edge_index, new_norm


def norm_contruction(edge_index,edge_weight, TYPE='V2V'):
    if TYPE == 'V2V':
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, add_self_loops=True)
    return edge_index, edge_weight




def process_data_HKG(args):
    # load data
    relations, edges, total_relations, total_edges, user_index, poi_index, total_nodes = load_data(args.city, args)
    all_relations = []

    
    test_data = {}
    train_data = {}
    check_in = relations['check_in']
    check_in_test = []
    check_in_train = []

    user_length = {} # The number of check-ins of each user
    current_user_length = {}

    for i in range(len(check_in)):
        if check_in[i][0] not in user_length:
            user_length[check_in[i][0]] = 1
        else:
            user_length[check_in[i][0]] += 1
    # record the number of check-ins for each user

    for i in range(len(check_in)):
        if check_in[i][0] not in current_user_length:
            current_user_length[check_in[i][0]] = 1
            check_in_train.append(check_in[i])
        else:
            # if current_user_length[check_in[i][0]] < user_length[check_in[i][0]] - 2:
            if current_user_length[check_in[i][0]] < user_length[check_in[i][0]]*0.7:
                check_in_train.append(check_in[i])
                current_user_length[check_in[i][0]] += 1
            else:
                check_in_test.append(check_in[i])



    check_in_train = np.array(check_in_train)
    check_in_test = np.array(check_in_test)
    # check_in = np.vstack((check_in_train, check_in_test))

    # test_data['check_in'] = check_in_test
    train_data['check_in'] = np.vstack((check_in_train, check_in_test))

    
    if args.ablation is False or (args.ablation is True and 'check_in' not in args.ablation_list):
        relations['check_in'] = {}
        for i in range(len(check_in_train)):
            relations['check_in'][i] = check_in_train[i]
        edges['check_in'] = copy.deepcopy(relations['check_in'])
    # test_index['check_in'] = set(np.random.choice(len(relations['check_in']), int(len(relations['check_in']) * (1-args.train_ratio)), replace=False))


    relation_count = 0
    for relation in relations.keys():
        if relation not in args.ablation_list or args.ablation is False:
            for idx in relations[relation].keys():
                all_relations.append([relation_count, relations[relation][idx]])
        relation_count += 1
    # combine all relations

    v_index = []
    e_index = []
    count = 0
    for edge in edges.keys():
        for idx in edges[edge].keys():
            # all_edges.append(edges[edge][idx])
            for v in edges[edge][idx]:
                v_index.append(v)
                e_index.append(count)
            count += 1
    # combine all edges

    hyperedge_num = count + 1
    # hypernode_num = np.array(all_relations)

    v_e = np.array([v_index, e_index])
    v_e = v_e.T[np.lexsort(v_e[::-1, :])].T

    relation_num = 0
    for keys, item in relations.items():
        relation_num += len(item)

     # = max(relation[0] for relation in all_relations) + 1
    entity_num = max([max(relation[1]) for relation in all_relations])
    print('Number of facts:', relation_count)
    print('Number of entities:', entity_num)
    print('Number of relations:', relation_num)
    return all_relations, relation_count, v_e, test_data, train_data, user_index, poi_index, hyperedge_num, total_nodes

def process_data_Flashback(args):
    check_in, poi_data, users, venues, friend = load_city_data(args.city)
    check_in, friend_list, user_index, poi_index = process_city_data(args.city, check_in, poi_data, users, venues, friend)
    return check_in, user_index, poi_index

def process_data_Graph_Flashback(args):
    check_in, poi_data, users, venues, friend = load_city_data(args.city)
    check_in, friend_list, user_index, poi_index = process_city_data(args.city, check_in, poi_data, users, venues, friend)
    lat = []
    lon = []
    for iter, row in check_in.iterrows():
        lat.append(row['lat_lon'][0])
        lon.append(row['lat_lon'][1])
    check_in['lat'] = lat
    check_in['lon'] = lon
    poi_data = np.array(check_in.groupby('venues_index').agg({'lat_lon': 'first'}).reset_index()).T
    check_in.drop(['lat', 'lon'], axis=1, inplace=True)

    poi_data = poi_data[1]
    return check_in, user_index, poi_index, poi_data, friend_list

def process_data_Stan(args):
    check_in, poi_data, users, venues, friend = load_city_data(args.city)
    check_in, friend_list, user_index, poi_index = process_city_data(args.city, check_in, poi_data, users, venues, friend)
    lat = []
    lon = []
    for iter, row in check_in.iterrows():
        lat.append(row['lat_lon'][0])
        lon.append(row['lat_lon'][1])
    check_in['lat'] = lat
    check_in['lon'] = lon
    poi_data = np.array(check_in.groupby('venues_index').agg({'lat_lon': 'first'}).reset_index()).T
    check_in.drop(['lat_lon'], axis=1, inplace=True)

    poi_data = poi_data[1]
    return check_in, user_index, poi_index, poi_data



