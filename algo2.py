

from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
import numpy as np
import time
import math
import os
import json
import polyline


# these two lines: making a new np.load func which allows pickling
np_load_old = np.load

def np_load_allow_pickle(*a, **k):
    """Allows numpy func to read in dict of arrays (I think thats the issue it solves)"""
    return np.load(*a, allow_pickle=True, **k)
#lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# load
"""
dict_from_app = np_load_allow_pickle('/Users/dftdatascience/Desktop/ev-charge-planner/flask/algo2_inputs_dict.npz')


# convert input to dict

algo2_inputs_dict = {}
for file in dict_from_app.files:
    algo2_inputs_dict[file] = dict_from_app[file]

algo2_inputs_dict.keys()



### loading input from dict
sample_ncr=algo2_inputs_dict['sample_ncr']
latlong_first=algo2_inputs_dict['latlong_first']
latlong_destination=algo2_inputs_dict['latlong_destination']
speed_comfort=algo2_inputs_dict['speed_comfort']
ev_charge_speed=algo2_inputs_dict['ev_charge_speed']
max_range=algo2_inputs_dict['max_range']
battery_size=algo2_inputs_dict['battery_size']
sample_ncr = pd.DataFrame(sample_ncr, columns=['latitude', 'longitude','rating', 'name', 'FastestConnector_kW'])
"""


osrm_url = '0.0.0.0:5000'
osrm_url = 'http://34.70.0.117:5000'
#osrm_url = 'http://34.132.175.14:5000'



def algorithm2(sample_ncr,latlong_first,latlong_destination,speed_comfort,ev_charge_speed,max_range,battery_size):

    sample_ncr_len = len(sample_ncr)

    #sample_ncr['lat_long'] = sample_ncr['latitude'].astype('str') + ',' + sample_ncr['longitude'].astype('str')
    sample_ncr['lat_long'] = sample_ncr['longitude'].astype('str') + ',' + sample_ncr['latitude'].astype('str')

    for_query = sample_ncr['lat_long']
    latlong_text_for_query = for_query.str.cat(sep=';')

    latlong_first_string = ','.join([str(x) for x in latlong_first])
    latlong_destination_string = ','.join([str(x) for x in latlong_destination])

    full_query_text = osrm_url + '/table/v1/driving/' + latlong_first_string + ';' + latlong_destination_string + ';' + latlong_text_for_query

    result = os.popen("curl '" + full_query_text + "'").read()

    travel_time_matrix = json.loads(result)['durations']
    travel_time_matrix = np.asarray(travel_time_matrix)

    full_query_text = osrm_url + '/route/v1/driving/' + latlong_first_string + ';' + latlong_destination_string + '?overview=false'
    result = os.popen("curl '" + full_query_text + "'").read()

    distance_start_to_end_miles = json.loads(result)['routes'][0]['legs'][0]['distance'] / 1600
    start_to_end_direct_seconds = json.loads(result)['routes'][0]['legs'][0]['duration']

    full_query_text_distances = osrm_url + '/table/v1/driving/' + latlong_first_string + ';' + latlong_destination_string + ';' + latlong_text_for_query + '?annotations=distance'
    result = os.popen("curl '" + full_query_text_distances + "'").read()

    distance_matrix = json.loads(result)['distances']
    distance_matrix = np.asarray(distance_matrix)
    distance_matrix = distance_matrix / 1600  # converting from metres to miles

    user_review_ratings = np.append(np.zeros(2), sample_ncr.rating)


    # returning html if journey can be made in one go, so no need for stopping anywhere
    if (distance_matrix[0, 1] < (max_range * 0.95)):
        return ('<h1>It looks like you can make the journey without stopping to charge! :)</h1>')



    # getting all possible combinations
    max_range = float(max_range)
    stops_count = np.min([(math.ceil((1.2 * distance_start_to_end_miles) / max_range) - 1), 6])  # maximum of 6 stop counts for computational tractability

    print('distance_start_to_end_miles')
    print(distance_start_to_end_miles)
    print(max_range)
    print((1.2 * distance_start_to_end_miles) / max_range)


    points_between_dict = {}          # #latlong_first is (long, lat)
    for i in range(1,stops_count+1):
        points_between_dict[i] = [
                                latlong_first[0] - ((latlong_first[0] - latlong_destination[0]) * (i / (stops_count+1))),
                                latlong_first[1] - ((latlong_first[1] - latlong_destination[1]) * (i / (stops_count+1))),
                                ]

    dist_between_points = math.sqrt((latlong_first[0] - latlong_destination[0])**2 + (latlong_first[0] - latlong_destination[0])**2) / (stops_count+1)

    # get list of tuples of start, destination, and all options
    list_tuples_coords = [latlong_first, latlong_destination]
    list_tuples_coords_all_cands = list(zip(sample_ncr['longitude'], sample_ncr['latitude']))
    for i in range(len(list_tuples_coords_all_cands)):
        list_tuples_coords.append(list_tuples_coords_all_cands[i])
    #list_tuples_coords = list_tuples_coords.append(list_tuples_coords_all_cands)
    print('len(list_tuples_coords)')
    print(len(list_tuples_coords))
    np_array_coords = np.array([list(x) for x in list_tuples_coords])
    print('np_array_coords shape')
    print(np_array_coords.shape)


    # find which of these are in the circles - and thus candidates for each stop
    candidates_for_each_stop_dict = {}
    for i in range(1,stops_count+1):
        longitude = points_between_dict[i][0]
        latitude = points_between_dict[i][1]

        distances =  np.sqrt((np_array_coords[:,0] - longitude)**2 + (np_array_coords[:,1] - latitude)**2)
        acceptable_candidates = np.where(distances < (dist_between_points * 1.4))[0]
        print(acceptable_candidates)
        candidates_for_each_stop_dict[i] = acceptable_candidates


    print('stops_count')
    print(stops_count)


    # getting all possible routes
    if stops_count == 1:
        journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], [1])).T.reshape(-1,3)
    if stops_count == 2:
        journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],[1])).T.reshape(-1,4)
    if stops_count == 3:
        journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],[1])).T.reshape(-1,5)
    if stops_count == 4:
        journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],[1])).T.reshape(-1,6)
    if stops_count == 5:
        journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],candidates_for_each_stop_dict[5],[1])).T.reshape(-1,7)
    if stops_count == 6:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],candidates_for_each_stop_dict[5],candidates_for_each_stop_dict[6],[1])).T.reshape(-1,8)


    journeys = np.array(journeys)

    print('shape')
    print(journeys.shape)
    journeys = journeys.astype('int16')




    ##### all journeys got by now!




    # get journey times and distances
    journey_times = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
    for i in range(journey_times.shape[1]):
        journey_times[:, i] = travel_time_matrix[journeys[:, i], journeys[:, i + 1]]

    journey_distances = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
    for i in range(journey_distances.shape[1]):
        print(i)
        print(journeys[:, i + 1])
        journey_distances[:, i] = distance_matrix[journeys[:, i], journeys[:, i + 1]]

    
    # filtering for routes which dont exceed max range at any point
    a = np.max(journey_distances, axis=1)
    ix = a < max_range
    print('sum legal routes:')
    print(np.sum(ix))
    journeys = journeys[ix, :]  
    print('journeys shape')
    print(journeys.shape)









    ##### 2nd attempt

    # trying adding one more stop with a slightly more generous radius if the above doesn't work
    if (float(np.sum(ix)) < 0.5):
        stops_count = stops_count + 1


        points_between_dict = {}          # #latlong_first is (long, lat)
        for i in range(1,stops_count+1):
            points_between_dict[i] = [
                                    latlong_first[0] - ((latlong_first[0] - latlong_destination[0]) * (i / (stops_count+1))),
                                    latlong_first[1] - ((latlong_first[1] - latlong_destination[1]) * (i / (stops_count+1))),
                                    ]

        dist_between_points = math.sqrt((latlong_first[0] - latlong_destination[0])**2 + (latlong_first[0] - latlong_destination[0])**2) / (stops_count+1)

        # get list of tuples of start, destination, and all options
        list_tuples_coords = [latlong_first, latlong_destination]
        list_tuples_coords_all_cands = list(zip(sample_ncr['longitude'], sample_ncr['latitude']))
        for i in range(len(list_tuples_coords_all_cands)):
            list_tuples_coords.append(list_tuples_coords_all_cands[i])
        #list_tuples_coords = list_tuples_coords.append(list_tuples_coords_all_cands)
        print('len(list_tuples_coords)')
        print(len(list_tuples_coords))
        np_array_coords = np.array([list(x) for x in list_tuples_coords])
        print('np_array_coords shape')
        print(np_array_coords.shape)


        # find which of these are in the circles - and thus candidates for each stop
        candidates_for_each_stop_dict = {}
        for i in range(1,stops_count+1):
            longitude = points_between_dict[i][0]
            latitude = points_between_dict[i][1]

            distances =  np.sqrt((np_array_coords[:,0] - longitude)**2 + (np_array_coords[:,1] - latitude)**2)
            acceptable_candidates = np.where(distances < (dist_between_points * 1.6))[0]   # radius of 1.6 mult instead of 1.4
            print(acceptable_candidates)
            candidates_for_each_stop_dict[i] = acceptable_candidates


        print('stops_count')
        print(stops_count)


        # getting all possible routes
        if stops_count == 1:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], [1])).T.reshape(-1,3)
        if stops_count == 2:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],[1])).T.reshape(-1,4)
        if stops_count == 3:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],[1])).T.reshape(-1,5)
        if stops_count == 4:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],[1])).T.reshape(-1,6)
        if stops_count == 5:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],candidates_for_each_stop_dict[5],[1])).T.reshape(-1,7)
        if stops_count == 6:
                journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],candidates_for_each_stop_dict[5],candidates_for_each_stop_dict[6],[1])).T.reshape(-1,8)


        journeys = np.array(journeys)

        print('shape')
        print(journeys.shape)
        journeys = journeys.astype('int16')




        ##### all journeys got by now!




        # get journey times and distances
        journey_times = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
        for i in range(journey_times.shape[1]):
            journey_times[:, i] = travel_time_matrix[journeys[:, i], journeys[:, i + 1]]

        journey_distances = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
        for i in range(journey_distances.shape[1]):
            print(i)
            print(journeys[:, i + 1])
            journey_distances[:, i] = distance_matrix[journeys[:, i], journeys[:, i + 1]]

        
        # filtering for routes which dont exceed max range at any point
        a = np.max(journey_distances, axis=1)
        ix = a < max_range
        print('sum legal routes:')
        print(np.sum(ix))
        journeys = journeys[ix, :]  
        print('journeys shape')
        print(journeys.shape)

        print('end of 2nd attempt')



    #### end of 2nd attempt










    #### 3rd attempt to mop up really tricky cases!
    if (float(np.sum(ix)) < 0.5):
        stops_count = stops_count + 1


        points_between_dict = {}          # #latlong_first is (long, lat)
        for i in range(1,stops_count+1):
            points_between_dict[i] = [
                                    latlong_first[0] - ((latlong_first[0] - latlong_destination[0]) * (i / (stops_count+1))),
                                    latlong_first[1] - ((latlong_first[1] - latlong_destination[1]) * (i / (stops_count+1))),
                                    ]

        dist_between_points = math.sqrt((latlong_first[0] - latlong_destination[0])**2 + (latlong_first[0] - latlong_destination[0])**2) / (stops_count+1)

        # get list of tuples of start, destination, and all options
        list_tuples_coords = [latlong_first, latlong_destination]
        list_tuples_coords_all_cands = list(zip(sample_ncr['longitude'], sample_ncr['latitude']))
        for i in range(len(list_tuples_coords_all_cands)):
            list_tuples_coords.append(list_tuples_coords_all_cands[i])
        #list_tuples_coords = list_tuples_coords.append(list_tuples_coords_all_cands)
        print('len(list_tuples_coords)')
        print(len(list_tuples_coords))
        np_array_coords = np.array([list(x) for x in list_tuples_coords])
        print('np_array_coords shape')
        print(np_array_coords.shape)


        # find which of these are in the circles - and thus candidates for each stop
        candidates_for_each_stop_dict = {}
        for i in range(1,stops_count+1):
            longitude = points_between_dict[i][0]
            latitude = points_between_dict[i][1]

            distances =  np.sqrt((np_array_coords[:,0] - longitude)**2 + (np_array_coords[:,1] - latitude)**2)
            acceptable_candidates = np.where(distances < (dist_between_points * 1.8))[0]   # radius of 1.4 mult instead of 1.4
            print(acceptable_candidates)
            candidates_for_each_stop_dict[i] = acceptable_candidates


        print('stops_count')
        print(stops_count)


        # getting all possible routes
        if stops_count == 1:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], [1])).T.reshape(-1,3)
        if stops_count == 2:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],[1])).T.reshape(-1,4)
        if stops_count == 3:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],[1])).T.reshape(-1,5)
        if stops_count == 4:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],[1])).T.reshape(-1,6)
        if stops_count == 5:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],candidates_for_each_stop_dict[5],[1])).T.reshape(-1,7)
        if stops_count == 6:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],candidates_for_each_stop_dict[5],candidates_for_each_stop_dict[6],[1])).T.reshape(-1,8)
        if stops_count == 7:
            journeys = np.array(np.meshgrid([0], candidates_for_each_stop_dict[1], candidates_for_each_stop_dict[2],candidates_for_each_stop_dict[3],candidates_for_each_stop_dict[4],candidates_for_each_stop_dict[5],candidates_for_each_stop_dict[6],candidates_for_each_stop_dict[7],[1])).T.reshape(-1,9)


        journeys = np.array(journeys)

        print('shape')
        print(journeys.shape)
        journeys = journeys.astype('int16')




        ##### all journeys got by now!




        # get journey times and distances
        journey_times = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
        for i in range(journey_times.shape[1]):
            journey_times[:, i] = travel_time_matrix[journeys[:, i], journeys[:, i + 1]]

        journey_distances = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
        for i in range(journey_distances.shape[1]):
            print(i)
            print(journeys[:, i + 1])
            journey_distances[:, i] = distance_matrix[journeys[:, i], journeys[:, i + 1]]

        
        # filtering for routes which dont exceed max range at any point
        a = np.max(journey_distances, axis=1)
        ix = a < max_range
        print('sum legal routes:')
        print(np.sum(ix))
        journeys = journeys[ix, :]  
        print('journeys shape')
        print(journeys.shape)

        print('end of 3rd attempt')









    ##### end of 3rd attempt


    print('float sum ix of ')
    print(float(np.sum(ix)))



    if (float(np.sum(ix)) < 0.5):
        return "cant_find_perfect_route"


    # getting new journey times and distances after filter
    journey_times = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
    for i in range(journey_times.shape[1]):
        journey_times[:, i] = travel_time_matrix[journeys[:, i], journeys[:, i + 1]]

    journey_distances = np.zeros((journeys.shape[0], journeys.shape[1] - 1))
    for i in range(journey_distances.shape[1]):
        print(i)
        print(journeys[:, i + 1])
        journey_distances[:, i] = distance_matrix[journeys[:, i], journeys[:, i + 1]]


    """
    ix = distance_matrix[:,1] < max_range
    np.isin(journeys[:,3] , np.where(ix)[0])
    """


    #np.savetxt('/Users/dftdatascience/Desktop/ev-charge-planner/flask/journey_distances.csv', journey_distances)

    # calc charging times using NCR and car kW
    a = np.asarray([99999999, 99999999]).astype('int')
    b = np.asarray(sample_ncr.FastestConnector_kW).astype('int')
    charge_speeds = np.concatenate([a, b])
    charge_speeds = np.minimum(charge_speeds, [ev_charge_speed])  # lowest of car or charger charge speed

    # matrix of charging speeds
    charge_speeds_all_stops = np.zeros((journeys.shape[0], journeys.shape[1] - 2))
    for i in range(charge_speeds_all_stops.shape[0]):
        for j in range(charge_speeds_all_stops.shape[1]):
            charge_speeds_all_stops[i, j] = charge_speeds[journeys[i, j + 1]]

    charge_times_all_stops = np.zeros((journeys.shape[0], journeys.shape[1] - 2))
    for stop_count in range(1, charge_speeds_all_stops.shape[1] + 1):
        charge_left = np.maximum(max_range - journey_distances[:, stop_count - 1], [0])
        charge_needed = np.maximum(journey_distances[:, stop_count] - charge_left, [0])
        hours_to_charge = (charge_needed / max_range) * battery_size / charge_speeds_all_stops[:, stop_count - 1]
        charge_times_all_stops[:, stop_count - 1] = hours_to_charge

    quality_all_stops = np.zeros((journeys.shape[0], journeys.shape[1] - 2))
    for stop_count in range(1, charge_speeds_all_stops.shape[1] + 1):
        quality_all_stops[:, stop_count - 1] = user_review_ratings[journeys[:, stop_count]]

    # journey-level stats for final calculation
    print('charge_times_all_stops')
    print(charge_times_all_stops)
    print('quality_all_stops')
    print(quality_all_stops)
    print('shapes')
    print(charge_times_all_stops.shape)
    print(quality_all_stops.shape)

    journey_niceness_weighted_avg = np.average(quality_all_stops, axis=1, weights=charge_times_all_stops)
    total_charge_time = np.sum(charge_times_all_stops, axis=1)
    total_journey_time = np.sum(journey_times, axis=1) / 3600

    # overall scores: is sensitive to the multiplier below, where a higher multiplier gives more weight to comfort
    sensitive_multiplier = 0.5
    score = (total_charge_time * journey_niceness_weighted_avg * speed_comfort * sensitive_multiplier) - total_journey_time - total_charge_time

    # removing invalid routes according to distance
    #valids_valids_ix = journey_distances.max(axis=1) < float(max_range)
    #ix = journey_distances.max(axis=1) < max_range
    #valids_ix_rowid = np.where(valids_ix)
    #best_journey_pos_of_valids = np.argmax(score[valids_ix_rowid]) # only look at valid rows
    #best_journey_pos = valids_ix_rowid[best_journey_pos_of_valids] # find pos within all rows

    best_journey_pos = np.argmax(score)
    best_journey = journeys[best_journey_pos]

    # formatting for start/end too
    to_add = pd.DataFrame({'name': ['start', 'end'],
                           'latitude': [latlong_first[1], latlong_destination[1]],
                           'longitude': [latlong_first[0], latlong_destination[0]]})

    # extracting winning journey coords and service stat names
    sample_ncr2 = to_add.append(sample_ncr[['name', 'latitude', 'longitude']])
    output_results = sample_ncr2.iloc[best_journey,]


    # bit of a hack: removing wherever a 2nd 'end' row is added (duplicate row)
    ix = range(0, np.where(output_results.name=='end')[0][0] + 1)
    output_results = output_results.iloc[ix, :]

    # get polylines for winning journey
    polyline_dict = {}
    for i in range(len(output_results) - 1):
        pair1 = "{:.6f}".format(output_results.iloc[i, :]['longitude']) + ',' + "{:.6f}".format(output_results.iloc[i, :][
            'latitude'])
        pair2 = "{:.6f}".format(output_results.iloc[i + 1, :]['longitude']) + ',' + "{:.6f}".format(output_results.iloc[i + 1, :][
            'latitude'])

        url_for_polylines = osrm_url + '/route/v1/car/' + pair1 + ';' + pair2
        response = os.popen("curl '" + url_for_polylines + "'").read()
        json_data = json.loads(response)
        poly_out = json_data["routes"][0]["geometry"]
        polyline_dict[i] = polyline.decode(poly_out)

    polyline_array = []  # convert to array
    keys = polyline_dict.keys()
    for key in polyline_dict:
        polyline_array.extend(polyline_dict[key])

    # formatting and prepping outputs
    destination_names = output_results.name[1:len(output_results) - 1]
    destination_names.reset_index(drop=True, inplace=True)
    destination_names = destination_names.tolist()
    pcodes = output_results.index[1:len(output_results) - 1]
    pcodes = pcodes.tolist()

    hrs_driving = total_journey_time[best_journey_pos]
    total_miles = np.sum(journey_distances, axis=1)[best_journey_pos]
    time_charging = total_charge_time[best_journey_pos]
    journey_niceness = journey_niceness_weighted_avg[best_journey_pos]

    marker_coords = list(zip(output_results.latitude, output_results.longitude))
    marker_coords = [list(x) for x in marker_coords]

    place_names = output_results.name.tolist()

    output_vals = {'polyline_array': polyline_array,
                   'output_results': output_results,
                   'destination_names': destination_names,
                   'pcodes': pcodes,
                   'place_names': place_names,
                   'hrs_driving': hrs_driving,
                   'total_miles': total_miles,
                   'time_charging': time_charging,
                   'journey_niceness': journey_niceness,
                   'marker_coords': marker_coords}

    return output_vals















