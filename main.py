
#exec(open('algorithm_1_function.py').read())

from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
import numpy as np
import time
import math
import os
import json
import polyline
from algorithm_1_function import algorithm_1_function
from algo2 import algorithm2



app = Flask(__name__)


pathway = app.root_path
postcode_lookup = pd.read_parquet('ukpostcodes_processed.parquet')
ratings = pd.read_csv('ratings.csv')   # used in algo 1






@app.route('/',methods = ['GET','POST'])
def landing():
    
    if request.method == 'POST':
        postcode_first=request.form['postcode_first']
        postcode_destination=request.form['postcode_destination']
        speed_comfort=request.form['speed_comfort']
        ev_charge_speed=request.form['ev_charge_speed']
        battery_size=request.form['battery_size']
        max_range=request.form['max_range']

        postcode_first = postcode_first.upper()
        postcode_destination = postcode_destination.upper()

        print(postcode_first)
        print(postcode_destination)

        if np.sum(postcode_lookup.postcode == postcode_first) < 0.5:
            return "<h1>Postcode for place you're leaving isn't recognised :( Please go back and check you put it in correctly</h1>"
        if np.sum(postcode_lookup.postcode == postcode_destination) < 0.5:
            return "<h1>Postcode for your destination isn't recognised :( Please go back and check you put it in correctly</h1>"



        print(postcode_first)
        print(postcode_destination)
        print(speed_comfort)
        print(ev_charge_speed)
        print(battery_size)
        print(max_range)

        speed_comfort = float(speed_comfort) / 100 # scaling to 0 - 1


        # get latlong of start and end
        print(postcode_first)
        first_ix = postcode_lookup['postcode'] == postcode_first
        second_ix = postcode_lookup['postcode'] == postcode_destination
        print(postcode_lookup.loc[first_ix,'latitude'])
        print(float(postcode_lookup.loc[first_ix,'latitude']))
        lat1 = float(postcode_lookup.loc[first_ix,'latitude'])
        long1 = float(postcode_lookup.loc[first_ix,'longitude'])

        latlong1 = [lat1, long1]
        print(latlong1)

        lat2 = float(postcode_lookup.loc[second_ix,'latitude'])
        long2 = float(postcode_lookup.loc[second_ix,'longitude'])
        latlong2 = [lat2, long2]


        # getting centre point for map and estimating best starting zoom level
        center = [(latlong1[0]+latlong2[0])/2, (latlong1[1]+latlong2[1])/2]

        # 70 miles per lat.long unit
        crow_flies_distance = np.sqrt((latlong1[0]-latlong2[0])**2 + (latlong1[1]-latlong2[1])**2) * 70
        if crow_flies_distance > 400:
             zoom_level = [7.0]
        elif crow_flies_distance > 200:
             zoom_level = [8.0]
        elif crow_flies_distance > 70:
             zoom_level = [10.0]
        else:
            zoom_level = [11.0]
        print('zoom level')
        print(zoom_level)






        #### do Algo 1
        # ratings (df)

        # sample_ncr = algo 1 output
        start_postcode = postcode_first
        end_postcode = postcode_destination
        comfort_rating = speed_comfort * 5  # scaled to 0-5
        ncr_data = ratings
        sample_ncr = algorithm_1_function(start_postcode,end_postcode,postcode_lookup,ncr_data,comfort_rating)
        print('finished algo 1')

        # ensure no more than 97 rows (update 12th Oct: no longer used as OSRM limit increased)
        #len_ncr = len(sample_ncr)
        #print(len_ncr)
        #rows_to_include = np.minimum(len_ncr, 97)
        #sample_ncr = sample_ncr.iloc[:rows_to_include,:]  # 


        #sample_ncr.to_csv('/Users/dftdatascience/Desktop/ev-charge-planner/flask/sample_ncr_created.csv')






        # input for Algo 2
        latlong_first=(long1, lat1)   # note that OSRM likes long/lat instead of usual lat/long order
        latlong_destination=(long2, lat2)
        speed_comfort
        ev_charge_speed=float(ev_charge_speed)
        max_range=float(max_range)
        battery_size=float(battery_size)
        
        



        # algo 2 inputs dict for testing
        algo2_inputs_dict = {
        'sample_ncr':sample_ncr,
        'latlong_first':latlong_first,
        'latlong_destination':latlong_destination,
        'speed_comfort':speed_comfort,
        'ev_charge_speed':ev_charge_speed,
        'max_range':max_range,
        'battery_size':battery_size
        }

        print(type(algo2_inputs_dict))


        #outfile = '/Users/dftdatascience/Desktop/ev-charge-planner/flask/algo2_inputs_dict.npz'
        #np.savez(outfile, **algo2_inputs_dict)


        # run Algo 2
        print('about to start algo 2')
        read_dictionary = algorithm2(sample_ncr,latlong_first,latlong_destination,speed_comfort,ev_charge_speed,max_range,battery_size)


        # catching where no need to charge
        if read_dictionary == '<h1>It looks like you can make the journey without stopping to charge! :)</h1>':
            print('can make journey without stopping to charge')
            return read_dictionary  

        if read_dictionary == "cant_find_perfect_route":
            print('cant_find_perfect_route')
            return "<h2>Sorry, it looks like we can't find you the perfect route :(</h2>\n\
        <h3>Your journey might need too many stops (ie, 8 or more!)</h3>\n\
        <h3>Or we might need to improve our work-in-progress algorithm :/</h3>\n\
        <h3>Try going back and adjusting the inputs</h3>"  

        # Load results of algo 2 from jupyter
        #read_dictionary = np.load('/Users/dftdatascience/Desktop/ev-charge-planner/flask/dummy_output.npy',allow_pickle='TRUE').item()
        

        # make lines from polylines and converting lines to trail of points (10 for each original point)
        lines = read_dictionary['polyline_array']
        p1, p2 = [], []
        for i in range(len(lines) - 1):
            p1.extend(list(np.linspace(lines[i+1][0], lines[i][0], num=10)))
            p2.extend(list(np.linspace(lines[i+1][1], lines[i][1], num=10)))

        lines_v2 = list(zip(p1, p2))
        lines_v2 = [list(x) for x in lines_v2]
        lines = lines_v2  # overwriting
        print('lines')




        # updating with dummary data
        dict_for_sidebar={
        "postcode_destination":postcode_destination,
        }

        dict_for_sidebar["time_charging"] = read_dictionary['time_charging']*60
        dict_for_sidebar["top_percent_locations_to_charge"]= read_dictionary['journey_niceness']
        dict_for_sidebar["hrs_driving"]=read_dictionary['hrs_driving']
        dict_for_sidebar["total_miles"]=read_dictionary['total_miles']
        dict_for_sidebar["destination_names"]=read_dictionary['destination_names']
        dict_for_sidebar["destination_postcodes"]=read_dictionary['pcodes']
        dict_for_sidebar["marker_coords"]=read_dictionary['marker_coords']
        dict_for_sidebar["place_names"]=read_dictionary['place_names']
        dict_for_sidebar["journey_niceness"]=read_dictionary['journey_niceness']


        print('zoom level')
        print(zoom_level)


        print('crow_flies_distance')
        print(crow_flies_distance)

        print('journey_niceness')
        print(read_dictionary['journey_niceness'])



        return render_template('map_output.html', points_from_flask=lines, dict_for_sidebar=dict_for_sidebar, center=center,
            zoom_level=zoom_level)

    return render_template('landing.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)






