# function which runs algorithm 1 script

import numpy as np 
import math


def algorithm_1_function(start_postcode,end_postcode,postcode_lookup,ncr_data,comfort_rating):


    # Set variables ######################################

    # number of chargers filter
    # must have atleast this number of chargepoints
    # this is a start value which may increase to reduce chargepoint options
    number_charge_points_threshold = 1

    # threshold for chargepoint quality "rating"
    # this is a start value which may increase to reduce chargepoint options
    # comfort rating is given as a number between 0 and 1 so mulitple by 5
    rating_threshold = float(comfort_rating)

    # max rating possible
    max_rating = 5

    # threshold increment
    threshold_increment = 1

    # max number of chargepoints that can be fed through to algorithm 2
    max_chargepoints = 240

    # minimum number of chargepoint that can be fed through to algorithm 2
    min_chargepoints = 180

    # number of chargers parameter when start being stricter with rating to filter
    number_charge_points_parameter = 5

    # ellipse factor
    # factor to squash ellipse
    ellipse_factor = 5

    # define functions ######################################


    def get_ellipse(start_lat_long,end_lat_long):

        # get equation of ellipse between those points where they are either end of longer radius
        # so need to find
        #   centre of ellipse (in terms of lat long)
        #   longer radius (in terms of lat long)
        #   shorter radius (in terms of lat long)
        #   angle A through which ellipse is rotated from diameter aligning with x axis (longitude)
        #   tan(A) = latitude difference/ longitude difference
        #   A = arctan(latitude difference/ longitude difference)


        # find centre of ellipse
        centre_lat = (start_lat_long['latitude'].iloc[0] + end_lat_long['latitude'].iloc[0]) / 2
        centre_long = (start_lat_long['longitude'].iloc[0] + end_lat_long['longitude'].iloc[0]) / 2

        # find radius = 0.5 * distance between points
        # use pythagoras
        lat_diff = abs(start_lat_long['latitude'].iloc[0] - end_lat_long['latitude'].iloc[0])
        long_diff = abs(start_lat_long['longitude'].iloc[0] - end_lat_long['longitude'].iloc[0])
        longer_radius = ellipse_factor * math.sqrt((long_diff) ** 2 + (lat_diff) ** 2)

        # work out latitude and longitude differences without absolute
        lat_diff_non_abs = end_lat_long['latitude'].iloc[0] - start_lat_long['latitude'].iloc[0]
        long_diff_non_abs = end_lat_long['longitude'].iloc[0] - start_lat_long['longitude'].iloc[0]

        # try setting smaller radius as half of longer radius
        shorter_radius = 0.5 * longer_radius

        angle = np.arctan(lat_diff_non_abs/long_diff_non_abs)

        return centre_lat,centre_long,longer_radius,shorter_radius,angle




    def filter_chargepoints(number_charge_points_threshold,rating_threshold,ncr_filtered) :

        ncr_filtered = ncr_filtered.loc[ncr_filtered['chargers_count'] >= number_charge_points_threshold]

        ncr_filtered = ncr_filtered.loc[ncr_filtered['rating'] >= rating_threshold]

        return ncr_filtered

    # clean data ##########################################


    # only keep variables we need
    ncr_data = ncr_data[['postcode','latitude','longitude','chargers_count','rating','name','FastestConnector_kW']]


    # convert post codes to latitude and longitude
    start_postcode_coordinates = postcode_lookup.loc[postcode_lookup['postcode'] == start_postcode]
    start_lat_long = start_postcode_coordinates[['latitude','longitude']]
    end_postcode_coordinates = postcode_lookup.loc[postcode_lookup['postcode'] == end_postcode]
    end_lat_long = end_postcode_coordinates[['latitude','longitude']]


    # find equation of ellipse #####################################
    # using function

    ellipse_centre_lat, ellipse_centre_long, longer_radius, shorter_radius, angle = get_ellipse(start_lat_long,end_lat_long)


    # to test if a coordinate is within an ellipse must satisfy
    # ((long- centre_long)cos(angle) + (lat-centre_lat)sin(angle))^2/longer_radius^2 +
    # ((long- centre_long)sin(angle) + (lat - centre_lat)cos(angle))^2/shorter_radius^2
    # < 1

    # now filter to keep only chargepoint which satisify ellipse equation
    # break up equation
    term_1 = ((ncr_data['longitude']-ellipse_centre_long)*np.cos(angle) + (ncr_data['latitude']-ellipse_centre_lat)*np.sin(angle))**2

    term_2 = ((ncr_data['longitude']-ellipse_centre_long)*np.sin(angle) - (ncr_data['latitude']-ellipse_centre_lat)*np.cos(angle))**2

    ncr_in_ellipse = ncr_data.loc[term_1/(longer_radius**2) + term_2/(shorter_radius**2) < 1]


    # filter chargepoints until there is only 100 left ##########################################

    # initate data frame

    ncr_filtered = ncr_in_ellipse

    while len(ncr_filtered) >= max_chargepoints:
        ncr_filtered = (filter_chargepoints(number_charge_points_threshold, rating_threshold, ncr_in_ellipse))

        # add condition which steps back if filtering is too much
        while (len(ncr_filtered) < min_chargepoints):
            threshold_increment = threshold_increment / 2
            rating_threshold -= threshold_increment
            ncr_filtered = (filter_chargepoints(number_charge_points_threshold, rating_threshold, ncr_in_ellipse))
            # print(number_charge_points_threshold, rating_threshold, len(ncr_filtered))
            if (len(ncr_filtered) >= max_chargepoints):
                threshold_increment = threshold_increment / 2
                rating_threshold += threshold_increment
                ncr_filtered = (filter_chargepoints(number_charge_points_threshold, rating_threshold, ncr_in_ellipse))
                # print(number_charge_points_threshold, rating_threshold, len(ncr_filtered))

        # if(rating_threshold.is_integer()):number_charge_points_threshold += 1
        if (rating_threshold < max_rating): rating_threshold += threshold_increment
        # print(number_charge_points_threshold, rating_threshold, len(ncr_filtered))


    # return data with name, lat,long, faster connector stamp and rating

    ncr_filtered = ncr_filtered[['latitude','longitude','rating','name','FastestConnector_kW']]

    return ncr_filtered

"""
# old version

def algorithm_1_function(start_postcode,end_postcode,postcode_lookup,ncr_data,comfort_rating):


    # Set variables ######################################

    # number of chargers filter
    # must have atleast this number of chargepoints
    # this is a start value which may increase to reduce chargepoint options
    number_charge_points_threshold = 1

    # threshold for chargepoint quality "rating"
    # this is a start value which may increase to reduce chargepoint options
    # comfort rating is given as a number between 0 and 1 so mulitple by 5
    rating_threshold = comfort_rating

    # max rating possible
    max_rating = 5

    # max number of chargepoints that can be fed through to algorithm 2
    max_chargepoints = 100

    # number of chargers parameter when start being stricter with rating to filter
    number_charge_points_parameter = 5

    # ellipse factor
    # factor to squash ellipse
    ellipse_factor = 0.5

    # define functions ######################################


    def get_ellipse(start_lat_long,end_lat_long):

        # get equation of ellipse between those points where they are either end of longer radius
        # so need to find
        #   centre of ellipse (in terms of lat long)
        #   longer radius (in terms of lat long)
        #   shorter radius (in terms of lat long)
        #   angle A through which ellipse is rotated from diameter aligning with x axis (longitude)
        #   tan(A) = latitude difference/ longitude difference
        #   A = arctan(latitude difference/ longitude difference)


        # find centre of ellipse
        centre_lat = (start_lat_long['latitude'].iloc[0] + end_lat_long['latitude'].iloc[0]) / 2
        centre_long = (start_lat_long['longitude'].iloc[0] + end_lat_long['longitude'].iloc[0]) / 2

        # find radius = 0.5 * distance between points
        # use pythagoras
        lat_diff = abs(start_lat_long['latitude'].iloc[0] - end_lat_long['latitude'].iloc[0])
        long_diff = abs(start_lat_long['longitude'].iloc[0] - end_lat_long['longitude'].iloc[0])
        longer_radius = ellipse_factor * math.sqrt((long_diff) ** 2 + (lat_diff) ** 2)

        # work out latitude and longitude differences without absolute
        lat_diff_non_abs = end_lat_long['latitude'].iloc[0] - start_lat_long['latitude'].iloc[0]
        long_diff_non_abs = end_lat_long['longitude'].iloc[0] - start_lat_long['longitude'].iloc[0]

        # try setting smaller radius as half of longer radius
        shorter_radius = 0.5 * longer_radius

        angle = np.arctan(lat_diff_non_abs/long_diff_non_abs)

        return centre_lat,centre_long,longer_radius,shorter_radius,angle




    def filter_chargepoints(number_charge_points_threshold,rating_threshold,ncr_filtered) :

        ncr_filtered = ncr_filtered.loc[ncr_filtered['chargers_count'] >= number_charge_points_threshold]

        ncr_filtered = ncr_filtered.loc[ncr_filtered['rating'] >= rating_threshold]

        return ncr_filtered

    # clean data ##########################################


    # only keep variables we need
    ncr_data = ncr_data[['postcode','latitude','longitude','chargers_count','rating','name','FastestConnector_kW']]


    # convert post codes to latitude and longitude
    start_postcode_coordinates = postcode_lookup.loc[postcode_lookup['postcode'] == start_postcode]
    start_lat_long = start_postcode_coordinates[['latitude','longitude']]
    end_postcode_coordinates = postcode_lookup.loc[postcode_lookup['postcode'] == end_postcode]
    end_lat_long = end_postcode_coordinates[['latitude','longitude']]


    # find equation of ellipse #####################################
    # using function

    ellipse_centre_lat, ellipse_centre_long, longer_radius, shorter_radius, angle = get_ellipse(start_lat_long,end_lat_long)


    # to test if a coordinate is within an ellipse must satisfy
    # ((long- centre_long)cos(angle) + (lat-centre_lat)sin(angle))^2/longer_radius^2 +
    # ((long- centre_long)sin(angle) + (lat - centre_lat)cos(angle))^2/shorter_radius^2
    # < 1

    # now filter to keep only chargepoint which satisify ellipse equation
    # break up equation
    term_1 = ((ncr_data['longitude']-ellipse_centre_long)*np.cos(angle) + (ncr_data['latitude']-ellipse_centre_lat)*np.sin(angle))**2

    term_2 = ((ncr_data['longitude']-ellipse_centre_long)*np.sin(angle) - (ncr_data['latitude']-ellipse_centre_lat)*np.cos(angle))**2

    ncr_in_ellipse = ncr_data.loc[term_1/(longer_radius**2) + term_2/(shorter_radius**2) < 1]


    # filter chargepoints until there is only 100 left ##########################################

    # initate data frame

    ncr_filtered = ncr_in_ellipse

    while len(ncr_filtered)>=max_chargepoints:
        ncr_filtered = (filter_chargepoints(number_charge_points_threshold,rating_threshold,ncr_filtered))
        number_charge_points_threshold +=1
        if(rating_threshold<max_rating & number_charge_points_threshold > number_charge_points_parameter): rating_threshold +=1
        # print(number_charge_points_threshold,rating_threshold,len(ncr_filtered))


    # return data with name, lat,long, faster connector stamp and rating

    ncr_filtered = ncr_filtered[['latitude','longitude','rating','name','FastestConnector_kW']]

    return ncr_filtered
    """