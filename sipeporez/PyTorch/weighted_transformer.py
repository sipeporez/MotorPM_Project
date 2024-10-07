import numpy as np 

def standardized(data, maxval):
    """ Standardized data starting from 1 to max_val

    parameters: 
    data(list) : a list of floats to be standardized
    maxval: maximum value of data to become

    return: 
    standardized(numpy array) : standardized from 1 to max_val    
    """
    
    data = np.sort(np.array(data))

    minval = np.min(data)
    maxval_current = np.max(data)
    normalized = (data - minval) / (maxval_current - minval)

    standardized = 1 + (normalized * (maxval - 1))
    return  standardized

def _find_starting_point(data, val):
    """Find the first value in data that is greater than sect_point."""
    for value in data:
        if value > val:
            return value
    return None

def weighted_function(data, const=.25): 
    """ Making a weighted function (similar to ReLU) customised to the parmeters. 

    parameters: 
    data(numpy array) : Input data to be transformed.
    const(float) : Range from 0 to 1. First `const` portion of the data will become 1. For the rest of the data, the values will scale linearly.

    return: Transformed data
    """

    if not (0 <= const <= 1):
        raise ValueError("const must be between 0 and 1.")
    
    data = np.sort(data)
    data_maxval = np.max(data) 
    sect_point_index = int(len(data) * const)
    weighted_data = np.zeros_like(data) 
    x_range = data_maxval - data[sect_point_index] ; 
    y_range = data_maxval - 1 ; 
    rest_slope = y_range / x_range if x_range > 0 else 0
    intercept = 1- data[sect_point_index] * rest_slope    

    for i, val in enumerate(data): 
        if i <= sect_point_index: 
            weighted_data[i] = 1; 
        else:           
            weighted_data[i] = (val) * rest_slope + intercept
      
    return weighted_data   