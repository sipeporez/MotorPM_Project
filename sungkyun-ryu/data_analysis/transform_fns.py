# # This was for list
# import numpy as np 

# def standardized(data, maxval):
#     """ Standardized data starting from 1 to max_val

#     parameters: 
#     data(list) : a list of floats to be standardized
#     maxval: maximum value of data to become

#     return: 
#     standardized(numpy array) : standardized from 1 to max_val    
#     """
    
#     data = np.sort(np.array(data))

#     minval = np.min(data)
#     maxval_current = np.max(data)
#     normalized = (data - minval) / (maxval_current - minval)

#     standardized = 1 + (normalized * (maxval - 1))
#     return  standardized

# def _find_starting_point(data, val):
#     """Find the first value in data that is greater than sect_point."""
#     for value in data:
#         if value > val:
#             return value
#     return None

# def weighted_function(data, rate=.25): 
#     """ Making a weighted function (similar to ReLU) customised to the parmeters. 

#     parameters: 
#     data(numpy array) : Input data to be transformed.
#     const(float) : Range from 0 to 1. First `rate` portion of the data will become 1. For the rest of the data, the values will scale linearly.

#     return: Transformed data
#     """

#     if not (0 <= rate <= 1):
#         raise ValueError("const must be between 0 and 1.")
    
#     data = np.sort(data)
#     data_maxval = np.max(data) 
#     sect_point_index = int(len(data) * rate)
#     weighted_data = np.zeros_like(data) 
#     x_range = data_maxval - data[sect_point_index] ; 
#     y_range = data_maxval - 1 ; 
#     rest_slope = y_range / x_range if x_range > 0 else 0
#     intercept = 1- data[sect_point_index] * rest_slope    

#     for i, val in enumerate(data): 
#         if i <= sect_point_index: 
#             weighted_data[i] = 1; 
#         else:           
#             weighted_data[i] = (val) * rest_slope + intercept
      
#     return weighted_data   

#This was for torch
import torch

def standardized(data, maxval):
    """ Standardized data starting from 1 to max_val 
    
    parameters: 
     data(list) : a list of floats to be standardized
     maxval: maximum value of data to become
    
     return: 
     standardized(torch) : standardized from 1 to max_val   
    
    """

    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a PyTorch tensor.")

    data = torch.sort(data).values

    minval = torch.min(data)
    maxval_current = torch.max(data)
    normalized = (data - minval) / (maxval_current - minval)

    standardized = 1 + (normalized * (maxval - 1))
    return standardized

def weighted_function(data, rate=0.25): 
    """ Making a weighted function (similar to ReLU) customized to the parameters. """

    if not (0 <= rate <= 1):
        raise ValueError("rate must be between 0 and 1.")

    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a PyTorch tensor.")
    
    data = torch.sort(data).values
    data_maxval = torch.max(data) 
    sect_point_index = int(len(data) * rate)
    weighted_data = torch.zeros_like(data) 
    x_range = data_maxval - data[sect_point_index] 
    y_range = data_maxval - 1 
    rest_slope = y_range / x_range if x_range > 0 else 0
    intercept = 1 - data[sect_point_index] * rest_slope    

    for i, val in enumerate(data): 
        if i <= sect_point_index: 
            weighted_data[i] = 1 
        else:           
            weighted_data[i] = (val) * rest_slope + intercept
      
    return weighted_data

def data_weighted(spec_freq, df_normal, maxval, rate, repeat_times=36):
    """
    Process the spectral frequency data.

    Parameters:
        spec_freq (torch.Tensor): The input spectral frequency tensor.
        df_normal (torch.Tensor): The normal DataFrame tensor.
        maxval (float): The maximum value for standardization.
        rate (float): The rate for the weighted function.
        repeat_times (int): Number of times to repeat the weighted array.

    Returns:
        torch.Tensor: The processed result after applying the operations.
    """
    # Standardize the frequency
    standardized_freq = standardized(spec_freq, maxval)

    # Apply the weighted function
    weighted_arr = weighted_function(standardized_freq, rate)

    # Expand the array
    expanded_arr = weighted_arr.repeat(repeat_times)

    # Perform element-wise multiplication
    result = df_normal * expanded_arr

    return result

def get_sorted_top_k_values(tensor, k=5000):
    """
    Extract the top k values from each row of a 2D tensor
    and sort them based on their original indices.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (num_rows, num_columns).
        k (int): The number of top values to extract from each row.
        
    Returns:
        torch.Tensor: A tensor of shape (num_rows, k) containing the sorted top k values.
    """
    top_values, top_indices = torch.topk(tensor, k, sorted=False)

    # Sort the top values according to the indices
    sorted_indices = torch.argsort(top_indices, dim=1)
    sorted_top_values = torch.gather(top_values, 1, sorted_indices)
    
    return sorted_top_values