import torch
from itertools import chain
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import time

def torch_standardized(data, maxval):
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

def bearing_weighted_function(data, rate=0.25): 
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

def csv_to_tensor_stack(file_name, cols=['spectrum_x_amp', 'spectrum_y_amp', 'spectrum_z_amp']):
    """  Index=True 로 저장된 csv 파일을 전처리하여 stack 된 tensor 반환

    file_name : 파일 경로
    cols : 컬럼 리스트, 기본값) ['spectrum_x_amp', 'spectrum_y_amp', 'spectrum_z_amp']
    """
    # csv 파일 로드
    df = pd.read_csv(file_name, header=[0], index_col=[0, 1])
    
    # 각 행의 리스트를 병합, 첫번째 데이터는 버림
    for col in cols:
        df[col] = df[col].apply(lambda x: [float(j) for j in x.split(',')][1:])
    
    # 각 컬럼의 데이터를 병합
    new_df = df[cols].apply(lambda x: list(chain.from_iterable(x)), axis=1)

    # 그룹별로 데이터를 모아서 텐서로 변환하고 스택
    tensors = [torch.tensor(item) for item in new_df.groupby(level=0).apply(lambda x: [item for sublist in x for item in sublist])]
    
    # 텐서들을 스택
    result = torch.stack(tensors)

    return result

def binary_labeling(normal_data, error_data, test_size=0.2, random_state=42):
    ''' 텐서 타입의 정상 데이터와 비정상 데이터를 입력받고\n
    정상데이터는 1, 비정상 데이터는 0 Label을 붙임\n
    sklearn의 train_test_split 모듈을 사용하여 총 4개의 데이터셋을 랜덤으로 섞어주는 함수\n
    input : 
        normal_data = 정상 데이터,
        error_data = 비정상 데이터,
        test_size = 데이터셋 비율 (기본값 0.2 = 20%)
        random_state = 난수 seed (기본값 42)
    return :
        X_train, X_test, y_train, y_test
    '''    
    # 데이터 레이블 생성
    normal_labels = torch.ones(normal_data.size(0), dtype=torch.long)  # 정상 데이터 레이블 (1)
    error_labels = torch.zeros(error_data.size(0), dtype=torch.long)  # 비정상 데이터 레이블 (0)
    
    # 데이터 결합
    combined_data = torch.cat((normal_data, error_data), dim=0)
    combined_labels = torch.cat((normal_labels, error_labels), dim=0)

    X_train, X_test, y_train, y_test = train_test_split(
    combined_data, 
    combined_labels,
    test_size=test_size,  # 기본값 : 20%를 테스트 데이터로 사용
    random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def min_max_standardization(data, minval=1, maxval=2): 
    min_current = min(data)
    max_current = max(data) 

    scaled_data = [minval + (value - min_current) * (maxval - minval) / (max_current - min_current) for value in data]

    return scaled_data

def rpm_approx_weighted_fn(df_spectrum, x_val_min, x_val_max, num_linspace): 
    """ approximating rpm from dominant amplitude on frequency domain
    parameters:
    df_spectrum(pands Series) 

    return:
    """
    df_max_index = pd.Series(df_spectrum.apply(lambda x: x.index(max(x)) if max(x) >= 0.02 else 0))
    df_max_index = df_max_index[df_max_index != 0]
    kde = gaussian_kde(df_max_index)
    kde_values = kde(np.linspace(x_val_min, x_val_max, num_linspace)) 
    kde_values_list = kde_values.tolist()

    return kde_values_list

def get_random_permutation(num_groups):
    seed = int(time.time() * 1000) % 10000 
    torch.manual_seed(seed)  
    return torch.randperm(num_groups)

def multi_datasets(df, multi_dim=2): 
    permuted_tensors_list = []
    num_rows = df.shape[0]
    grouped_tensor = df.view(num_rows, 12, 6144)

    for _ in range(multi_dim): 
        permuted_indices = get_random_permutation(12)
        permuted_tensor = grouped_tensor[:, permuted_indices, :]
        
        permuted_tensors_list.append(permuted_tensor)

    stacked_tensor = torch.cat(permuted_tensors_list, dim=0)
    final_tensor = stacked_tensor.view(-1, 73728)

    return final_tensor