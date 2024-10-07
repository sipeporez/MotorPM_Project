import pandas as pd
import torch
from itertools import chain

def PostProcessing(file_name, cols=['spectrum_x_amp', 'spectrum_y_amp', 'spectrum_z_amp']):
    """  Index=True 로 저장된 csv 파일을 전처리하여 stack 된 tensor 반환

    file_name : 파일 경로
    cols : 컬럼 리스트, 기본값) ['spectrum_x_amp', 'spectrum_y_amp', 'spectrum_z_amp']
    """
    # csv 파일 로드
    df = pd.read_csv(file_name, header=[0], index_col=[0, 1])
    
    # 각 행의 리스트를 병합, 첫번째 데이터는 버림
    for col in cols:
        df[col] = df[col].apply(lambda x: [float(j) for j in x.replace('[', '').replace(']', '').split(',')][1:])
    
    # 각 컬럼의 데이터를 병합
    new_df = df[cols].apply(lambda x: list(chain.from_iterable(x)), axis=1)

    # 그룹별로 데이터를 모아서 텐서로 변환하고 스택
    tensors = [torch.tensor(item) for item in new_df.groupby(level=0).apply(lambda x: [item for sublist in x for item in sublist])]
    
    # 텐서들을 스택
    result = torch.stack(tensors)

    return result
