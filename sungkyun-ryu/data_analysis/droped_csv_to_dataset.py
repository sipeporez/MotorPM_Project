import pandas as pd
from itertools import chain
import torch

#'data/5528_droped_data.csv'
def PostProcessing(file_name, model_type):
    if model_type not in ['reg', 'cls']:
        raise ValueError("model_type must be either 'reg' or 'cls'")
        
    df = pd.read_csv(file_name)

    # string 리스트를 float 리스트로 변환
    df['spectrum_x_amp'] = df['spectrum_x_amp'].apply(lambda x: [float(i) for i in x.split(',')])
    df['spectrum_y_amp'] = df['spectrum_y_amp'].apply(lambda x: [float(i) for i in x.split(',')])
    df['spectrum_z_amp'] = df['spectrum_z_amp'].apply(lambda x: [float(i) for i in x.split(',')])

    # 12일씩 묶어서 그룹화
    grouped_df = df.groupby(df.index // 12).apply(lambda x: x.reset_index(drop=True))
    
    # 노말 데이터와 에러 데이터 나누기 -> 1일때만 정상
    normal_df = grouped_df[grouped_df['imbalance_health'] == 1]
    error_df = grouped_df[grouped_df['imbalance_health'] != 1]

    # 사용하지 않는 컬럼 처리
    normal_df = normal_df.drop(columns=['date', 'asset_id','time','misalignment_health', 'looseness_health', 'bearing_health'])
    error_df = error_df.drop(columns=['date','asset_id','time','misalignment_health', 'looseness_health', 'bearing_health'])

    # 에셋 마다 다르므로 변경해야 함, 현재는 5528 에셋 기준
    # train -> 나머지 | dev -> 3~4월 | test -> 7월
    train_normal = pd.concat([normal_df.iloc[:372], normal_df.iloc[780:-408]]) # 나머지
    dev_normal = normal_df.iloc[372:780] # 3~4월
    test_normal = normal_df.iloc[-408:] # 7월

    train_error = pd.concat([error_df.iloc[:60], error_df.iloc[96:-36]]) # 나머지
    dev_error = error_df.iloc[60:96] # 3~4월
    test_error = error_df.iloc[-36:] # 7월

    
    def df_to_tensor (df, cols = ['spectrum_x_amp', 'spectrum_y_amp', 'spectrum_z_amp']):

        # 각 행의 리스트를 병합, 첫번째 데이터는 버림
        for col in cols:
            df[col] = df[col].apply(lambda x: [float(j) for j in x][1:])
        
        # 각 컬럼의 데이터를 병합
        new_df = df[cols].apply(lambda x: list(chain.from_iterable(x)), axis=1)
    
        # 라벨링용 imbalance_health 처리
        imb = df['imbalance_health']
        imb = imb.reset_index(drop=True)
        imb = imb.groupby(imb.index // 12).first()
        
        # 그룹별로 데이터를 모아서 텐서로 변환하고 스택
        tensors = [torch.tensor(item) for item in new_df.groupby(level=0).apply(lambda x: [item for sublist in x for item in sublist])]
        
        # 텐서들을 스택
        result = torch.stack(tensors)
    
        return result, imb

    train_normal, train_im = df_to_tensor(train_normal)
    train_error, train_im = df_to_tensor(train_error)
    
    test_normal, test_im = df_to_tensor(test_normal)
    test_error, test_im = df_to_tensor(test_error)
    
    dev_normal, dev_im = df_to_tensor(dev_normal)
    dev_error, dev_im = df_to_tensor(dev_error)

    def Labeling_data_reg(normal_data, error_data, imb):
        # 데이터 레이블 생성
        normal_labels = torch.ones(normal_data.size(0), dtype=torch.long)  # 정상 데이터 레이블 (1)
        if model_type == 'cls': # 이진 분류인 경우
            error_labels = torch.zeros(error_data.size(0), dtype=torch.long)  # 비정상 데이터 레이블 (0)
        elif model_type == 'reg': # 회귀인 경우
            error_labels = torch.tensor(imb.values, dtype=torch.float32)  # 비정상 데이터 레이블 (imbalance 값)
        
        # 데이터 결합
        combined_data = torch.cat((normal_data, error_data), dim=0)
        combined_labels = torch.cat((normal_labels, error_labels), dim=0)
    
        return combined_data, combined_labels
    
    comb_data_train, comb_labels_train =  Labeling_data_reg(train_normal, train_error, train_im)
    comb_data_test, comb_labels_test =  Labeling_data_reg(test_normal, test_error, test_im)
    comb_data_dev, comb_labels_dev =  Labeling_data_reg(dev_normal, dev_error, dev_im)

    return [[comb_data_train, comb_labels_train],[comb_data_test, comb_labels_test],[comb_data_dev, comb_labels_dev]]