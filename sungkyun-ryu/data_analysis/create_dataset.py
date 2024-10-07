import torch
from sklearn.model_selection import train_test_split

def LabelingData(normal_data, error_data, test_size=0.2, random_state=42):
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