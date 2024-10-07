import requests
import torch

def get_Lastest_Tensor(db_url = 'http://192.168.0.126:8080/flask?asset_id=' , asset_id = '55285839-9b78-48d8-9f4e-573190ace016'):
    """ REST API를 사용하여 DB에서 가장 마지막에 저장된 spectrum 데이터를 요청(GET)하고 stack된 tensor를 반환
    db_url : URL 주소
    asset_id : asset_id
    
    """
    url = db_url+asset_id
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return torch.tensor(data)
    else:
        print("Request failed with status code:", response.status_code)