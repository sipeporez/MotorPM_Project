import torch
import pickle
import pandas as pd 
import seaborn as sns
from flask import Flask, request, jsonify

app = Flask(__name__)

# model = torch.load('1003)stacked_model(0.68).pth')
# model.eval()  # 평가 모드로 전환 (학습 시 필요하지 않은 동작 비활성화)

def weight_function(wf_broadcast = 100000):
    wf_broadcast = wf_broadcast
    file_path='data/weight/5520_spectrum_x_weights.pkl'
    with open(file_path, 'rb') as f: 
        data = pickle.load(f)
    wf_x = torch.tensor(data)
    file_path='data/weight/5520_spectrum_y_weights.pkl'
    with open(file_path, 'rb') as f: 
        data = pickle.load(f)
    wf_y = torch.tensor(data)
    file_path='data/weight/5520_spectrum_z_weights.pkl'
    with open(file_path, 'rb') as f: 
        data = pickle.load(f)
    wf_z = torch.tensor(data)
    wf_xyz = torch.stack((wf_x, wf_y, wf_z))
    wf_xyz_mul = wf_xyz.repeat(12, 1) * wf_broadcast
    return wf_xyz_mul

def json_to_vstack(data):
    cols = ["x1", "y1", "z1",
    "x2", "y2", "z2",
    "x3", "y3", "z3",
    "x4", "y4", "z4",
    "x5", "y5", "z5",
    "x6", "y6", "z6",
    "x7", "y7", "z7",
    "x8", "y8", "z8",
    "x9", "y9", "z9",
    "x10", "y10", "z10",
    "x11", "y11", "z11",
    "x12", "y12", "z12"]

    for i in cols:
        data[i] = data[i][1:]
    
    # for col in cols:
    #     print(f"{col}: {len(data[col])}")

    data_list = [data[i] for i in cols]
    tensor = torch.tensor(data_list).reshape(1, 1, 36, 2048)

    wf_xyz_mul = weight_function()

    tensor_mul = tensor *  wf_xyz_mul
    return tensor_mul


def model_eval(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)
    model = torch.load('1005)stacked_model(dev-0.9705).pth', map_location=torch.device(device))

    model.eval()
    with torch.no_grad():
        outputs = model(data)  # 레이블 없는 텐서를 device로 이동

        # 이진 예측 (threshold는 front에서 설정)
        predict = (torch.sigmoid(outputs)).float()

    return predict.item()


@app.route('/pdm', methods=['POST'])
def submit():
    # JSON 데이터 받기
    data = request.json
    tensor = json_to_vstack(data)
    
    # 응답 생성
    response = model_eval(tensor)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
