{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "da318f53-a095-466a-8834-cced81a034c9",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import requests\n",
    "\n",
    "def get_Lastest_Tensor(db_url = 'http://192.168.0.126:8080/flask?asset_id=' , asset_id = '55285839-9b78-48d8-9f4e-573190ace016'):\n",
    "    \"\"\" REST API를 사용하여 DB에서 가장 마지막에 저장된 spectrum 데이터를 요청(GET)하고 stack된 tensor를 반환\n",
    "    db_url : URL 주소\n",
    "    asset_id : asset_id\n",
    "    \n",
    "    \"\"\"\n",
    "    url = db_url+asset_id\n",
    "    response = requests.get(url)\n",
    "    \n",
    "    if response.status_code == 200:\n",
    "        data = response.json()\n",
    "        return torch.tensor(data)\n",
    "    else:\n",
    "        print(\"Request failed with status code:\", response.status_code)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "e3fa5891-48da-424f-a5e8-1139a3812cdd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "http://192.168.0.126:8080/flask?asset_id=55285839-9b78-48d8-9f4e-573190ace016\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "tensor([1.505291293142363429069519042969e-04, 1.496677723480388522148132324219e-04,\n",
       "        2.063751308014616370201110839844e-04,  ...,\n",
       "        3.585517697501927614212036132812e-04, 2.573264646343886852264404296875e-04,\n",
       "        1.282929442822933197021484375000e-04])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "get_Lastest_Tensor()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "782c6058-6e48-43ca-94ff-f3e5100cba6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "list"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cf9e27c4-66a0-43bc-a9ce-d9551e5bbefe",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "6f32c688-e325-476f-8e9c-1ee97c322231",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_data = torch.tensor(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f524070f-39f4-482c-8e1c-7f2216587b73",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([1.505291293142363429069519042969e-04, 1.496677723480388522148132324219e-04,\n",
       "        2.063751308014616370201110839844e-04,  ...,\n",
       "        3.585517697501927614212036132812e-04, 2.573264646343886852264404296875e-04,\n",
       "        1.282929442822933197021484375000e-04])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "158cfb21-1930-411a-bd22-33c15b361280",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "cat(): argument 'tensors' (position 1) must be tuple of Tensors, not Tensor",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[32], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcat\u001b[49m\u001b[43m(\u001b[49m\u001b[43mx_data\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[1;31mTypeError\u001b[0m: cat(): argument 'tensors' (position 1) must be tuple of Tensors, not Tensor"
     ]
    }
   ],
   "source": [
    "torch.cat(x_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "756a0fd1-695d-401b-961e-311b3cc1b28c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
