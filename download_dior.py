import openxlab
from openxlab.dataset import info
from openxlab.dataset import get

from openxlab.dataset import download

openxlab.login(ak="nvgjz5rxpl2j3vqx1blr", sk="r63aqyvlg294ppz7dz5al2nwjv1lrbmow50aenbj")

info(dataset_repo='OpenDataLab/DIOR')  # 数据集信息查看

get(dataset_repo='OpenDataLab/DIOR', target_path='./data')  # 数据集下载

download(dataset_repo='OpenDataLab/DIOR', source_path='./data/README.md', target_path='./data')  # 数据集文件下载
