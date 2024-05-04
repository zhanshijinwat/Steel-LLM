import openxlab
# go https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0 to apply ak and sk
openxlab.login(ak="***", sk="***")
from openxlab.dataset import download
download(dataset_repo='OpenDataLab/WanJuan1_dot_0',source_path='/raw/nlp', target_path='/data1/wanjuan')
