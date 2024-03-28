import openxlab
openxlab.login(ak="xxx", sk="xxx")
from openxlab.dataset import download
download(dataset_repo='OpenDataLab/WanJuan1_dot_0',source_path='/raw/nlp', target_path='/data/wanjuan')