# 默认已经在 env_setup 路径下
pip install -r requirements-bitahub-include-vww.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
rm -rf edgeml.egg-info
pip install -e .
echo `pwd` > /usr/local/lib/python3.8/dist-packages/edgeml.egg-link
echo `pwd` > /usr/local/lib/python3.8/dist-packages/easy-install.pth
pip install -e edgeml_pytorch/cuda/
echo "$(pwd)/edgeml_pytorch/cuda" > /usr/local/lib/python3.8/dist-packages/fastgrnn-cuda.egg-link
echo "$(pwd)/edgeml_pytorch/cuda" >> /usr/local/lib/python3.8/dist-packages/easy-install.pth
