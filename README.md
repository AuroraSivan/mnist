mnist手写集识别

1、运行安装库
pip install torch==2.2.0 torchvision==0.17.0 matplotlib==3.8.0 numpy==1.24.0

model.py	构建神经网络结构
train.py	加载数据，训练模型
predict.py	使用训练好的模型进行预测
utils.py	数据加载与模型保存加载
config.py	超参数统一管理
main.py	    项目统一入口（调用训练+推理）