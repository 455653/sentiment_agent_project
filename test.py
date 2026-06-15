import torch
import jieba
import pandas as pd
import crewai
import chainlit

print("PyTorch版本:", torch.__version__)
print("jieba分词测试:", list(jieba.cut("这个商品质量很好")))
print("环境配置成功！")