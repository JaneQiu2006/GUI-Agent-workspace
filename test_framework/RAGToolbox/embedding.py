import numpy as np
from .modeling_bert import JinaBertModel
from numpy.linalg import norm
from typing import List

class Jinaembedding:
    def __init__(self, path: str):
        if not path:
            raise ValueError("必须通过 config.embedding_model 提供 embedding 模型目录")
        self.path = path
        self.embedding_model=JinaBertModel.from_pretrained(path)
    
    def get_embedding(self,content:str=''):
        return self.embedding_model.encode([content])[0]
    
    def compare(self, text1: str, text2: str):
        
        cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
        embeddings = self.embedding_model.encode([text1, text2])
        return cos_sim(embeddings[0], embeddings[1])

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
