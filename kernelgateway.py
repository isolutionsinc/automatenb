import json
import requests
import numpy as np

req = json.loads(REQUEST)
res = dict(data=np.random.randn(5, 4).tolist(), request=req)
print(json.dumps(res))