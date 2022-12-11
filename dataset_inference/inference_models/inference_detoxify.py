from detoxify import Detoxify

# each model takes in either a string or a list of strings

results = Detoxify('unbiased').predict('example text')

# import numpy as np
# from detoxify import Detoxify
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# detoxify = Detoxify('original', device=device)

# mylist = ["This is cool", "this is even cooler"]

# def run_detox_inference(b_col):
#     detox_result = detoxify.predict(b_col)
#     print(detox_result)
#     # detox_values = np.asanyarray(detox_result.values())
#     # print(detox_values)
#     # detox = list(map(list, zip(*detox)))

# run_detox_inference(mylist)