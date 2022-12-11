import fire
import webdataset as wds
from config import cache_path, output_path, target_path, input_path, benchmark, keep_cols, delete_batch_scripts_after_download
import os
import subprocess
import webdataset as wds
import pandas as pd
from torch.utils.data import DataLoader
from itertools import islice
import numpy as np
from io import BytesIO
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, CenterCrop
from torchvision import models
import io
from PIL import Image
import torch
from time import time
from tqdm import tqdm

from inference_models.inference_mobilenetv3 import inference_on_batch_col
from inference_models.inference_clip_h import preprocopenclip224, inference_on_batch_col

def decodebyte(x):
    return Image.open(io.BytesIO(x)).convert("RGB")

transform = Compose([
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
])

def worker(current_shard):
    try:
        bs = 256
        dataset_url = 'pipe:aws s3 cp ' + input_path + f'{current_shard:06d}.tar -'
        ##############################################################
        ####### Here you need to specify the processing params #######
        ##############################################################

        ### MobilenetV3 example
        # ds = (
        #     wds
        #     .WebDataset(dataset_url, handler=wds.warn_and_continue)
        #     .map_dict(jpg=decodebyte).map_dict(jpg=transform)
        # )
        
        ### Clip H example
        ds = wds.WebDataset(dataset_url, handler=wds.ignore_and_continue).map_dict(jpg = preprocopenclip224)

        ##############################################################
        ##############################################################
        ###########################################################

        dl = DataLoader(ds, num_workers=2, batch_size=bs, pin_memory=True, shuffle=False)

        if benchmark == True:
            start_time = time()

        all_data = {}
        for key in keep_cols:
            all_data[key] = []

        inference_data = []
        image_embeddings = []
        
        for b in tqdm(dl, total=int(9540/bs) + 1):
            ###########################################################
            ####### Here you need to plug in your desired model #######
            ###########################################################

            ### MobilenetV3 example
            # inference_data.extend(inference_on_batch_col(b['jpg']))

            ### Clip H example
            scores, image_features = inference_on_batch_col(b['jpg'])
            inference_data.extend(scores)
            image_embeddings.extend(image_features)

            ###########################################################
            ###########################################################
            ###########################################################

            for key in keep_cols:
                all_data[key].extend(b[key])

        df = pd.DataFrame(all_data)
        df.to_parquet(f'{target_path}/{current_shard:06d}.parquet')

        with open(f'{output_path}/{current_shard:06d}_scores.npy', "wb") as f:
            npb = BytesIO()
            np.save(npb, np.asanyarray(inference_data))
            f.write(npb.getbuffer())

        with open(f'{output_path}/{current_shard:06d}_image_emb.npy', "wb") as f:
            npb = BytesIO()
            np.save(npb, np.asanyarray(image_embeddings))
            f.write(npb.getbuffer())


        if benchmark == True:
            print("###########################################################################")
            print(f"Processing one shard with {len(inference_data)} samples took {time() - start_time:.2f}s.")
            print(f"Estimated time for 2B rows: {((time() - start_time)/len(inference_data)*2000000000)/(60*60*24):.2f} days.")
            print("###########################################################################")
        
        subprocess.run(["aws", "s3" , "cp", f'{output_path}/{current_shard:06d}_scores.npy', f"{target_path}/{current_shard:06d}_scores.npy"])
        subprocess.run(["aws", "s3" , "cp", f'{output_path}/{current_shard:06d}_image_emb.npy', f"{target_path}/{current_shard:06d}_image_emb.npy"])

        try:
            os.remove(f'{output_path}/{current_shard:06d}_scores.npy')
            os.remove(f'{output_path}/{current_shard:06d}_image_emb.npy')
        except:
            print(e)

    except Exception as e:
        print(e)
        
        try:
            os.rename(cache_path + f"/sbatch_script_{current_shard:06d}.sh", cache_path + f"/sbatch_script_{current_shard:06d}_failed.sh")
        except:
            print(e)
        
        
        try:
            with open("error_logs.txt", "a") as f:
                f.write(f"Shard number {current_shard:06d}: " + str(e) + '\n')
        except Exception as e:
            print('Could not write error logs...')
            print(e)
    else:
        if delete_batch_scripts_after_download:
            fp = f"/sbatch_script_{current_shard:06d}.sh"
            if os.path.isfile(fp):
                os.remove(cache_path + fp)

            fp = f"/sbatch_script_{current_shard:06d}_failed.sh"
            if os.path.isfile(fp):
                os.remove(cache_path + fp)
    

if __name__ == "__main__":
    fire.Fire(worker)