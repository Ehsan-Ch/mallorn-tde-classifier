"""Build per-object physical features without reading target labels into extraction."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mallorn.data import BANDS,load_dataset,sha256
from mallorn.physics_features import object_features


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['train','test'],default='train')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    destination=root/'artifacts/improvement';destination.mkdir(parents=True,exist_ok=True)
    train,test,photo,manifest=load_dataset(root/'data/mallorn')
    log=train if args.mode=='train' else test
    meta=log.set_index('object_id')[['Z','EBV']]
    photo=photo[photo.object_id.isin(meta.index)]
    bandmap={b:i for i,b in enumerate(BANDS)}
    tasks=[]
    for oid,g in photo.groupby('object_id',sort=True):
        row=meta.loc[oid]
        tasks.append((oid,float(row.Z),float(row.EBV),g['Time (MJD)'].to_numpy(),
                      g.Flux.to_numpy(),g.Flux_err.to_numpy(),g.Filter.map(bandmap).to_numpy()))
    rows={};start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,(oid,features) in enumerate(pool.map(object_features,tasks,chunksize=10),1):
            rows[oid]=features
            if n%100==0: print(args.mode,n,len(tasks),'seconds',round(time.monotonic()-start,1),flush=True)
    new=pd.DataFrame.from_dict(rows,orient='index').sort_index(axis=1).rename_axis('object_id')
    new=new.replace([np.inf,-np.inf],np.nan)
    base=pd.read_csv(root/'artifacts/official_001/features.csv',index_col='object_id').loc[new.index]
    for name,frame in [('physics',new),('enhanced',base.join(new))]:
        frame.to_csv(destination/f'{name}_{args.mode}.csv',float_format='%.15g')
    (destination/f'physics_{args.mode}_manifest.json').write_text(json.dumps({
        'objects':len(new),'new_features':len(new.columns),'source_sha256':sha256(root/'src/mallorn/physics_features.py'),
        'features_sha256':sha256(destination/f'enhanced_{args.mode}.csv'),
        'elapsed_seconds':time.monotonic()-start,'label_free':True,
        'approximations':'Matern32 time/log-wavelength GP; fixed wavelength scale; max160 observations; monochromatic blackbody; flat LCDM H0=70 Om=.3'},indent=2)+'\n')
    print('DONE',new.shape,flush=True)
