import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
from mallorn.data import BANDS,load_dataset,sha256
from mallorn.morphology_features import object_features

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','test'],default='train');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];dest=root/'artifacts/improvement'
    train,test,photo,_=load_dataset(root/'data/mallorn');log=train if args.mode=='train' else test
    meta=log.set_index('object_id')[['Z','EBV']];photo=photo[photo.object_id.isin(meta.index)]
    tasks=[]
    for oid,g in photo.groupby('object_id',sort=True):
        row=meta.loc[oid]
        tasks.append((oid,float(row.Z),float(row.EBV),g['Time (MJD)'].to_numpy(),g.Flux.to_numpy(),g.Flux_err.to_numpy(),g.Filter.map({b:i for i,b in enumerate(BANDS)}).to_numpy()))
    rows={};start=time.monotonic()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for n,(oid,f) in enumerate(pool.map(object_features,tasks,chunksize=10),1):
            rows[oid]=f
            if n%300==0:print(args.mode,n,len(tasks),'seconds',round(time.monotonic()-start,1),flush=True)
    f=pd.DataFrame.from_dict(rows,orient='index').rename_axis('object_id').sort_index(axis=1).replace([np.inf,-np.inf],np.nan)
    base=pd.read_csv(dest/f'enhanced_{args.mode}.csv',index_col='object_id')
    base.join(f).to_csv(dest/f'complete_{args.mode}.csv',float_format='%.15g')
    (dest/f'morphology_{args.mode}_manifest.json').write_text(json.dumps({'objects':len(f),'new_features':len(f.columns),'elapsed_seconds':time.monotonic()-start,'source_sha256':sha256(root/'src/mallorn/morphology_features.py'),'features_sha256':sha256(dest/f'complete_{args.mode}.csv')},indent=2)+'\n')
    print('DONE',f.shape,flush=True)
