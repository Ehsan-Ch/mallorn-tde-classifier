"""Small invented light curves for software checks ONLY; no astronomical validity."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def make_fixture(root: Path, train_count=72, test_count=12, seed=917):
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    logs={"train":[],"test":[]}
    curves={"train":[],"test":[]}
    for mode,count in (("train",train_count),("test",test_count)):
        for i in range(count):
            label=int(i%4==0)
            object_id=f"synthetic_{mode}_{i:04d}"
            split=f"split_{i%2+1:02d}"
            row={"object_id":object_id,"Z":float(rng.uniform(.05,.7)),"Z_err":np.nan if mode=="train" else .1,
                 "EBV":float(rng.uniform(0,.12)),"split":split,"English Translation":"invented fixture"}
            if mode=="train": row.update(target=label,SpecType="TDE" if label else "AGN")
            logs[mode].append(row)
            peak=rng.uniform(59020,59040)
            for band_no,band in enumerate("ugrizy"):
                if i%9==0 and band=="u": continue
                time=np.sort(rng.uniform(58950,59200,18))
                shape=(15*np.exp(-((time-peak)/(45 if label else 15))**2)
                       if i%3 else 5+5*np.sin((time-peak)/22))
                error=rng.uniform(.6,1.5,len(time))
                flux=shape*(1+band_no*(.12 if label else .35))+rng.normal(0,error)
                curves[mode].extend({"object_id":object_id,"Time (MJD)":float(t),"Flux":float(f),
                                      "Flux_err":float(e),"Filter":band,"split":split}
                                     for t,f,e in zip(time,flux,error))
        pd.DataFrame(logs[mode]).to_csv(root/f"{mode}_log.csv",index=False)
        frame=pd.DataFrame(curves[mode])
        for split,g in frame.groupby("split"):
            folder=root/split
            folder.mkdir(exist_ok=True)
            g.drop(columns="split").to_csv(folder/f"{mode}_full_lightcurves.csv",index=False)
    pd.DataFrame({"object_id":[r["object_id"] for r in reversed(logs["test"])],"prediction":0}).to_csv(root/"sample_submission.csv",index=False)
    (root/"SYNTHETIC_FIXTURE.json").write_text(json.dumps({"synthetic":True,"purpose":"Software tests; no competition evidence"}))
