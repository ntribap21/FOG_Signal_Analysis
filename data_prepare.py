"""Batch preparation shared by GUI and CLI. No scaling or training here."""
import argparse
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import pandas as pd
from signal_processing import (Config,read_daphnet,select_runs,preprocess,create_windows,feature_table,channel_names)


def prepare_recording(recording,config):
    selected=select_runs(recording,config)
    processed,fs=preprocess(selected,config)
    windows,counts=create_windows(processed,selected,config,fs)
    table,names=feature_table(recording,windows,config,fs)
    report={'Recording_ID':recording.recording,'Subject_ID':recording.subject,'Status':'ok',
            'Raw_Samples':len(recording.frame),'Raw_Outside':int((recording.frame.Label==0).sum()),
            'Selected_Samples':sum(len(d) for _,d in selected),'Selected_Runs':len(selected),
            'Windows':len(windows),'Non_FOG_Windows':sum(w.label==0 for w in windows),
            'FOG_Windows':sum(w.label==1 for w in windows),**counts}
    return table,names,report


def prepare_files(paths,config,progress=None):
    config.validate();tables=[];reports=[];names=None;seen=set()
    for i,path in enumerate(paths):
        if progress:progress(i,len(paths),Path(path).name)
        try:
            recording=read_daphnet(path)
            if recording.recording in seen:raise ValueError('Duplicate Recording_ID; use one copy per recording.')
            seen.add(recording.recording)
            table,current,report=prepare_recording(recording,config)
            if names is not None and current!=names:raise ValueError('Feature schema mismatch.')
            names=current;tables.append(table);reports.append(report)
        except Exception as exc:
            reports.append({'Recording_ID':Path(path).stem,'Status':'error','Error':str(exc)})
    if progress:progress(len(paths),len(paths),'Complete')
    result=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    if result.empty:
        detail='; '.join(str(r.get('Error','no complete valid windows')) for r in reports)
        raise ValueError('No valid windows produced. '+detail)
    return result,names,pd.DataFrame(reports)


def export_dataset(table,names,config,destination,format='both',report=None):
    if format not in ('csv','npz','both'):raise ValueError('Export format must be csv, npz, both.')
    if table.empty:raise ValueError('No windows to export.')
    config.validate()
    root=Path(destination).expanduser().resolve();root=root.with_suffix('') if root.suffix.lower() in ('.csv','.npz') else root
    root.parent.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(table[names].to_numpy(float)).all():raise ValueError('Nonfinite feature values.')
    arrays={'X':table[names].to_numpy(np.float32),'y':table.Label.to_numpy(np.int8),
            'feature_names':np.asarray(names,dtype=str),
            'subject_id':table.Subject_ID.to_numpy(str),'recording_id':table.Recording_ID.to_numpy(str),
            'segment_id':table.Segment_ID.to_numpy(np.int32),'window_id':table.Window_ID.to_numpy(np.int32),
            'window_start':table.Start_Time.to_numpy(float),'window_end':table.End_Time.to_numpy(float),
            'fog_ratio':table.FOG_Ratio.to_numpy(np.float32),'config_id':table.Config_ID.to_numpy(str),
            'config_json':np.asarray(json.dumps(config.dictionary(),ensure_ascii=False))}
    saved=[]
    # Stage all output bytes before installing final files.
    with tempfile.TemporaryDirectory(dir=root.parent) as temp:
        temp=Path(temp)
        if format in ('csv','both'):
            target=Path(str(root)+'.csv');staged=temp/target.name;table.to_csv(staged,index=False);saved.append((staged,target))
        if format in ('npz','both'):
            target=Path(str(root)+'.npz');staged=temp/target.name;np.savez_compressed(staged,**arrays);saved.append((staged,target))
        target=Path(str(root)+'_config.json');staged=temp/target.name
        staged.write_text(json.dumps(config.dictionary(),indent=2,ensure_ascii=False),encoding='utf-8');saved.append((staged,target))
        if report is not None:
            target=Path(str(root)+'_report.csv');staged=temp/target.name;report.to_csv(staged,index=False);saved.append((staged,target))
        for staged,target in saved:os.replace(staged,target)
    return [str(target) for _,target in saved]


def main():
    parser=argparse.ArgumentParser(description='Prepare a DAPHNET feature dataset from raw files.')
    parser.add_argument('--input',required=True,help='File or directory of raw TXT/CSV recordings')
    parser.add_argument('--config',required=True,help='JSON configuration saved by GUI')
    parser.add_argument('--output',required=True,help='Output filename prefix')
    parser.add_argument('--format',choices=['csv','npz','both'],default='both')
    args=parser.parse_args();source=Path(args.input)
    paths=sorted([*source.glob('*.txt'),*source.glob('*.csv')]) if source.is_dir() else [source]
    if not paths:parser.error('No input recordings found.')
    config=Config.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8')))
    table,names,report=prepare_files(paths,config,lambda i,n,name:print(f'{i}/{n}: {name}'))
    files=export_dataset(table,names,config,args.output,args.format,report)
    print(report.to_string(index=False));print('\n'.join(files))
    if (report.Status=='error').any():print('WARNING: failed recordings were excluded; inspect report before training.')

if __name__=='__main__':main()
