"""One batch/interactive export route. No GUI imports and no alternate algorithms."""
RELEASE_VERSION = 'fog-unified-1.3-features'
import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from signal_processing import Config,prepare_recording,recording_id,merge_configs,GUI_SENSORS,GUI_AXES

from feature_registry import feature_names, axis_names, window_names

FEATURES=['Mean','RMS','Std','Variance','Min','Max','SMA','Dominant_Frequency','Total_Power','Loco_Power','Freeze_Power','Freeze_Index','Low_Freq_Ratio','PSE']


def discover_inputs(inputs):
    paths=[]
    for item in inputs:
        path=Path(item).expanduser().resolve()
        if not path.exists():raise ValueError('Input không tồn tại: '+str(path))
        if path.is_dir():paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in ('.txt','.csv')))
        elif path.suffix.lower() in ('.txt','.csv'):paths.append(path)
        else:raise ValueError('Chỉ nhận raw TXT/CSV: '+str(path))
    paths=list(dict.fromkeys(paths))
    if not paths:raise ValueError('Không có raw file.')
    identities=[recording_id(p) for p in paths]
    if len(set(identities))!=len(identities):raise ValueError('Có nhiều bản sao của cùng Recording_ID; chỉ chọn một bản.')
    return paths


def prepare_files(paths,config,progress=None):
    config=Config.from_dict(config);paths=discover_inputs(paths)
    tables=[];reports=[];names=None
    for i,path in enumerate(paths):
        if progress:progress(i,len(paths),path.name)
        try:
            table,current,report=prepare_recording(path,config)
            if names is not None and current!=names:raise ValueError('Feature schema mismatch.')
            names=current;tables.append(table);report['File']=str(path);reports.append(report)
        except Exception as exc:
            reports.append(dict(Recording_ID=recording_id(path),File=str(path),Status='error',Error=str(exc)))
    if progress:progress(len(paths),len(paths),'Complete')
    result=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    if result.empty:raise ValueError('Không có cửa sổ hợp lệ. '+'; '.join(str(r.get('Error','no complete windows')) for r in reports))
    return result,names,pd.DataFrame(reports)


def export_dataset(table,names,config,destination,format='both',report=None):
    config=Config.from_dict(config);configuration=config.dictionary()
    if format not in ('csv','npz','both'):raise ValueError('Format phải là csv, npz hoặc both.')
    if table.empty:raise ValueError('Không có dữ liệu để xuất.')
    root=Path(destination).expanduser().resolve()
    if root.suffix.lower() in ('.csv','.npz','.json'):root=root.with_suffix('')
    root.parent.mkdir(parents=True,exist_ok=True)
    required=['Subject_ID','Recording_ID','Segment_ID','Window_ID','Start_Time','End_Time','Label','FOG_Ratio','Config_ID']
    if not set(required+names).issubset(table.columns):raise ValueError('Thiếu feature hoặc metadata.')
    values=table[names].to_numpy(float)
    if np.isinf(values).any(): raise ValueError('Feature chứa Inf.')
    if np.nanmax(np.abs(values),initial=0)>np.finfo(np.float32).max: raise ValueError('Feature vượt giới hạn float32.')
    if not table.Label.isin([0,1]).all():raise ValueError('ML label phải là 0/1.')
    if not (table.Config_ID==config['config_id']).all():raise ValueError('Config_ID của bảng không khớp cấu hình xuất.')
    payload=dict(X=table[names].to_numpy(np.float32),y=table.Label.to_numpy(np.int8),feature_names=np.asarray(names,dtype=str),subject_id=table.Subject_ID.to_numpy(str),recording_id=table.Recording_ID.to_numpy(str),segment_id=table.Segment_ID.to_numpy(np.int32),window_id=table.Window_ID.to_numpy(np.int32),window_start=table.Start_Time.to_numpy(float),window_end=table.End_Time.to_numpy(float),fog_ratio=table.FOG_Ratio.to_numpy(np.float32),config_id=table.Config_ID.to_numpy(str),config_json=np.asarray(json.dumps(configuration,ensure_ascii=False)))
    staged=[]
    with tempfile.TemporaryDirectory(dir=root.parent) as temp:
        temp=Path(temp)
        if format in ('csv','both'):
            target=Path(str(root)+'.csv');file=temp/target.name;table.to_csv(file,index=False);staged.append((file,target))
        if format in ('npz','both'):
            target=Path(str(root)+'.npz');file=temp/target.name;np.savez_compressed(file,**payload);staged.append((file,target))
        target=Path(str(root)+'_config.json');file=temp/target.name;file.write_text(json.dumps(configuration,indent=2,ensure_ascii=False),encoding='utf-8');staged.append((file,target))
        if report is not None:
            target=Path(str(root)+'_report.csv');file=temp/target.name;report.to_csv(file,index=False);staged.append((file,target))
        for file,target in staged:os.replace(file,target)
    return [str(target) for _,target in staged]


def wide_from_gui(table,config):
    """Interactive feature display uses axis rows; exported data always uses one row/window."""
    config=Config.from_dict(config)
    rid=config.get('recording_id')
    if not rid:raise ValueError('Thiếu Recording_ID khi xuất cửa sổ từ GUI.')
    names=feature_names(config)
    rows=[]
    for window,part in table.groupby('Window_ID',sort=False):
        if part.duplicated(['Sensor','Axis']).any():raise ValueError('Kênh trùng trong một window.')
        m=part.iloc[0];row=dict(Subject_ID=re.match(r'S\d+',rid).group(),Recording_ID=rid,Segment_ID=int(m.Segment_ID),Window_ID=int(window),Start_Time=float(m.Start_Time),End_Time=float(m.End_Time),Label=int(m.Binary_Label),FOG_Ratio=float(m.FOG_Ratio),Config_ID=config['config_id'])
        for sensor in config['sensors']:
            for axis in config['axes']:
                axis_name=dict(zip(GUI_AXES,['X','Y','Z']))[axis]
                channel=part[(part.Sensor==sensor)&(part.Axis==axis_name)]
                if len(channel)!=1:raise ValueError('Thiếu kênh trong window.')
                row.update({f'{sensor}_{axis}_{f}':float(channel.iloc[0][f]) for f in axis_names(config)})
        for key in window_names(config):
            if key not in part: raise ValueError('Thiếu đặc trưng cửa sổ: '+key)
            v=part[key].to_numpy(float)
            if not np.all(np.isclose(v,v[0],equal_nan=True)): raise ValueError('Đặc trưng dùng chung giữa các trục không khớp: '+key)
            row[key]=float(v[0])
        rows.append(row)
    return pd.DataFrame(rows),names


def gui_export(table,config,prefix,format_index,report=None):
    config=Config.from_dict(config);wide,names=wide_from_gui(table,config)
    return export_dataset(wide,names,config,prefix,['both','npz','csv'][format_index],report)


def main():
    p=argparse.ArgumentParser(description='Unified DAPHNET preparation: original numerical algorithms.')
    p.add_argument('--input',required=True,nargs='+');p.add_argument('--config',required=True,nargs='+');p.add_argument('--output',required=True);p.add_argument('--format',choices=['csv','npz','both'],default='both')
    args=p.parse_args()
    try:
        configs=[Config.from_dict(json.loads(Path(path).read_text(encoding='utf-8'))) for path in args.config]
        config=merge_configs(configs);table,names,report=prepare_files(args.input,config,lambda i,n,name:print(f'{i}/{n}: {name}'))
        files=export_dataset(table,names,config,args.output,args.format,report);print(report.to_string(index=False));print('\n'.join(files))
        if (report.Status=='error').any():print('Có file lỗi; xem report trước khi train.')
    except (ValueError,TypeError,KeyError,OSError) as exc:p.exit(2,'ERROR: '+str(exc)+'\n')

if __name__=='__main__':main()
