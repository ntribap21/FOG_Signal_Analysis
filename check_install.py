"""Check filenames, imports and shared pipeline version before running the GUI."""
from pathlib import Path
import ast
import importlib
import sys

EXPECTED='fog-unified-1.2-no-dwt'
FILES=['feature_registry.py','main.py','signal_processing.py','data_prepare.py','batch_processing.py','train_fog.py']

def main():
    root=Path(__file__).resolve().parent
    errors=[]
    for name in FILES:
        path=root/name
        if not path.is_file():errors.append('Thiếu '+name);continue
        try:
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            versions=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RELEASE_VERSION' for t in n.targets) and isinstance(n.value,ast.Constant)]
            if versions!=[EXPECTED]:errors.append(name+' không cùng phiên bản.')
        except (SyntaxError,UnicodeError) as exc:errors.append(name+': '+str(exc))
    if errors:
        print('\n'.join(errors));return 1
    sys.path.insert(0,str(root))
    for name in ['feature_registry','signal_processing','data_prepare','batch_processing','train_fog','main']:
        try:
            module=importlib.import_module(name)
            if Path(module.__file__).resolve().parent!=root:raise ValueError('Nạp module từ thư mục khác: '+str(module.__file__))
            if module.RELEASE_VERSION!=EXPECTED:raise ValueError('Phiên bản không khớp.')
            print('OK '+name+' -> '+str(module.__file__))
        except (Exception,SystemExit) as exc:
            print('ERROR '+name+': '+str(exc));print('Kiểm tra tên file và chạy: pip install -r requirements.txt');return 1
    from signal_processing import ProcessingEngine,Config
    print('OK ProcessingEngine / Config. Có thể chạy: python main.py')
    return 0

if __name__=='__main__':sys.exit(main())
