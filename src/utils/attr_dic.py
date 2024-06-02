import json
import datetime

from pathlib import Path as path


class AttrDict(dict):
    def __setitem__(self, key: str, value):
        self.__setattr__(key, value)
        
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value
        super().__setitem__(__name, __value)
        
    def set_create_time(self, create_time=None):
        if not create_time:
            self.create_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.create_time = create_time
    
    def __repr__(self):
        target_dic = {}
        for k,v in self.items():
            try:
                json.dumps(v)
                target_dic[k] = v
                continue
            except:
                pass
            try:
                target_dic[k] = str(v)
                continue
            except:
                pass
            raise TypeError(f'wrong type\n{v}: {type(v)}')
        return json.dumps(target_dic, ensure_ascii=False, indent=4)

    def _dump_json(self, json_path, overwrite=True):
        json_path = path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if not json_path.exists() or overwrite:
            with open(json_path, 'w', encoding='utf8')as f:
                f.write(str(self)+'\n')
    
    @classmethod
    def load_json(cls, json_path):
        json_path = path(json_path)
        with open(json_path, 'r', encoding='utf8')as f:
            dic = json.load(f)
        res = cls()
        for k in dic:
            res[k] = dic[k]
        return res
        # return cls(**dic)
