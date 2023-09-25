import yaml

class ParamsUtil():

    def __init__(self,configpath:str = '../configs',configfile:str = 'train_config_288.yaml',confgtype:str = 'train_config'):
        self.config_path = configpath
        self.config_file = configfile
        cf = self.config_path+'/'+configfile
        with open(cf,'r')as f:
            datas = yaml.load(f,Loader=yaml.FullLoader)
        self.data = datas[confgtype]
        self.lists = list(datas[confgtype].items())
    def __getitem__(self, item):
        return self.lists[item]

    def __len__(self):
        return len(self.datas)

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError('Param {0} not defined in {1}'.format(key,self.config_file))
        return self.data[key]
