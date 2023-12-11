import yaml

class ParamsUtil():

    def __init__(self,config_path:str = '../configs'):
        self.config_path = config_path
        cf = self.config_path+'/'+'train_config.yaml'
        with open(cf,'r')as f:
            datas = yaml.load(f,Loader=yaml.FullLoader)
        self.data = datas['train_config']
        self.lists = list(datas['train_config'].items())
    def __getitem__(self, item):
        return self.lists[item]

    def __len__(self):
        return len(self.datas)

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError('Param {0} not defined in {1}'.format(key,self.config_file))
        return self.data[key]
