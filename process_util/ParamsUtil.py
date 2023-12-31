import yaml

class ParamsUtil():
    def __init__(self,config_path:str = '../configs'):
        self.config_path = config_path
        cf = self.config_path+'/'+'train_config.yaml'
        with open(cf,'r')as f:
            datas = yaml.load(f,Loader=yaml.FullLoader)
        self.data = datas['train_config']

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError('Param {0} not defined in {1}'.format(key,self.config_file))
        return self.data[key]

    def set_param(self,key,value):
        self.data[key]=value

