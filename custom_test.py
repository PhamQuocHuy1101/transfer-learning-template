from easyConfig import setup_config
import utils

config = setup_config('config', 'model/iFormer-S', False)

model = utils.load_template('network', config.name, config.args)
print(model.stages[0])