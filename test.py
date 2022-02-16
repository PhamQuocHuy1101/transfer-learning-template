import torch
import pandas as pd
import config as cf
from network import Model
from data.model_dataset import ModelDataset
from data.transform import to_tensor
import time

start = time.time()
model = Model(cf.model['n_class'], cf.model['backbone'], cf.model['drop'])
model.to(device = cf.device)

checkpoint = torch.load(cf.model['checkpoint'], map_location = cf.device)
model.load_state_dict(checkpoint['model'])

loading_time = time.time() - start

test_df = pd.read_csv(cf.data['test_csv'])
test_data = ModelDataset(cf.data['dir_path'], test_df.path, test_df.label, to_tensor)

accuracy = 0.0
predict = []
start = time.time()
model.eval()
with torch.no_grad():
    for i in range(len(test_data)):
        inputs, labels = test_data[i]

        inputs = inputs.to(device = cf.device)
        outputs = model(inputs.unsqueeze(0))
        outputs = outputs.cpu()

        y_predict = torch.softmax(outputs, dim = 1).argmax(dim = 1)
        accuracy += torch.sum(y_predict == labels).item()
        predict.append(y_predict[0].item())

print("Loading: ", loading_time)
print("Predict time: ", time.time() - start)
print("Accuracy: ", accuracy / len(test_data))
test_df['predict'] = predict
test_df.to_csv('./predict.csv', index = False)
