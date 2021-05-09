from pathlib import Path
import Pneumonia_detection_net
import data_processing

#data = data_processing.data_preprocessing_for_net()
#data.train_valid_preprocessing()

act = Pneumonia_detection_net.nn_for_pneumonia_detection()
#net = act.load_model("state_dict_model.pt")
act.to_train()
