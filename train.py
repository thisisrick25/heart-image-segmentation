from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss

import torch
from utilities import train
from preprocess import prepare
import config

def main():
    data_path = config.DATA_TRAIN_TEST_PATH
    model_path = config.MODEL_RESULT_PATH

    data_in = prepare(data_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    #loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
    loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, amsgrad=True)

    train(model, data_in, loss_function, optimizer, config.MAX_EPOCHS, model_path, config.TEST_INTERVAL, device)

if __name__ == '__main__':
    main()