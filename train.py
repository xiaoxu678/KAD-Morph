import os
import csv
import time
import torch
import numpy as np
from argparse import ArgumentParser

# project imports
import datagenerators
import networks
import losses


def Dice(vol1, vol2, labels=None, nargout=1):

    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem)
    return (dicem, labels)


def NJD(displacement):

    D_y = (displacement[1:, :-1, :-1, :] - displacement[:-1, :-1, :-1, :])
    D_x = (displacement[:-1, 1:, :-1, :] - displacement[:-1, :-1, :-1, :])
    D_z = (displacement[:-1, :-1, 1:, :] - displacement[:-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    Ja_value = D1 - D2 + D3

    return np.sum(Ja_value < 0)


def _decode_name(name):
    if isinstance(name, (bytes, np.bytes_)):
        return name.decode()
    return str(name)


def _load_vol_only(path, name):
    full_path = os.path.join(path, _decode_name(name))
    vol = datagenerators.load_volfile(full_path, np_var='vol')
    vol = vol[np.newaxis, np.newaxis, ...]
    return vol


def _load_vol_seg(path, name):
    full_path = os.path.join(path, _decode_name(name))
    npz_data = datagenerators.load_volfile(full_path, np_var='all')

    vol = npz_data['vol'][np.newaxis, np.newaxis, ...]
    seg = npz_data['label'][np.newaxis, np.newaxis, ...]
    return vol, seg


def _load_train_batch(train_dir, pairs_subset, batch_indices):
    fixed_batch = []
    moving_batch = []
    for idx in batch_indices:
        pair = pairs_subset[idx]
        fixed_batch.append(_load_vol_only(train_dir, pair[0]))
        moving_batch.append(_load_vol_only(train_dir, pair[1]))

    if len(batch_indices) > 1:
        fixed = np.concatenate(fixed_batch, axis=0)
        moving = np.concatenate(moving_batch, axis=0)
    else:
        fixed = fixed_batch[0]
        moving = moving_batch[0]

    # keep shape style compatible with existing losses
    zero = np.zeros((1), dtype=np.float32)
    return [fixed, moving], [fixed, zero, fixed, zero]


def _select_epoch_pairs(train_pairs_arr, epoch, pairs_per_epoch):
    total = len(train_pairs_arr)
    if total == 0:
        raise ValueError("Empty training pairs.")

    start = (epoch * pairs_per_epoch) % total
    indices = (np.arange(pairs_per_epoch) + start) % total
    return train_pairs_arr[indices], start


def _resolve_device(device):
    device = str(device).strip().lower()
    if device == 'cpu':
        return torch.device('cpu')

    if device.startswith('gpu'):
        gpu_id = int(device[3:])
    elif device.startswith('cuda:'):
        gpu_id = int(device.split(':', 1)[1])
    else:
        raise ValueError(f"Unsupported device '{device}'. Use cpu, gpuN, or cuda:N.")

    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")

    num_devices = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= num_devices:
        raise ValueError(f"Requested GPU {gpu_id}, but only {num_devices} CUDA device(s) are visible.")

    torch.cuda.set_device(gpu_id)
    return torch.device(f'cuda:{gpu_id}')


def _validate_one_set(model, spatial_transformer, grid_transformer, affine_transformer, data_dir, pairs_arr, device, tag):
    valid_dice = []
    valid_affine = []
    valid_njd = []

    total = len(pairs_arr)
    for idx, valid_pair in enumerate(pairs_arr, start=1):
        print(f'[{tag}] {idx}/{total}')

        fixed_vol, fixed_seg = _load_vol_seg(data_dir, valid_pair[0])
        moving_vol, moving_seg = _load_vol_seg(data_dir, valid_pair[1])

        fixed_vol = torch.from_numpy(fixed_vol).to(device).float()
        fixed_seg = torch.from_numpy(fixed_seg).to(device).float()
        moving_vol = torch.from_numpy(moving_vol).to(device).float()
        moving_seg = torch.from_numpy(moving_seg).to(device).float()

        with torch.no_grad():
            pred = model(fixed_vol, moving_vol)
            if len(pred) > 4:
                warped_seg = grid_transformer(moving_seg, pred[4])
                flow_tensor = networks.grid_to_flow(pred[4])
            else:
                warped_seg = spatial_transformer(moving_seg, pred[1])
                flow_tensor = pred[1]
            affine_seg = affine_transformer(moving_seg, pred[3])

        fixed_seg_np = fixed_seg.detach().cpu().numpy().squeeze()
        warped_seg_np = warped_seg.detach().cpu().numpy().squeeze()
        affine_seg_np = affine_seg.detach().cpu().numpy().squeeze()

        valid_dice.append(Dice(warped_seg_np, fixed_seg_np))
        valid_affine.append(Dice(affine_seg_np, fixed_seg_np))

        flow = flow_tensor.detach().cpu().permute(0, 2, 3, 4, 1).numpy().squeeze()
        valid_njd.append(NJD(flow))

    return {
        'dice': float(np.mean(valid_dice)),
        'affine': float(np.mean(valid_affine)),
        'njd': float(np.mean(valid_njd)),
    }


def train(train_dir,
          train_pairs,
          valid_dir,
          valid_pairs,
          valid_dir2,
          valid_pairs2,
          valid_dir3,
          valid_pairs3,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size):

    # preparation
    train_pairs_arr = np.load(os.path.join(train_dir, train_pairs), allow_pickle=True)

    if valid_dir2 is None:
        valid_dir2 = valid_dir
    if valid_pairs2 is None:
        valid_pairs2 = valid_pairs
    if valid_dir3 is None:
        valid_dir3 = valid_dir
    if valid_pairs3 is None:
        valid_pairs3 = valid_pairs

    valid_pairs_arr_1 = np.load(os.path.join(valid_dir, valid_pairs), allow_pickle=True)
    valid_pairs_arr_2 = np.load(os.path.join(valid_dir2, valid_pairs2), allow_pickle=True)
    valid_pairs_arr_3 = np.load(os.path.join(valid_dir3, valid_pairs3), allow_pickle=True)

    # prepare model folder
    os.makedirs(model_dir, exist_ok=True)

    csv_name = './csv/NICE_Net_4_dataset.csv'
    os.makedirs(os.path.dirname(csv_name), exist_ok=True)
    # csv_header = [
    #     'epoch', 'train_loss',
    #     'valid1_Dice', 'valid1_Affine', 'valid1_NJD',
    #     'valid2_Dice', 'valid2_Affine', 'valid2_NJD',
    #     'valid3_Dice', 'valid3_Affine', 'valid3_NJD',
    # ]
    # need_header = (not os.path.exists(csv_name)) or (os.path.getsize(csv_name) == 0)
    # if need_header:
    #     with open(csv_name, 'a', newline='') as f:
    #         csv.writer(f).writerow(csv_header)

    # device handling
    device = _resolve_device(device)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        print(f'Using CUDA device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        print('Using CPU')

    # prepare the model
    model = networks.NICE_Trans()
    model.to(device)
    if load_model != './':
        print('loading', load_model)
        state_dict = torch.load(load_model, map_location=device)
        model.load_state_dict(state_dict)

    # transfer model
    spatial_transformer = networks.SpatialTransformer_block(mode='nearest')
    spatial_transformer.to(device)
    spatial_transformer.eval()

    grid_transformer = networks.GridTransformer_block(mode='nearest')
    grid_transformer.to(device)
    grid_transformer.eval()

    affine_transformer = networks.AffineTransformer_block(mode='nearest')
    affine_transformer.to(device)
    affine_transformer.eval()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # prepare losses
    loss_fns = [losses.NCC(win=9).loss, losses.Regu_loss, losses.NCC(win=9).loss]
    loss_weights = [1.0, 1.0, 1.0]

    # fixed train-pair chunk per epoch as requested
    train_pairs_per_epoch = 500

    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()

        epoch_pairs, start_idx = _select_epoch_pairs(train_pairs_arr, epoch, train_pairs_per_epoch)
        print(f'Epoch {epoch + 1}: using training pair window start={start_idx + 1}, size={len(epoch_pairs)}')

        # training
        model.train()
        train_losses = []
        train_total_loss = []

        for step in range(steps_per_epoch):
            print('epoch:', epoch, '------', step + 1, '/', steps_per_epoch)

            b_start = (step * batch_size) % len(epoch_pairs)
            batch_indices = (np.arange(b_start, b_start + batch_size) % len(epoch_pairs)).tolist()

            inputs_np, labels_np = _load_train_batch(train_dir, epoch_pairs, batch_indices)
            inputs = [torch.from_numpy(d).to(device).float() for d in inputs_np]
            labels = [torch.from_numpy(d).to(device).float() for d in labels_np]

            pred = model(*inputs)

            loss = 0.0
            loss_list = []
            for i, loss_fn in enumerate(loss_fns):
                curr_loss = loss_fn(labels[i], pred[i]) * loss_weights[i]
                loss_list.append(curr_loss.item())
                loss = loss + curr_loss
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation on 3 datasets
        model.eval()
        valid_1 = _validate_one_set(
            model, spatial_transformer, grid_transformer, affine_transformer, valid_dir, valid_pairs_arr_1, device, 'valid1'
        )
        valid_2 = _validate_one_set(
            model, spatial_transformer, grid_transformer, affine_transformer, valid_dir2, valid_pairs_arr_2, device, 'valid2'
        )
        valid_3 = _validate_one_set(
            model, spatial_transformer, grid_transformer, affine_transformer, valid_dir3, valid_pairs_arr_3, device, 'valid3'
        )

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)

        train_total = float(np.mean(train_total_loss)) if train_total_loss else 0.0
        train_parts = np.mean(train_losses, axis=0) if train_losses else np.zeros((3,), dtype=np.float32)
        train_loss_info = 'Train loss: %.4f  (%.4f, %.4f, %.4f)' % (
            train_total, train_parts[0], train_parts[1], train_parts[2]
        )

        valid_1_info = 'Valid1 DSC/Aff/NJD: %.4f / %.4f / %.2f' % (
            valid_1['dice'], valid_1['affine'], valid_1['njd']
        )
        valid_2_info = 'Valid2 DSC/Aff/NJD: %.4f / %.4f / %.2f' % (
            valid_2['dice'], valid_2['affine'], valid_2['njd']
        )
        valid_3_info = 'Valid3 DSC/Aff/NJD: %.4f / %.4f / %.2f' % (
            valid_3['dice'], valid_3['affine'], valid_3['njd']
        )

        print(' - '.join((epoch_info, time_info, train_loss_info, valid_1_info, valid_2_info, valid_3_info)), flush=True)

        # save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, '%02d.pt' % (epoch + 1)))

        # write csv
        # with open(csv_name, 'a', newline='') as f:
        #     csv.writer(f).writerow([
        #         epoch + 1,
        #         train_total,
        #         valid_1['dice'], valid_1['affine'], valid_1['njd'],
        #         valid_2['dice'], valid_2['affine'], valid_2['njd'],
        #         valid_3['dice'], valid_3['affine'], valid_3['njd'],
        #     ])
        # write csv
        with open(csv_name, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"loss:{train_total:.3f}",
                f"valid1_Dice:{valid_1['dice']:.3f}",
                f"valid1_Affine:{valid_1['affine']:.3f}",
                f"valid1_NJD:{valid_1['njd']:.3f}",
                f"valid2_Dice:{valid_2['dice']:.3f}",
                f"valid2_Affine:{valid_2['affine']:.3f}",
                f"valid2_NJD:{valid_2['njd']:.3f}",
                f"valid3_Dice:{valid_3['dice']:.3f}",
                f"valid3_Affine:{valid_3['affine']:.3f}",
                f"valid3_NJD:{valid_3['njd']:.3f}",
            ])


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default='../TrainDataset/',
                        help="folder with training data")
    parser.add_argument("--train_pairs", type=str,
                        dest="train_pairs", default='train_pairs.npy',
                        help="training pairs(.npy)")

    parser.add_argument("--valid_dir", type=str,
                        dest="valid_dir", default='../Mindboggle/',
                        help="folder with validation data #1")
    parser.add_argument("--valid_pairs", type=str,
                        dest="valid_pairs", default='valid_pairs.npy',
                        help="validation pairs(.npy) #1")

    parser.add_argument("--valid_dir2", type=str,
                        dest="valid_dir2", default='../LPBA/',
                        help="folder with validation data #2")
    parser.add_argument("--valid_pairs2", type=str,
                        dest="valid_pairs2", default='valid_pairs.npy',
                        help="validation pairs(.npy) #2")

    parser.add_argument("--valid_dir3", type=str,
                        dest="valid_dir3", default='../Buckner/',
                        help="folder with validation data #3")
    parser.add_argument("--valid_pairs3", type=str,
                        dest="valid_pairs3", default='valid_pairs.npy',
                        help="validation pairs(.npy) #3")

    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/4_dataset',
                        help="models folder")

    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load model file to initialize with")

    parser.add_argument("--device", type=str, default='gpu3',
                        dest="device", help="cpu, gpuN, or cuda:N")

    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial epoch")

    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=250,
                        help="number of epoch")

    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=500,
                        help="iterations of each epoch")

    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch size")

    args = parser.parse_args()
    train(**vars(args))
