import shutil
from ImitateModel import create_model
from FetchDemoDataset import FetchDemoDataset, FetchRobotDemoDataset
import torch
from torchvision import transforms
import os
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time
import argparse

parser = argparse.ArgumentParser(
    description='Trains a Resnet 18 model on the gym FetchReach environment.')
parser.add_argument('--train_root', type=str,
                    default='/tmp/xirl_format_datasets/train/',
                    help="directory for training datasets.")
parser.add_argument('--val_root', type=str,
                    default='/tmp/xirl_format_datasets/valid/',
                    help="directory for validation datasets.")
parser.add_argument('--train_dataset', type=str,
                    default='reach_state',
                    help="name of dataset to train on.")
parser.add_argument('--valid_dataset', type=str,
                    default='reach_state',
                    help="name of dataset to validate on.")
parser.add_argument('--output_dir', type=str,
                    default='/tmp/resnet_train_results/',
                    help="training result directory")
parser.add_argument('--epoch', type=int, default=200,
                    help="number of epochs, default 200")
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help="weight decay, default is 0.0")
parser.add_argument('--window_len', type=int, default=1,
                    help="window length for video frames.")
parser.add_argument('--loss_type', type=str, default='mse',
                    help="loss type, default MSE loss.")
parser.add_argument('--train_id', type=str, default=None,
                    help="train id. Required", required=True)
parser.add_argument('--use_state', dest='use_state', action='store_true',
                    help="Whether or not to use state.")
parser.add_argument('--no_use_state', dest='use_state', action='store_false',
                    help="Whether or not to use state.")
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help="Overwrite output log folder if it exists.")
parser.add_argument('--debug', dest='debug', action='store_true',
                    help="Debug mode for ipdb.")
parser.add_argument('--real_robot', dest='real_robot', action='store_true',
                    help="Whether or not dataset is collected on real robot.")
parser.add_argument('--show_acc', dest='show_acc', action='store_true',
                    help="Show accuracy instead of loss in progress bar.")
parser.set_defaults(use_state=False, debug=False, overwrite=False,
                    real_robot=False, show_acc=False)
args = parser.parse_args()


def train_model(model, device, dataloaders, dataset_sizes, criterion, optimizer,
                scheduler, num_epochs=25):
    begin = time.time()
    # pd_log = pd.DataFrame(columns=['train_loss', 'val_loss'])
    try:
        os.makedirs(osp.join(args.output_dir, args.train_id))
    except FileExistsError:
        fol = osp.join(args.output_dir, args.train_id)
        if args.overwrite:
            shutil.rmtree(fol)
            os.makedirs(fol)
        else:
            raise FileExistsError(
                    f"Folder '{fol}' exists. Add --overwrite to overwrite.")

    train_loss = 0.0
    val_loss = 0.0
    lowest_val_loss = 1000.
    train_acc = 0.0
    val_acc = 0.0
    best_acc = 0.0
    iter = trange(num_epochs)
    for epoch in iter:
        if args.show_acc:
            iter.set_description_str(f"EP {epoch + 1}/{num_epochs}; "
                                     f"TR {train_acc * 100:.2f}; VA {val_acc * 100:.2f}/{best_acc * 100:.2f}")
        else:
            iter.set_description_str(f"EP {epoch + 1}/{num_epochs}; "
                                     f"TR {train_loss:.4f}; VA {val_loss:.4f}/{lowest_val_loss:.4f}")
        # define both losses for calculation
        running_loss_train = 0.0
        running_loss_val = 0.0
        running_corrects_train = 0
        running_corrects_val = 0

        # first calculate val loss, then train loss.
        model.eval()
        torch.set_grad_enabled(False)
        # Iterate over data for validation.
        for inputs, labels in dataloaders['val']:
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
                batch_size = inputs.size(0)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_size = inputs['images'].size(0)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                if args.show_acc:
                    _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss_val += loss.item() * batch_size
            if args.show_acc:
                running_corrects_val += torch.sum(preds == labels.data)

        # Iterate over data.
        model.train()
        torch.set_grad_enabled(True)

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
                batch_size = inputs.size(0)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_size = inputs['images'].size(0)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                if args.show_acc:
                    _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # statistics
            running_loss_train += loss.item() * batch_size
            if args.show_acc:
                running_corrects_train += torch.sum(preds == labels.data)

        scheduler.step()

        train_loss = running_loss_train / dataset_sizes['train']
        val_loss = running_loss_val / dataset_sizes['val']
        if args.show_acc:
            train_acc = running_corrects_train.double() / dataset_sizes['train']
            val_acc = running_corrects_val.double() / dataset_sizes['val']

        # print(f'\rEpoch {epoch+1}/{num_epochs} ' +
        #       'Train Loss: {:.4f}, '
        #       'Val Loss: {:.4f}'.format(train_loss, val_loss), end='')
        if not (epoch + 1) % 10:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'use_state': args.use_state,
                    'state_dim': state_dim
                },
                osp.join(
                    args.output_dir,
                    args.train_id,
                    f"checkpoint_{args.train_id}_{epoch + 1}.pt"
                )
            )
        
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'use_state': args.use_state,
                    'state_dim': state_dim
                },
                osp.join(
                    args.output_dir,
                    args.train_id,
                    f"checkpoint_{args.train_id}_min.pt"
                )
            )
        
        if args.show_acc and val_acc > best_acc:
            best_acc = val_acc
        # pd_log.loc[epoch] = (train_loss, val_loss)
        # pd_log.to_hdf(f'log_{args.train_id}.h5', 'log')

    time_elapsed = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


def load_data():
    # Data augmentation and normalization for training
    mean = torch.tensor([0.5452, 0.7267, 0.6137])
    stdev = torch.tensor([0.2064, 0.2000, 0.1730])
    data_transforms = {}
    data_transforms['train'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ColorJitter(1, 1, 1, 0.5),
        # transforms.RandomAutocontrast(),
        # transforms.RandomEqualize(),
        # ransforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, stdev)
    ])
    data_transforms['val'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, stdev)
    ])
    dirs = {
        'train': osp.join(args.train_root, args.train_dataset),
        'val': osp.join(args.val_root, args.valid_dataset)
    }
    datasets = {
        mode:
        (
        FetchDemoDataset(
            dirs[mode],
            osp.join(dirs[mode], 'labels.npy'),
            transform=data_transforms[mode],
            window_len=args.window_len,
            loss_type=args.loss_type,
            use_state=args.use_state
        ) if not args.real_robot else
        FetchRobotDemoDataset(
            dirs[mode],
            osp.join(dirs[mode], 'labels.npy'),
            transform=data_transforms[mode],
            window_len=args.window_len,
            loss_type=args.loss_type,
            use_state=args.use_state
        )
        )
        for mode in ['train', 'val']
    }
    dataloaders = {
        mode:
        DataLoader(datasets[mode], batch_size=10,
                   shuffle=(mode == 'train'), num_workers=0 if args.debug else 4)
        for mode in ['train', 'val']
    }
    dataset_sizes = {
        mode: len(datasets[mode])
        for mode in ['train', 'val']
    }
    state_dim = 0
    if args.use_state:
        state_dim = datasets['train'].state_dim
    return dataloaders, dataset_sizes, state_dim


def main():
    global state_dim
    dataloaders, dataset_sizes, state_dim = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_conv, criterion, optimizer_conv, exp_lr_scheduler = create_model(
        args.window_len, device, args.loss_type, args.use_state, state_dim
    )
    model_conv = train_model(model_conv, device, dataloaders, dataset_sizes,
                             criterion, optimizer_conv, exp_lr_scheduler,
                             num_epochs=args.epoch)


if __name__ == '__main__':
    main()
