import os
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.stl10 import STL10PAIR
from model.resnet import ResNet18
from tools.normalize import Normalize
from model.npc_model import NonParametricClassifier
from tools.averagetracker import AverageTracker
from tools.tools import check_clustering_metrics
from losses.crossview_contrastive_Loss import crossview_contrastive_Loss
from losses.Loss_ID import Loss_ID
from losses.Loss_FMI import Loss_FMI


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=str, default="4")
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=2000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args


def main():
    args = parse()
    batch_size, epochs = args.batch_size, args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=96),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    train_data= STL10PAIR(root='data', split='labeled', transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)

    low_dim = 128
    net = ResNet18(low_dim=low_dim)
    norm = Normalize(2)
    npc = NonParametricClassifier(input_dim=low_dim,
                                  output_dim=len(train_data),
                                  tau=1.0,
                                  momentum=0.5)
    loss_id = Loss_ID(tau2=2.0)
    loss_fmi = Loss_FMI()
    net, norm = net.to(device), norm.to(device)
    npc, loss_id, loss_fmi = npc.to(device), loss_id.to(device), loss_fmi.to(device)
    # lr = 0.03
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=0.05,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=False,
                                dampening=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        [500, 950, 1350, 2050, 2350, 2750, 3350, 3750, 4250, 4550],
                                                        gamma=0.5)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net,
                                    device_ids=range(len(
                                        args.gpus.split(","))))
        torch.backends.cudnn.benchmark = True

    trackers = {n: AverageTracker() for n in ["loss", "loss_id", "loss_fmi", "loss_imi"]}


    if os.path.exists(args.pretrained):
        print('Restart from checkpoint {}'.format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
        net.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print('No checkpoint file at {}'.format(args.pretrained))
        start_epoch = 0
        net = net.cuda()

    file_name = "demo.py"
    results = {'Epochs': [], 'loss_id': [],'loss_fmi':[],'loss_imi':[], 'k_means_acc': [], 'k_means_nmi': [],
               'k_means_ari': []}
    save_name_pre = '{}_{}_{}'.format(file_name, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    tmp = 1
    for epoch in range(start_epoch+1, epochs + 1):
            net.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
            for pos_1, pos_2, target, index in train_bar:
                optimizer.zero_grad()
                inputs_1 = pos_1.to(device, dtype=torch.float32, non_blocking=True)
                inputs_2 = pos_2.to(device, dtype=torch.float32, non_blocking=True)
                indexes = index.to(device, non_blocking=True)
                features_1 = norm(net(inputs_1))
                features_2 = norm(net(inputs_2))
                outputs = npc(features_1, indexes)

                loss_imi = crossview_contrastive_Loss(features_1, features_2)

                loss_id = loss_id(outputs, indexes)
                loss_fmi = loss_fmi(features_1)
                #loss_fmi = loss_fmi(features_2)

                tot_loss = loss_id + 0.00001 * loss_fmi + 0.000001 * loss_imi
                tot_loss.backward()

                optimizer.step()
                # track loss
                trackers["loss"].add(tot_loss.item())
                trackers["loss_id"].add(loss_id.item())
                trackers["loss_imi"].add(loss_imi.item())
                trackers["loss_fmi"].add(loss_fmi.item())
            lr_scheduler.step()

            # logging
            # postfix = {name: t.avg() for name, t in trackers.items()}
            # epoch_bar.set_postfix(**postfix)
            # for t in trackers.values():
            #     t.reset()

            # check clustering acc
            # torch.cuda.empty_cache()
            if (epoch == 0) or (((epoch + 1) % 10) == 0):
                acc, nmi, ari = check_clustering_metrics(npc, train_loader)
                print("Epoch:{} Loss_id:{} Loss_mr:{} Loss_cl:{} Kmeans ACC, NMI, ARI = {}, {}, {}".format(epoch+1, loss_id, loss_fmi, loss_imi, acc, nmi, ari))

                results['Epochs'].append(tmp * 10)
                results['loss_id'].append(loss_id.item())
                results['loss_fmi'].append(loss_fmi.item())
                results['loss_imi'].append(loss_imi.item())
                results['k_means_acc'].append(acc)
                results['k_means_nmi'].append(nmi)
                results['k_means_ari'].append(ari)
                # Checkpoint
                print('Checkpoint ...')
                torch.save({'optimizer': optimizer.state_dict(), 'model': net.state_dict(),
                            'epoch': epoch + 1}, args.pretrained)

                tmp = tmp + 1
                # save statistics   df = pd.DataFrame.from_dict(d, orient='index')
                data_frame = pd.DataFrame.from_dict(data=results, orient='index')
                data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre))



if __name__ == "__main__":
    main()
