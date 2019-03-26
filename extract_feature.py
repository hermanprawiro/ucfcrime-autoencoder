import os
import numpy as np
import torch
import torchvision

from datasets import ucfcrime
from models import conv3d
from utils.videotransforms import video_transforms, volume_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_lists(root_dir, train=True):
    annotation_file = 'train_list_road.txt' if train else 'test_list_road.txt'
    annotation_lists = [row.strip('\n') for row in open(os.path.join('datasets', annotation_file), 'r')]
    lists = []
    for row in annotation_lists:
        row_split = row.split(' ')
        img_path = row_split[0]
        label = int(row_split[1])
        img_count = int(row_split[2])
        lists.append((img_path, label, img_count))
    
    return lists

def main():
    root_dir = "E:\\Datasets\\UCFCrime_img_best"
    output_dir = "E:\\Datasets\\UCFCrime_AE"
    batch_size = 16
    base_size = 8

    checkpoint_prefix = 'conv3dae'

    test_tfs = video_transforms.Compose([
        video_transforms.Resize(224),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor()
    ])

    model = conv3d.Conv3DAE(input_channels=3, base_size=base_size, extract_feature=True).to(device)

    checkpoint_path = os.path.join('checkpoints', '{}.tar'.format(checkpoint_prefix))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('Checkpoint loaded, last epoch = {}'.format(checkpoint['epoch'] + 1))

    # Train
    folder_list = generate_lists(root_dir, train=True)
    for row in folder_list:
        img_dir, label, img_count = row
        extract_feature(model, root_dir, os.path.join(output_dir, 'Train'), img_dir, label, img_count, transforms=test_tfs)

    # Test
    folder_list = generate_lists(root_dir, train=False)
    for row in folder_list:
        img_dir, label, img_count = row
        extract_feature(model, root_dir, os.path.join(output_dir, 'Test'), img_dir, label, img_count, transforms=test_tfs)

@torch.no_grad()
def extract_feature(model, input_root, output_root, img_dir, label, img_count, transforms):
    input_dir = os.path.join(input_root, img_dir)
    print('Currently processing {}'.format(input_dir))
    dataset = ucfcrime.UCFCrimeSingle(input_dir, label, img_count, transforms=transforms, clip_stride=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    model.eval()

    outputs = []
    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(device)
        outs = model(inputs)

        batch_size = inputs.shape[0]
        outs = outs.detach().view(batch_size, -1).cpu()
        outputs.append(outs)
    outputs = torch.cat(outputs, dim=0)

    # raw
    output_path = os.path.join(output_root.replace('_AE', '_AE_raw'), img_dir + '.npy')
    save_raw(outputs, output_path)

    # instances
    output_path = os.path.join(output_root, img_dir + '.npy')
    build_instances(outputs, output_path)

def build_instances(features, output_file, num_instances=32):
    instances_start_ids = np.round(np.linspace(0, len(features) - 1, num_instances + 1)).astype(np.int)

    segments_features = []
    for i in range(num_instances):
        start = instances_start_ids[i]
        end = instances_start_ids[i + 1]

        if start == end:
            instance_features = features[start, :]
        elif end < start:
            instance_features = features[start, :]
        else:
            instance_features = torch.mean(features[start:end, :], dim=0)
        
        instance_features = torch.nn.functional.normalize(instance_features, p=2, dim=0)
        segments_features.append(instance_features.numpy())
    
    segments_features = np.array(segments_features)
    np.save(output_file, segments_features)

def save_raw(features, output_file):
    features = features.numpy()
    np.save(output_file, features)


if __name__ == "__main__":
    main()