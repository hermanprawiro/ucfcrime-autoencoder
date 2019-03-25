from datasets.ucfcrime import UCFCrime
from utils.videotransforms import video_transforms, volume_transforms

tfs = video_transforms.Compose([
    video_transforms.Resize(224),
    video_transforms.RandomCrop(224)
])

root_dir = 'D:\\Datasets\\UCFCrime_img'
dataset = UCFCrime(root_dir, transforms=tfs)
print(dataset[0])