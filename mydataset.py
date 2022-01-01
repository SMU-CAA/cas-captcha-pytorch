import os
from abc import ABC

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import one_hot


class MyDataset(Dataset, ABC):
    def __init__(self, root_dir):
        super(MyDataset, self).__init__()
        self.image_path = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # the size should be width x length
            transforms.Resize((140, 400)),
            transforms.Grayscale()
        ])
        print("loaded {} images from {}".format(len(self.image_path), root_dir))

    def __len__(self):
        return self.image_path.__len__()

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image_tensor = self.transforms(Image.open(image_path))
        filename_label = image_path.split("/")[-1].split("_")[0]
        # back to the real captcha
        real_label = filename_label.replace("add", "+").replace("sub", "-").replace("mul", "*") + "="
        # text to vector, and flatten the vector (important)
        label_tensor = one_hot.text_to_vector(real_label).view(1, -1)[0]
        return image_tensor, label_tensor


if __name__ == '__main__':
    # tensorboard log directory
    writer = SummaryWriter("logs")
    train_dataset = MyDataset("./datasets/train/")
    image, label = train_dataset[0]
    print(image.shape, label.shape)
    writer.add_image("image", image, 1)
    writer.close()
