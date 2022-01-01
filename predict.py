import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import common
from mydataset import MyDataset
import one_hot


def test_predict():
    test_dataset = MyDataset("./datasets/test/")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_length = test_dataset.__len__()
    m = torch.load("model.pth").cuda()
    m.eval()
    correct = 0
    for i, (image, label) in enumerate(test_dataloader):
        image = image.cuda()
        label = label.cuda()
        # print(label.shape)
        label = label.view(-1, common.captcha_array.__len__())
        # print(label.shape)
        label_text = one_hot.vector_to_text(label)
        predict_output = m(image)
        predict_output = predict_output.view(-1, common.captcha_array.__len__())
        predict_output_text = one_hot.vector_to_text(predict_output)
        # print(predict_output.shape, predict_output_text)
        if predict_output_text == label_text:
            correct += 1
            print("预测正确：正确值:{},预测值:{}".format(label_text, predict_output_text))
        else:
            print("预测失败:正确值:{},预测值:{}".format(label_text, predict_output_text))
    print("正确率{}".format(correct / test_length * 100))


def pred_pic(pic_path):
    image = Image.open(pic_path)
    image_tensor = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((140, 400)),
        transforms.ToTensor()
    ])
    image_tensor = image_tensor(image)
    image_tensor = torch.reshape(image_tensor, (-1, 1, 140, 400))
    trained_model = torch.load("model.pth")
    predict_output = trained_model(image_tensor)
    predict_output = predict_output.view(-1, len(common.captcha_array))
    predict_output_text = one_hot.vector_to_text(predict_output)
    print(predict_output_text)


if __name__ == '__main__':
    test_predict()
