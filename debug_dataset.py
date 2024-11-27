from data.stero_blur_dataset import SteroBlurDataset
import cv2 as cv
dataset = SteroBlurDataset('/home/dyblurGS/data/nerfstudio/poster', factor=2)


for data in dataset:
    img = data['imgs'].numpy()[..., ::-1]
    print(f"{data=}")
    cv.imshow('1', img)
    cv.waitKey(0)
cv.destroyAllWindows()