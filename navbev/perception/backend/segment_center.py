import cv2
import glob
import numpy as np


def find_center_rectangle(image):
    mask = cv2.imread(image, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)
        # Find the center of the rectangle
        center_x = x + w // 2
        center_y = y + h // 2

        return [center_x, center_y]


def main():
    image_path_folder = "/home/rrc/Downloads/Talk2Car-RefSeg/val_masks_new/"
    num_images = len(
        glob.glob(
            "/home/rrc/Downloads/Talk2Car-RefSeg/val_masks_new/gt_img_ann_train*.png"
        )
    )

    goalPoints = np.zeros([num_images, 2])

    for i in range(num_images):
        image_path = image_path_folder + "gt_img_ann_train_" + str(i) + ".png"
        center = find_center_rectangle(image_path)
        goalPoints[i] = center

    np.save("goalPoints.npy", goalPoints)


if __name__ == "__main__":
    main()
