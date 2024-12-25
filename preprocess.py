import cv2
import os


def get_breast_region_bounding_box(image):

    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])
    for contour in contours[1:]:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        x = min(x, tmp_x)
        y = min(y, tmp_y)
        w = max(x + w, tmp_x + tmp_w) - x
        h = max(y + h, tmp_y + tmp_h) - y

    return x, y, w, h


def crop_image_based_on_bbox(image, x, y, w, h):
    return image[y:y + h, x:x + w]


def batch_process(images_folder, masks_folder, output_images_folder, output_masks_folder):
    for image_file in os.listdir(images_folder):

        if os.path.exists(os.path.join(masks_folder, image_file)):
            image_path = os.path.join(images_folder, image_file)
            mask_path = os.path.join(masks_folder, image_file)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            try:
                
                x, y, w, h = get_breast_region_bounding_box(image)

                
                cropped_image = image[y:y + h, x:x + w]
                cropped_mask = mask[y:y + h, x:x + w]

                
                cv2.imwrite(os.path.join(output_images_folder, image_file), cropped_image)
                cv2.imwrite(os.path.join(output_masks_folder, image_file), cropped_mask)
            except (ValueError, IndexError) as e:
                print(f"Error processing file: {image_file}. Error message: {e}")



images_folder = r"..."
masks_folder = r"..."
output_images_folder = r"..."
output_masks_folder = r"..."
batch_process(images_folder, masks_folder, output_images_folder, output_masks_folder)
