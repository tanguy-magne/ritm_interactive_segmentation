import multiprocessing
import argparse
import os
import itertools
import shutil
from tqdm.auto import tqdm

from skimage import io



def crop_image(img, path):
    """
    Crop the image and the corresponding ground truth, and saves all sub images

    img : string, image to crop
    path : string, path to directory containing images. 
            At this path their should be two sub directories "images/" and "gt/"
    """
    
    res_x = 500
    res_y = 500
    n_x = 5000 // res_x
    n_y = 5000 // res_y

    # Load image and ground truth
    image_path = path + "images/" + img
    im = io.imread(image_path)
    gt_path = path + "gt/" + img
    im_gt = io.imread(gt_path)

    # Crop
    for x in range(n_x):
        for y in range(n_y):
            io.imsave(f"{path}/img_crop/{img.split('.')[0]}_{y}_{x}.png", im[y*res_y : (y+1)*res_y, x*res_x:(x+1)*res_x], check_contrast=False)
            io.imsave(f"{path}/gt_crop/{img.split('.')[0]}_{y}_{x}.png", im_gt[y*res_y : (y+1)*res_y, x*res_x:(x+1)*res_x], check_contrast=False)
    
    return f"Done cropping image {img}."



def multiprocess_crop(path):
    """
    Multiprocessing function that allows to crop all images in a parallel way

    path : string, path to directory containing images. 
            At this path their should be two sub directories "images/" and "gt/"
    """
    PROCESSES = 8
    print('Creating pool with %d processes' % PROCESSES)

    with multiprocessing.Pool(PROCESSES) as pool:
        
        # Find path for images to crop and check that they all have ground truth
        dict_dataset = {x: os.listdir(path + x) for x in ["images/", "gt/"]}
        assert dict_dataset["gt/"] == dict_dataset["images/"], "There are some images without ground truth or ground truth without images in the dataset."
        
        # Create folder for croped images 
        if not os.path.isdir(path + "img_crop/"):
            os.makedirs(path + "img_crop/")
        if not os.path.isdir(path + "gt_crop/"):
            os.makedirs(path + "gt_crop/")

        # Run Cropping
        TASKS = [[img, path] for img in dict_dataset["gt/"]]
        print(f"There are {len(TASKS)} images to crop.")
        results = [pool.apply_async(crop_image, t) for t in TASKS]

        for r in results:
            print('\t', r.get())
        print()


def split_train_test(path):
    """
    Split the cropped dataset into a training and a test set

    path : string, path to directory containing cropped images. 
            At this path their should be two sub directories "img_crop/" and "gt_crop/"
    
    """

    if not os.path.isdir(path + "img_crop_train/"):
        os.makedirs(path + "img_crop_train/")
    if not os.path.isdir(path + "img_crop_test/"):
        os.makedirs(path + "img_crop_test/")
    if not os.path.isdir(path + "gt_crop_train/"):
        os.makedirs(path + "gt_crop_train/")
    if not os.path.isdir(path + "gt_crop_test/"):
        os.makedirs(path + "gt_crop_test/")

    lo_files = os.listdir(path + 'img_crop/')

    lo_names = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    lo_idx = ['1', '2', '3', '4', '5']
    all_test_files = [''.join(x) for x in itertools.product(lo_names, lo_idx)]

    for file in tqdm(lo_files):
        if file.split('_')[0].endswith(tuple(all_test_files)):
            shutil.copyfile(path + 'gt_crop/' + file, path + 'gt_crop_test/' + file)
            shutil.copyfile(path + 'img_crop/' + file, path + 'img_crop_test/' + file)
        else:
            shutil.copyfile(path + 'gt_crop/' + file, path + 'gt_crop_train/' + file)
            shutil.copyfile(path + 'img_crop/' + file, path + 'img_crop_train/' + file)



if __name__ == '__main__':
    # example of path : C:/Users/tangu/Documents/MVA/S2/1.SatelliteImage/Project/data/AerialImageDataset/train/

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Crop a dataset of images and ground truth.')
    parser.add_argument('path', metavar='dataset_path', type=str,
                        help='path to the folder conatining the dataset')
    parser.add_argument('--no_crop', action="store_true",
                    help='whether to crop the images in the dataset')
    parser.add_argument('--no_split', action="store_true",
                    help='whether to split the dataset into train and test')

    args = parser.parse_args()
    if not args.no_crop:
        print("---\nCropping Images\n---")
        multiprocess_crop(args.path)
    if not args.no_split:
        print("---\nSplitting into train/test\n---")
        split_train_test(args.path)
    
    print("Done !")
