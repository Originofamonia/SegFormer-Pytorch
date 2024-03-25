import os
from glob import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fnmatch import fnmatch
from scipy.interpolate import interp1d


def list_dirs(folder):
    dir_list = []
    for root, dirs, files in os.walk(folder):
        # for file in files:
        #     file_list.append(os.path.join(root, file))
        for d in dirs:
            dir_list.append(os.path.join(root,d))
    return dir_list


def main():
    folder_path = '/mnt/hdd/WHOLE'
    files = list_dirs(folder_path)
    for file in files:
        print(file)


def read_example():
    """
    read test files for one id 
    front_laser  img  joints
    """
    three_folders = ['front_laser']
    # idx = f'1662758477604343545.pkl'
    files = glob(f'/mnt/hdd/WHOLE/nh/joints/*', recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    # root = f'/mnt/hdd/WHOLE/nh/'
    for p in files:
        # path = os.path.join(root, p, idx)
        with open(p, 'rb') as f:
            data = pickle.load(f)
            # print(data['ranges'])
            vector = np.array(data['ranges'])
            # Calculate angles
            angles = np.linspace(0, np.pi, len(vector), endpoint=False)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            # Create polar plot
            ax.plot(angles, vector)
            # Set the theta range to 180 degrees
            ax.set_thetamin(0)
            ax.set_thetamax(180) # only draw 180 degrees

            # Add a title and show the plot
            idx = p.split('/')[-1].strip('.pkl')
            plt.savefig(f"output/depth/{idx}.png")


def get_base_names(directory, type):
    files = glob(f'{directory}/**/{type}/*', recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    base_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
    return base_names


def get_full_names(directory, type):
    files = glob(f'{directory}/**/{type}/*', recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    # base_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
    return files


def find_common_files():
    """
    find common files between RGB and depth data
    why no common base names???
    """
    buildings = ['elb','erb','nh','uce', 'wh']
    building_id = 4
    # img_dir = f'/mnt/hdd/segmentation_indoor_images/nh'
    # img_dir = f'/mnt/hdd/WHOLE/{buildings[building_id]}/front_laser'
    # depth_dir = f'/mnt/hdd/WHOLE/{buildings[building_id]}/img'
    img_dir = f'/data/zak/robot/extracted/elb/9_27_1/img'
    depth_dir = f'/data/zak/robot/extracted/elb/9_27_1/front_laser'

    # Get base names from both directories
    base_names_img = get_base_names(img_dir)
    base_names_depth = get_base_names(depth_dir)

    # Find common base names
    common_base_names = set(base_names_img) & set(base_names_depth)
    print(common_base_names)


def read_zak_files():
    """
    read front_laser and img from zak_extracted to check matching
    why still no matching???
    """
    root = f'/mnt/hdd/zak_extracted/elb'
    # img_dir = os.path.join(root, 'img')
    # laser_dir = os.path.join(root, 'front_laser')

    img_base_names = get_base_names(root, 'img')
    laser_base_names = get_base_names(root, 'front_laser')
    print(len(img_base_names), len(laser_base_names))
    common_base_names = set(img_base_names) & set(laser_base_names)
    if common_base_names:
        print(common_base_names)


def draw_by_index():
    """
    since base names don't match, try to draw by index
    """
    root = f'/mnt/hdd/zak_extracted/elb'
    img_full_names = get_full_names(root, 'img')
    laser_full_names = get_full_names(root, 'front_laser')
    print(len(img_full_names), len(laser_full_names))
    assert len(img_full_names) == len(laser_full_names)
    for img, laser in zip(img_full_names, laser_full_names):
        img_data = plt.imread(img)
        img_basename = os.path.splitext(os.path.basename(img))[0]
        laser_basename = os.path.splitext(os.path.basename(laser))[0]
        with open(laser, 'rb') as f:
            data = pickle.load(f)
            # print(data['ranges'])
            vector = np.array(data['ranges'])
            # Calculate angles
            angles = np.linspace(0, np.pi, len(vector), endpoint=False)
            fig, axes = plt.subplots(2, 1)
            axes[0].imshow(img_data)
            axes[0].set_title(f'{img}')
            axes[0].axis('off')
            
            axes[1] = plt.subplot(212, projection='polar')
            # Create polar plot
            axes[1].plot(angles, vector)
            # Set the theta range to 180 degrees
            axes[1].set_thetamin(0)
            axes[1].set_thetamax(180) # only draw 180 degrees
            axes[1].set_title(f'{laser}')
            # axes[1].axis('off')
            # Add a title and show the plot
            idx = f'{img_basename}_{laser_basename}'
            plt.savefig(f"output/depth/{idx}.png")
            plt.close(fig)


def base_full_mapping(img_full_names):
    img_base_full_dict = {}
    for fullname in img_full_names:
        base = int(os.path.splitext(os.path.basename(fullname))[0])
        img_base_full_dict[base] = fullname
    return img_base_full_dict


def plot_after_sort():
    """
    sort img and laser data, then plot by index
    """
    root = f'/mnt/hdd/zak_extracted/elb'
    img_full_names = get_full_names(root, 'img')
    laser_full_names = get_full_names(root, 'front_laser')
    print(len(img_full_names), len(laser_full_names))
    assert len(img_full_names) == len(laser_full_names)

    img_base_full_dict = base_full_mapping(img_full_names)
    laser_base_full_dict = base_full_mapping(laser_full_names)
    # img_basenames = [int(os.path.splitext(os.path.basename(x))[0]) for x in img_full_names]
    # laser_basenames = [int(os.path.splitext(os.path.basename(x))[0]) for x in laser_full_names]
    sorted_img_dict = dict(sorted(img_base_full_dict.items()))
    sorted_laser_dict = dict(sorted(laser_base_full_dict.items()))
    
    for (k1, v1), (k2, v2) in zip(sorted_img_dict.items(), sorted_laser_dict.items()):
        # print(f"Key1: {key1}, Value1: {value1}, key2: {key2}, Value2: {value2}")
        img_data = plt.imread(v1)

        with open(v2, 'rb') as f:
            data = pickle.load(f)
            # print(data['ranges'])
            vector = np.array(data['ranges'])
            # Calculate angles
            angles = np.linspace(data['angle_min'], data['angle_max'], len(vector), endpoint=False)
            fig, axes = plt.subplots(2, 1)
            axes[0].imshow(img_data)
            axes[0].set_title(f'{k1}')
            axes[0].axis('off')
            
            axes[1] = plt.subplot(212, projection='polar')
            # Create polar plot
            axes[1].plot(angles, vector)
            # Set the theta range to 180 degrees
            # axes[1].set_thetamin(0)
            # axes[1].set_thetamax(180) # only draw 180 degrees
            axes[1].set_title(f'{k2}')
            # axes[1].axis('off')
            # Add a title and show the plot
            idx = f'{k1}_{k2}'
            plt.savefig(f"output/depth/{idx}.png")
            plt.close(fig)


def plot_labels():
    """
    plot mapping from labels/
    change to only plot depth with indoor traversability images
    """
    pattern = '/mnt/hdd/zak_extracted/labels/**/*/info.csv'
    info_csv_files = glob(pattern, recursive=True)
    degrees = list(range(0,180,10))
    for info in info_csv_files:
        df = pd.read_csv(info)
        for index, row in df.iterrows():
            img_filename = row['img'].replace("/data/zak/robot/extracted", "/mnt/hdd/zak_extracted")
            laser_filename = row['front_laser'].replace("/data/zak/robot/extracted", "/mnt/hdd/zak_extracted")
            print(img_filename, laser_filename)
            img_basename = os.path.splitext(os.path.basename(img_filename))[0]
            laser_basename = os.path.splitext(os.path.basename(laser_filename))[0]
            img_data = plt.imread(img_filename)

            with open(laser_filename, 'rb') as f:
                data = pickle.load(f)
                # print(data['ranges'])
                vector = np.array(data['ranges'][::-1])
                # Calculate angles
                angles = np.linspace(data['angle_min'], data['angle_max'], len(vector), endpoint=False)
                fig, axes = plt.subplots(2, 1)
                axes[0].imshow(img_data)
                axes[0].set_title(f'{img_filename}')
                axes[0].axis('off')
                
                axes[1] = plt.subplot(212, projection='polar')
                # Create polar plot
                axes[1].plot(angles, vector)
                # Set the theta range to 180 degrees
                # axes[1].set_thetamin(0)
                # axes[1].set_thetamax(180)
                axes[1].set_theta_zero_location('N')
                axes[1].set_title(f'{laser_filename}')
                axes[1].set_xticks(np.deg2rad(degrees))  # Convert degrees to radians
                axes[1].set_xticklabels(degrees)
                # axes[1].axis('off')
                # Add a title and show the plot
                idx = f'{img_basename}_{laser_basename}'
                plt.savefig(f"output/depth/{idx}.png")
                plt.close(fig)


def create_trav_csv():
    """
    create a mapping between indoor trav dataset's image and laser from info.csv
    Done
    """
    
    img_pattern = f'/mnt/hdd/segmentation_indoor_images/*/*/images/*.jpg'
    images = glob(img_pattern, recursive=True)
    csv_pattern = '/mnt/hdd/zak_extracted/labels/**/*/info.csv'
    info_csv_files = glob(csv_pattern, recursive=True)
    erb_info = pd.read_csv(info_csv_files[3])
    uc_info = pd.read_csv(info_csv_files[4])
    wh_info = pd.read_csv(info_csv_files[5])
    common_buildings = {'erb':erb_info, 'uc':uc_info, 'wh':wh_info}
    for k, df in common_buildings.items():
        scene_pattern = f'*/{k}/*'
        scene_imgs = [x for x in images if fnmatch(x, scene_pattern)]
        df['img_basename'] = df['img'].apply(os.path.basename)
        mapping_df = pd.DataFrame(columns=['img', 'laser'])
        for img in scene_imgs:
            img_base = os.path.basename(img)
            matching_rows = df[df['img_basename'].str.contains(img_base)]
            if not matching_rows.empty:
                laser_path = matching_rows['front_laser'].values[0]
                laser_filename = laser_path.replace("/data/zak/robot/extracted", "/mnt/hdd/zak_extracted")
                mapping_df.loc[len(mapping_df.index)] = [img, laser_filename]
        mapping_df.to_csv(f'output/{k}_laser_mapping.csv')


def find_crop_degrees():
    """
    find crop degrees of laser to match its image
    read mapping csv files, draw their images
    """
    scenes = ['erb', 'uc', 'wh']
    sector_left = -45 #-135
    sector_right = 45 # 135
    degrees = list(range(sector_left,sector_right,10))
    angle_min = -27
    angle_max = 36
    angle_rad_min = np.deg2rad(angle_min)
    angle_rad_max = np.deg2rad(angle_max)
    min_pct = (angle_min+45)/90  # percentile for cropping
    max_pct = (angle_max+45)/90
    for sc in scenes:
        csv_file = f'output/{sc}_laser_mapping.csv'
        df = pd.read_csv(csv_file)
        for i, row in df.iterrows():
            img_basename = os.path.splitext(os.path.basename(row['img']))[0]
            laser_basename = os.path.splitext(os.path.basename(row['laser']))[0]
            img_data = plt.imread(row['img'])

            with open(row['laser'], 'rb') as f:
                data = pickle.load(f)

                vector = np.array(data['ranges'][::-1])[540:900]
                angles = np.linspace(np.deg2rad(sector_left), np.deg2rad(sector_right), len(vector), endpoint=False)

                fig, axes = plt.subplots(2, 2)
                axes[0,0].imshow(img_data)
                axes[0,0].set_title(f"{row['img']}")
                axes[0,0].axis('off')
                
                axes[1,0] = plt.subplot(212, projection='polar')
                # Create polar plot
                axes[1,0].plot(angles, vector)
                # Add lines at 0 and 90 degrees
                axes[1,0].plot([angle_rad_max, angle_rad_max], [0, 6], color='red', linestyle='--')  # Line at 0 degrees
                axes[1,0].plot([angle_rad_min, angle_rad_min], [0, 6], color='blue', linestyle='--')  # Line at 90 degrees
                axes[1,0].set_thetamin(sector_left)
                axes[1,0].set_thetamax(sector_right)
                axes[1,0].set_theta_zero_location('N')
                axes[1,0].set_title(f"{row['laser']}",loc='center', pad=-30)
                axes[1,0].set_xticks(np.deg2rad(degrees))  # Convert degrees to radians
                axes[1,0].set_xticklabels(degrees)
                
                cropped_vector = vector[int(min_pct* len(vector)):int(max_pct*len(vector))][::-1]
                x_original = np.linspace(0, 1, len(cropped_vector)) # x-coordinates for original data
                f = interp1d(x_original, cropped_vector)
                x_new = np.linspace(0, 1, img_data.shape[1]) # x-coordinates for new data
                interp_vector = f(x_new)
                tiled_vector = np.tile(interp_vector, (img_data.shape[0],1))
                axes[0,1].imshow(img_data)
                axes[0,1].imshow(tiled_vector, cmap='autumn', alpha=0.45)
                axes[0,1].axis('off')
                axes[0,1].set_title(f"{angle_min}~{angle_max}",loc='center', pad=-30)
                
                idx = f'{img_basename}_{laser_basename}'
                plt.savefig(f"output/depth/{idx}.png")
                plt.close(fig)


if __name__ == '__main__':
    # main()
    # read_example()
    # find_common_files()
    # read_zak_files()
    # draw_by_index()
    # plot_after_sort()
    # plot_labels()
    # create_trav_csv()
    find_crop_degrees()
