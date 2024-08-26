import os
from glob import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fnmatch import fnmatch
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap


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
    dpi = 200
    scenes = ['erb', 'uc', 'wh']
    sector_left = -45 #-135
    sector_right = 45 # 135
    angle_min = -26
    angle_max = 36
    # scanner_fov_rad = np.deg2rad(angle_max - angle_min)
    # focal_length = 800  # in pixels
    # principal_point = (320, 240)  # center of the image (img_width/2, img_height/2)
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
            rgb_image = plt.imread(row['img'])
            img_width = rgb_image.shape[1]
            img_height = rgb_image.shape[0]

            with open(row['laser'], 'rb') as f:
                data = pickle.load(f)

                vector = np.array(data['ranges'][::-1])[540:900]
                angles = np.linspace(np.deg2rad(sector_left), np.deg2rad(sector_right), len(vector), endpoint=False)

                fig, axes = plt.subplots(2, 2)
                axes[0,0].imshow(rgb_image)
                axes[0,0].plot([0, img_width], [0, img_height], color='green', linewidth=0.9)
                axes[0,0].plot([img_width, 0], [0, img_height], color='green', linewidth=0.9)
                axes[0,0].figure.dpi = dpi
                # axes[0,0].set_title(f"{row['img']}")
                axes[0,0].axis('off')

                axes[1,0] = plt.subplot(223, projection='polar')
                axes[1,0].plot(angles, vector)
                axes[1,0].plot([angle_rad_max, angle_rad_max], [0, 5.1], color='red', linestyle='--')
                axes[1,0].plot([angle_rad_min, angle_rad_min], [0, 5.1], color='blue', linestyle='--')
                axes[1,0].set_thetamin(sector_left)
                axes[1,0].set_thetamax(sector_right)
                axes[1,0].set_theta_zero_location('N')
                # axes[1,0].set_xlabel(f"{row['laser']}")
                axes[1,0].set_xticks(np.pi/180. * np.linspace(sector_left, sector_right, 10, endpoint=False))
                # axes[1,0].set_xticklabels(degrees)
                axes[1,0].figure.axes[2].set_axis_off()

                depth_vector = vector[int(min_pct* len(vector)):int(max_pct*len(vector))][::-1]
                x_original = np.linspace(0, 1, len(depth_vector)) # x-coordinates for original data
                f = interp1d(x_original, depth_vector)
                x_new = np.linspace(0, 1, rgb_image.shape[1]) # x-coordinates for new data
                interp_vector = f(x_new)
                tiled_vector = np.tile(interp_vector, (rgb_image.shape[0],1))
                axes[0,1].imshow(rgb_image)
                axes[0,1].imshow(tiled_vector, cmap='autumn', vmin=0, vmax=5, alpha=0.45)
                axes[0,1].plot([0, img_width], [0, img_height], color='green', linewidth=0.9)
                axes[0,1].plot([img_width, 0], [0, img_height], color='green', linewidth=0.9)
                axes[0,1].figure.dpi = dpi
                axes[0,1].axis('off')
                idx = f'{img_basename}_{laser_basename}'
                np.save(f'output/1d_2d/{idx}_1d.npy', interp_vector)

                heatmap = np.zeros((rgb_image.shape[0], interp_vector.size))
                normed_depth = interp_vector / 5.0
                normed_depth[normed_depth < 1.0] * 1.8
                num_ones = (normed_depth * rgb_image.shape[0]).astype(np.int32)
                for i in range(normed_depth.size):
                    heatmap[-num_ones[i]:, i] = 1

                np.save(f'output/1d_2d/{idx}_2d.npy', heatmap)
                axes[1,1].imshow(rgb_image)
                axes[1,1].imshow(heatmap, cmap='autumn', alpha=0.4)
                axes[1,1].figure.dpi = dpi
                axes[1,1].axis('off')
                
                
                plt.subplots_adjust(hspace=0.01, wspace=0.01)
                plt.savefig(f"output/depth/{idx}.png",bbox_inches='tight', pad_inches=0.01, dpi=dpi)
                plt.close(fig)


def draw_individual_depth():
    """
    From mapping tables, draw RGB and depth respectively
    """
    dpi = 200
    scenes = ['erb', 'uc', 'wh']
    sector_left = -45 #-135
    sector_right = 45 # 135
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

                fig1, ax1 = plt.subplots()
                ax1.imshow(img_data)
                ax1.figure.dpi = dpi
                ax1.axis('off')
                plt.savefig(f'output/proposal/{img_basename}_rgb.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig1)

                fig2, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                ax.plot(angles, vector)
                ax.plot([angle_rad_max, angle_rad_max], [0, 5.1], color='red', linestyle='--')
                ax.plot([angle_rad_min, angle_rad_min], [0, 5.1], color='blue', linestyle='--')
                ax.set_thetamin(sector_left)
                ax.set_thetamax(sector_right)
                ax.set_theta_zero_location('N')
                # ax.set_xlabel(f"{row['laser']}")
                ax.set_xticks(np.pi/180. * np.linspace(sector_left, sector_right, 10, endpoint=False))
                # ax.set_xticklabels(degrees)
                # ax.figure.axes[0].set_axis_off()
                plt.savefig(f'output/proposal/{img_basename}_laser.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig2)


def indoor_trav_example():
    """
    draw a 3*5 figure for trav examples
    """
    path = f'data/examples'
    dpi = 300
    image_list = glob(os.path.join(path, '*'))
    fig, axs = plt.subplots(3, 5, figsize=(15, 7))

    # Flatten the axs array to easily iterate over it
    axs = axs.flatten()

    # Iterate over each image path and corresponding axis object
    for i, (image_path, ax) in enumerate(zip(image_list, axs)):
        # Load and plot the image
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.figure.dpi = dpi
        ax.axis('off')  # Turn off axis labels

        # Set title if needed (optional)
        # ax.set_title(f'Image {i+1}')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig(f'output/examples/example.png', bbox_inches='tight', pad_inches=0)


def fss_example():
    """
    draw img and mask for FSS example
    """
    dpi = 300
    colors = ['darkgray', 'lime']
    cmap = ListedColormap(colors)
    imgs = ['/mnt/hdd/segmentation_indoor_images/uc/positive/images/1661556213386775009.jpg',
            '/mnt/hdd/segmentation_indoor_images/uc/challenging/images/1661556118033713572.jpg']
    
    for img in imgs:
        img_basename = os.path.splitext(os.path.basename(img))[0]
        mask = img.replace('/images/', '/labels/').replace('.jpg', '.npy')
        mask_data = np.load(mask)
        img_data = plt.imread(img)
        fig1, ax1 = plt.subplots()
        ax1.imshow(img_data)
        ax1.figure.dpi = dpi
        ax1.axis('off')
        plt.savefig(f'output/examples/{img_basename}_rgb.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig1)
    
        fig2, ax2 = plt.subplots()
        ax2.imshow(mask_data, cmap=cmap)
        ax2.figure.dpi = dpi
        ax2.axis('off')
        plt.savefig(f'output/examples/{img_basename}_mask.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig2)


def draw_existing_fss():
    """
    draw existing FSS methods fails
    """
    episode = [
        '/mnt/hdd/segmentation_indoor_images/wh/challenging/images/1664302672125608198.jpg',
        '/mnt/hdd/segmentation_indoor_images/elb/challenging/images/1664300491521946661.jpg',
        'fs_6', 'fs_10']  # s, q, before, after

    dpi = 300
    alpha = 0.7
    colors = ['#00000000', 'lime']
    cmap = ListedColormap(colors)
    q_img = plt.imread(episode[1])
    s_img = plt.imread(episode[0])
    q_pred_filename = episode[1].split('/')[-1].strip('.jpg')
    q_target_filename = episode[1].replace('/images', '/labels', 1)
    q_target_filename = q_target_filename.replace('.jpg', '.npy')
    q_target = np.load(q_target_filename)

    # s_pred_filename = episode[0].split('/')[-1].strip('.jpg')
    s_target_filename = episode[0].replace('/images', '/labels', 1)
    s_target_filename = s_target_filename.replace('.jpg', '.npy')
    s_target = np.load(s_target_filename)

    before_filename = f'output/{episode[2]}/{q_pred_filename}_{episode[2]}.npy'
    after_filename = f'output/{episode[3]}/{q_pred_filename}_{episode[3]}.npy'
    before = np.load(before_filename)
    after = np.load(after_filename)
    # Create a sample figure
    fig, axs = plt.subplots(2,3, figsize=(8, 4), dpi=dpi)  # w,h
    axs[0,0].imshow(s_img)
    # axs[0].imshow(s_target, cmap=cmap, alpha=alpha)
    axs[0,0].set_title(f's_img')
    axs[0,0].figure.dpi = dpi
    axs[0,0].axis('off')
    
    axs[0,1].imshow(s_img)
    axs[0,1].imshow(s_target, cmap=cmap, alpha=alpha)
    axs[0,1].set_title(f's_label')
    axs[0,1].figure.dpi = dpi
    axs[0,1].axis('off')

    axs[0,2].axis('off')

    axs[1,0].imshow(q_img)
    axs[1,0].set_title(f'q_img')
    axs[1,0].figure.dpi = dpi
    axs[1,0].axis('off')

    axs[1,1].imshow(q_img)
    axs[1,1].imshow(q_target, cmap=cmap, alpha=alpha)
    axs[1,1].set_title(f'q_label')
    axs[1,1].axis('off')
    axs[1,1].figure.dpi = dpi

    axs[1,2].imshow(q_img)
    axs[1,2].imshow(before, cmap=cmap, alpha=alpha)
    axs[1,2].set_title(f'Before')
    axs[1,2].axis('off')
    axs[1,2].figure.dpi = dpi

    img_filename = f'output/examples/before_after.png'
    plt.savefig(img_filename, bbox_inches='tight',pad_inches=0.0, dpi=dpi)
    plt.close(fig)

    fig1, ax1 = plt.subplots()
    ax1.imshow(q_img)
    ax1.imshow(after, cmap=cmap, alpha=alpha)
    ax1.figure.dpi = dpi
    ax1.axis('off')
    plt.savefig(f'output/examples/q_after.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig1)


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
    # draw_individual_depth()
    # indoor_trav_example()
    # fss_example()
    # draw_existing_fss()
