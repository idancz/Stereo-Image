import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from stereo_algorithms import Algorithms


COST1 = 0.5
COST2 = 3.0
WIN_SIZE = 3
DISPARITY_RANGE = 20


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


def forward_map(left_image, labels):
    labels -= DISPARITY_RANGE
    mapped = np.zeros_like(left_image)
    for row in range(left_image.shape[0]):
        cols = range(left_image.shape[1])
        mapped[row,
               np.clip(cols - labels[row, ...], 0, left_image.shape[1] - 1),
               ...] = left_image[row, cols, ...]
    return mapped


def load_data(is_your_data=False):
    # Read the data:
    if is_your_data:
        left_image = mpimg.imread('my_image_left.png')
        right_image = mpimg.imread('my_image_right.png')
    else:
        left_image = mpimg.imread('image_left.png')
        right_image = mpimg.imread('image_right.png')
    return left_image, right_image


def main():
    global COST1, COST2, WIN_SIZE, DISPARITY_RANGE
    left_image, right_image = load_data()
    solution = Algorithms()
    # Compute Sum-Square-Diff distance
    tt = tic()
    ssdd = solution.ssd_distance(left_image.astype(np.float64),
                                 right_image.astype(np.float64),
                                 win_size=WIN_SIZE,
                                 dsp_range=DISPARITY_RANGE)
    print(f"SSDD calculation done in {toc(tt):.4f}[seconds]")

    # Construct naive disparity image
    tt = tic()
    label_map = solution.naive_labeling(ssdd)
    print(f"Naive labeling done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.subplot(1, 2, 2)
    plt.imshow(label_map)
    plt.colorbar()
    plt.title('Naive Depth')

    # Smooth disparity image - Dynamic Programming
    tt = tic()
    label_smooth_dp = solution.dp_labeling(ssdd, COST1, COST2)
    print(f"Dynamic Programming done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_dp)
    plt.colorbar()
    plt.title('Smooth Depth - DP')

    # Compute forward map of the left image to the right image.
    mapped_image_smooth_dp = forward_map(left_image, labels=label_smooth_dp)
    # plot left image, forward map image and right image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_smooth_dp)
    plt.title('Smooth Forward map - DP')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')

    # Generate a dictionary which maps each direction to a label map:
    tt = tic()
    direction_to_vote = solution.dp_labeling_per_direction(ssdd, COST1, COST2)
    print(f"Dynamic programming in all directions done in {toc(tt):.4f}"
          f"[seconds]")

    # Plot all directions as well as the image, in the center of the plot:
    plt.figure()
    for i in range(1, 1 + 9):
        plt.subplot(3, 3, i)
        if i < 5:
            plt.imshow(direction_to_vote[i])
            plt.title(f'Direction {i}')
        elif i == 5:
            plt.imshow(left_image)
            plt.title(f'Left Image')
        else:
            plt.imshow(direction_to_vote[i - 1])
            plt.title(f'Direction {i - 1}')

    # Smooth disparity image - Semi-Global Mapping
    tt = tic()
    label_smooth_sgm = solution.sgm_labeling(ssdd, COST1, COST2)
    print(f"SGM done in {toc(tt):.4f}[seconds]")

    # Plot Semi-Global Mapping result:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_sgm)
    plt.colorbar()
    plt.title('Smooth Depth - SGM')

    # Plot the forward map based on the Semi-Global Mapping result:
    mapped_image_smooth_sgm = forward_map(left_image, labels=label_smooth_sgm)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mapped_image_smooth_sgm)
    plt.title('Smooth Forward map - SGM')
    plt.subplot(1, 3, 3)
    plt.imshow(right_image)
    plt.title('Right Image')

    ###########################################################################
    ########################### YOUR IMAGE PLAYGROUND #########################
    ###########################################################################
    COST1 = 0.1           # YOU MAY CHANGE THIS
    COST2 = 0.8          # YOU MAY CHANGE THIS
    WIN_SIZE = 3          # YOU MAY CHANGE THIS
    DISPARITY_RANGE = 10  # YOU MAY CHANGE THIS

    your_left_image, your_right_image = load_data(is_your_data=True)
    solution = Algorithms()
    # Compute Sum-Square-Diff distance
    tt = tic()
    your_ssdd = solution.ssd_distance(your_left_image.astype(np.float64),
                                      your_right_image.astype(np.float64),
                                      win_size=WIN_SIZE,
                                      dsp_range=DISPARITY_RANGE)
    print(f"SSDD calculation on your image took: {toc(tt):.4f}[seconds]")

    # plot all directions as well as the image, in the center of the plot
    tt = tic()
    your_direction_to_vote = solution.dp_labeling_per_direction(your_ssdd,
                                                                COST1,
                                                                COST2)
    print(f"Dynamic programming in all directions took: {toc(tt):.4f}[seconds]")
    # Plot all directions as well as the image, in the center of the plot:
    plt.figure()
    for i in range(1, 1 + 9):
        plt.subplot(3, 3, i)
        if i < 5:
            plt.imshow(your_direction_to_vote[i])
            plt.title(f'Direction {i}')
        elif i == 5:
            plt.imshow(your_left_image)
            plt.title(f'Your Left Image')
        else:
            plt.imshow(your_direction_to_vote[i - 1])
            plt.title(f'Direction {i - 1}')

    # Smooth disparity image - Semi-Global Mapping
    tt = tic()
    your_label_smooth_sgm = solution.sgm_labeling(your_ssdd, COST1, COST2)
    print(f"SGM on your image done in {toc(tt):.4f}[seconds]")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(your_left_image)
    plt.title('Your Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(your_label_smooth_sgm)
    plt.colorbar()
    plt.title('Your Smooth Depth - SGM')

    # Plot the forward map based on the Semi-Global Mapping result:
    your_mapped_image_smooth_sgm = forward_map(your_left_image,
                                               labels=your_label_smooth_sgm)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(your_left_image)
    plt.title('Your Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(your_mapped_image_smooth_sgm)
    plt.title('Your Smooth Forward map - SGM')
    plt.subplot(1, 3, 3)
    plt.imshow(your_right_image)
    plt.title('Your Right Image')

    plt.show()


def bonus():
    from stereo_algorithms import Bonus
    global COST1, COST2, WIN_SIZE, DISPARITY_RANGE
    COST1 = 0.1
    COST2 = 0.8
    WIN_SIZE = 3
    DISPARITY_RANGE = 20
    K = 0.2
    P_NORM = 1
    POW = 0.5
    print(f"############### Start Bonus Part 1 ###############")
    left_image, right_image = load_data()
    solution = Algorithms()
    bon = Bonus()
    tt = tic()
    print(f"Calculate ssd norm L{P_NORM}")
    my_ssdd = bon.ssd_distance(left_image.astype(np.float64),
                                 right_image.astype(np.float64),
                                 win_size=WIN_SIZE,
                                 dsp_range=DISPARITY_RANGE, p=P_NORM)
    print(f"Bonus SSDD calculation done in {toc(tt):.4f}[seconds]")

    # Construct naive disparity image
    tt = tic()
    label_map = solution.naive_labeling(my_ssdd)
    print(f"Bonus Naive labeling done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.subplot(1, 2, 2)
    plt.imshow(label_map)
    plt.colorbar()
    plt.title(f'Bonus Naive Depth Norm L{P_NORM}')

    print(f"Calculate smooth depth of ssd norm L{P_NORM}")
    tt = tic()
    label_smooth_dp = bon.dp_labeling(my_ssdd, K, POW)
    print(f"Bonus Dynamic Programming done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_dp)
    plt.colorbar()
    plt.title(f'Bonus Smooth Depth Norm L{P_NORM}- DP')

    # ssdd = my_ssdd
    print(f"############### Start Bonus Part 2 ###############")
    print(f"Calculate ssd distance by given method")
    tt = tic()
    ssdd = solution.ssd_distance(left_image.astype(np.float64),
                               right_image.astype(np.float64),
                               win_size=WIN_SIZE,
                               dsp_range=DISPARITY_RANGE)
    print(f"Normal SSDD calculation done in {toc(tt):.4f}[seconds]")
    print(f"Calculate DP smooth depth using new method")
    tt = tic()
    label_smooth_dp = bon.dp_labeling(ssdd, K, POW)
    print(f"Bonus Dynamic Programming done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_dp)
    plt.colorbar()
    plt.title('Bonus New Smooth Depth - DP')

    print(f"Calculate SGM using new method")
    tt = tic()
    # label_smooth_sgm = solution.sgm_labeling(ssdd, COST1, COST2)
    label_smooth_sgm = bon.sgm_labeling(ssdd, K, POW)
    print(f"Bonus SGM done in {toc(tt):.4f}[seconds]")

    # Plot Semi-Global Mapping result:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_sgm)
    plt.colorbar()
    plt.title('Bonus New Smooth Depth - SGM')

    plt.show()


if __name__ == "__main__":
    main()
    bonus()