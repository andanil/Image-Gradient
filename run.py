import argparse
import cv2
from image_gradient_calculator import compute_gradients_among_directions


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-i', '--image', type=str, required=True, help='path to input image')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='path to output directory')
    return parser


def main(args):
    image = cv2.imread(args.image)
    print(image.shape)
    gx, gy = compute_gradients_among_directions(image)

    cv2.imwrite(f'{args.output_dir}/h_gradient.png', gx)
    cv2.imwrite(f'{args.output_dir}/v_gradient.png', gy)

    cv2.imshow('Input', image)
    cv2.imshow('Horizontal Gradient', gx)
    cv2.imshow('Vertical Gradient', gy)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Gradient calculation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
