import numpy as np
from cv2 import cv2
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm


def conv(input: np.ndarray, filters: np.ndarray, st=1) -> np.ndarray:
    _conv_check(input, filters, st)
    filter_height, filter_width, filter_color, filter_number = filters.shape
    input_height, input_width = input.shape
    output_height = int((input_height - filter_height) / st + 1)
    output_width = int((input_width - filter_width) / st + 1)
    output = np.zeros(shape=(output_height, output_width, filter_number))

    for i in tqdm(range(filter_number), desc="conv"):
        for h in range(output_height):
            for w in range(output_width):
                sum = 0
                for y in range(filter_height):
                    for x in range(filter_width):
                        for c in range(filter_color):
                            sum += (
                                input[h * st + y][w * st + x][c]
                                * filters[y][x][c][i]
                            )

                output[h][w][i] = sum

    return output


def fast_conv(input: np.ndarray, filter: np.ndarray) -> np.ndarray:
    _conv_check(input, filter)

    filter_height, filter_width, filter_color, filter_number = filter.shape
    input_height, input_width, i_c = input.shape
    output_height = int((input_height - filter_height) + 1)
    output_width = int((input_width - filter_width) + 1)

    str = as_strided(
        input,
        (output_height, output_width, filter_height, filter_width, i_c),
        input.strides[:2] + input.strides,
    )
    return np.tensordot(str, filter, axes=3)



def batch_norm(input: np.ndarray, b: float = 0, g: float = 1) -> np.ndarray:
    mean = input.mean()
    std = input.std()
    return ((input - mean) / std) * g + b


def _conv_check(input: np.ndarray, filter: np.ndarray, st=1):
    if st < 1:
        raise RuntimeError("Step < 1!")

    if len(input.shape) != len(filter.shape[:-1]):
        raise RuntimeError("Image shape != filter shape")

    if any(
        input.shape[i] < filter.shape[i]
        for i in range(len(input.shape))
    ):
        raise RuntimeError(
            f"Image is smaller than filter {input.shape} {filter.shape[:-1]}"
        )

def max_pooling(input: np.ndarray, height: int, width: int) -> np.ndarray:
    if not 0 < height <= input.shape[0] or not 0 < width <= input.shape[1]:
        raise RuntimeError("Incorrect kernel size")
    output = np.empty(
        shape=(
            int(input.shape[0] / height),
            int(input.shape[1] / width),
            input.shape[2],
        )
    )
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            output[h, w] = np.max(
                input[h * height : (h + 1) * height, w * width : (w + 1) * width],
                axis=(0, 1),
            )

    return output


def relu(input: np.ndarray) -> np.ndarray:
    return np.maximum(input, 0)


def pixel_wise_softmax(input: np.ndarray) -> np.ndarray:
    input_height, input_width, _ = input.shape
    output = np.zeros(shape=input.shape)

    for h in range(input_height):
        for w in range(input_width):
            output[h, w, :] = softmax(input[h, w])

    return output


def softmax(input: np.ndarray) -> np.ndarray:
    return np.exp(input) / sum(np.exp(input))


if __name__ == "__main__":
    # Intitial
    image = cv2.imread("image2.jpg")
    filters = np.random.normal(size=(3, 3, 3, 5))

    # Steps  img -> conv -> norm -> ReLU -> pool -> softmax
    img_conv = fast_conv(image, filters)
    img_batch = batch_norm(img_conv)
    img_relu = relu(img_batch)
    img_pooling = max_pooling(img_relu, height=2, width=2)
    img_pixel_wise = pixel_wise_softmax(img_pooling)


    # Results
    print("CONV shape ", img_conv.shape)
    print("BATCH NORMALIZATION shape ", img_batch.shape)
    print("RELU shape", img_relu.shape)
    print("MAX POOLING shape ", img_pooling.shape)
    print("PIXELWISE SOFTMAX shape", img_pixel_wise.shape)
    cv2.imshow('image', image)
    cv2.waitKey()