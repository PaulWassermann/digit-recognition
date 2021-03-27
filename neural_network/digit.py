from PIL import Image
import numpy as np


class Digit:

    def __init__(self, value=None, path_to_image=None, image=None, array=None):

        if value:
            self.value = value
            self.expected_output = np.array([0.0] * self.value + [1.0] + [0] * (9 - self.value))

        if path_to_image is not None:
            self.image = Image.open(path_to_image)

        elif image is not None:
            self.image = image

        # Resizing the image so its size matches the network input layer format if necessary
        if self.image.width != 28 or self.image.height != 28:
            self.image = self.image.resize((28, 28), Image.ANTIALIAS)

        # Converting the image in grayscale
        self.grayscale_image = self.image.convert("L")

        if not array:
            array = np.array(self.grayscale_image.getdata(), dtype=np.uint8)

        max_value = np.amax(array)

        self.array = 1 - (array / max_value)
        print(self.array)






