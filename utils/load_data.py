import numpy
import os
import env

TRAIN_DATA_PATH = env.TRAIN_DATA
TEST_DATA_PATH = env.TEST_DATA

TRAIN_CAPTIONS = os.path.join(TRAIN_DATA_PATH, 'train_captions.txt')
TRAIN_IMAGES = os.path.join(TRAIN_DATA_PATH, 'train_images_names.txt')
TRAIN_VECTORS = os.path.join(TRAIN_DATA_PATH, 'train_images_vectors.bin')
TEST_CAPTIONS = os.path.join(TEST_DATA_PATH, 'test_captions.txt')
TEST_IMAGES = os.path.join(TEST_DATA_PATH, 'test_images_names.txt')
TEST_VECTORS = os.path.join(TEST_DATA_PATH, 'test_images_vectors.bin')


def load_file(file_names, file_vectors, num_vectors, vector_dimensions):
    assert os.path.isfile(file_names), "no existe archivo " + file_names
    assert os.path.isfile(file_vectors), "no existe archivo " + file_vectors
    print("leyendo " + file_names)
    names = [line.strip() for line in open(file_names)]
    assert num_vectors == len(names), "no cuadra largo archivo " + len(names)
    print("leyendo " + file_vectors)

    mat = numpy.fromfile(file_vectors, dtype=numpy.float32)
    vectors = numpy.reshape(mat, (num_vectors, vector_dimensions))

    print(str(num_vectors) + " vectores de largo " + str(vector_dimensions))
    return (names, vectors)


def load_captions(file_captions):
    assert os.path.isfile(file_captions), "no existe archivo " + file_captions
    return [line.strip().split("\t") for line in open(file_captions, encoding='utf-8')]


def load_train_vectors():
    return load_file(TRAIN_IMAGES, TRAIN_VECTORS, 20000, 2048)


def load_test_vectors():
    return load_file(TEST_IMAGES, TEST_VECTORS, 1000, 2048)


def load_train_captions():
    return load_captions(TRAIN_CAPTIONS)


def load_test_captions():
    return load_captions(TEST_CAPTIONS)


train_captions = load_train_captions()
test_captions = load_test_captions()

if __name__ == '__main__':
    for i in range(6):
        print("Imagen \"" + train_captions[i][0] + "\" tiene caption \"" + train_captions[i][1] + "\"")
    (train_names, train_vectors) = load_train_vectors()
    (test_names, test_vectors) = load_test_vectors()

    print("Imagen \"" + train_names[0] + "\" tiene descriptor visual " + str(train_vectors[0]) + " de dimension " + str(
        len(train_vectors[0])))
