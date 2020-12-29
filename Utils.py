import matplotlib.pyplot as plt
import numpy


def file_print(file_name, text):
    f = open(file_name, 'w')
    f.write(str(text))
    f.close()


def get_cool_matrix(classesQuantity, real_classes, predicted_classes):  # real_classes = (img, label)
    cool_matrix = [[(0, 0)] * classesQuantity]
    for i in range(real_classes):
        for j in range(classesQuantity):
            if cool_matrix[real_classes[i][1]][j][1] < predicted_classes[i][j]:
                cool_matrix[real_classes[i][1]][j] = (real_classes[i][0], predicted_classes[i][j])
    return cool_matrix


def print_cool_matrix(class_names, real_classes, predicted_classes, file_name):
    classesQuantity = len(class_names)
    cool_matrix = get_cool_matrix(classesQuantity, real_classes, predicted_classes)

    plt.figure(figsize=(50, 50))
    fig, ax = plt.subplots(nrows=classesQuantity, ncols=classesQuantity)
    position = numpy.array(classesQuantity)
    ax.set_xticks(position)
    ax.set_xticklabels(class_names)
    ax.set_yticks(position)
    ax.set_yticklabels(class_names)
    for i in range(classesQuantity):
        for j in range(classesQuantity):
            ax[i, j].imshow(cool_matrix[i][j][0])
    fig.savefig(file_name)
    plt.show()
