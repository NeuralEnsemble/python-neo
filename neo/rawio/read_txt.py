from neo.core import ImageSequence
import inspect


def read_sequence(file_name=None, path=None, nb_frame=None, nb_row=None, nb_column=None, units=None,
                  sampling_rate=None, spatial_scale=None, **metadata):
    # check parameter
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        if values[i] == None:
            raise ValueError(str(i) + ' require a value')

    file = open(path + '/' + file_name + '.txt', 'r')
    data = file.read()

    liste_value = []
    record = []
    for i in range(len(data)):

        if data[i] == "\n" or data[i] == "\t":
            t = "".join(str(e) for e in record)
            liste_value.append(t)
            record = []
        else:
            record.append(data[i])

    for i in range(1, len(liste_value)):
        liste_value[i] = float(liste_value[i])

    # number of frame and resolution of image need to be told in the metadata in order to
    # be extracted properly
    data = []
    nb = 0
    for i in range(nb_frame):
        data.append([])
        for y in range(nb_row):
            data[i].append([])
            for x in range(nb_column):
                data[i][y].append(liste_value[nb])
                nb += 1

    # ImageSequence require spatialscale, units , sampling_rate or sampling_period

    image_sequence = ImageSequence(image_data=data, units=units, sampling_rate=sampling_rate,
                                   spatial_scale=spatial_scale)

    return image_sequence
