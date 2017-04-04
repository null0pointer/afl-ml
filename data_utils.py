def random_split(input, rate):
    first = []
    second = input
    while len(first) / len(input) < rate:
        index = randrange(0, len(second))
        first.append(second[index])
        second = second[:index] + second[index+1:]
    return first, second

def ordered_split(input, rate):
    split_index = int(len(input) * rate)
    first = input[:split_index]
    second = input[split_index:]
    return first, second

def split_inputs_and_labels(raw_data, label_column, nclasses):
    inputs = []
    labels = []
    for row in raw_data:
        label_value = int(float(row[label_column]))
        label = [0] * nclasses
        label[label_value] = 1
        labels.append(label)
        input_vector = row[:label_column] + row[label_column + 1:]
        input_vector = [float(x) for x in input_vector]
        inputs.append(input_vector)
    return inputs, labels
