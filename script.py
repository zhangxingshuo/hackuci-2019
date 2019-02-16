from __future__ import with_statement

def parse_attrs(filename):
    image_attrs = dict()
    with open("dataset/selfie_dataset.txt") as file:
        for line in file:
            raw_data = line.split()
            name = raw_data[0]
            num_data = [float(elem) for elem in raw_data[1:]]
            image_attrs[name] = num_data
    return image_attrs

if __name__ == '__main__':
    attrs = parse_attrs("dataset/selfie_dataset.txt")
    print(attrs)