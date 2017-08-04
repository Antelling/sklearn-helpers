import numpy as np

# okay so let's make a list of points
points = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 3],
    [1, 1, 4],
    [1, 1, 5],
    [1, 2, 1],
    [1, 2, 2],
    [1, 2, 3],
    [1, 2, 4],
    [1, 2, 5],
    [1, 3, 1],
    [1, 3, 2],
    [1, 3, 3],
    [1, 3, 4],
    [1, 3, 5],
    [1, 4, 1],
    [1, 4, 2]
]

# and a list of labels
labels = [
    0,
    0,
    3,
    2,
    1,
    1,
    2,
    3,
    0,
    4,
    1,
    2,
    1,
    3,
    2,
    0,
    1
]

# now we define a pattern
pattern = [
    [0, 0, 1],
    [0, 0, 2]
]

# and the labels
pattern_labels = [0, 0]


def match_pattern(points, points_l, pattern, pattern_l):

    points = slide_to_origin(points)
    pattern = slide_to_origin(pattern)

    # we want to make a quick lookup table for points
    lookup_table = form_table(points, points_l)

    pattern_offsets = []

    # okay now loop over every point in points, then every point in pattern, and see if they all match
    for i, point in enumerate(points):
        point = np.array(point)
        should_continue = True

        for i, offset in enumerate(pattern):
            if not should_continue:
                break
            target_label = pattern_l[i]

            real_position = (np.array(offset) + point).tolist()
            try:
                a, b, c = real_position
                actual_label = lookup_table[a][b][c]

            except KeyError:
                should_continue = False
                break #there is no point at this location
            if not actual_label == target_label:
                should_continue = False
                break

        if should_continue: #the loop completed successfully, every point has a match
                pattern_offsets.append(point.tolist())

    return pattern_offsets


def form_table(points, points_l):
    table = {}
    for i, point in enumerate(points):
        a, b, c = point
        if not a in table:
            table[a] = {}
        if not b in table[a]:
            table[a][b] = {}
        table[a][b][c] = points_l[i]
    return table


def slide_to_origin(points):
    vector = []
    for dimension in 0, 1, 2:
        dimension_positions = [point[dimension] for point in points]
        min = np.min(dimension_positions)
        slide = -min  # so if 1 is the dimension of the smallest point, we add -1 to every point
        vector.append(slide)
    return (np.array(points) + np.array(vector)).tolist()  # use numpy arrays to get free vector math

print(match_pattern(points, labels, pattern, pattern_labels))

