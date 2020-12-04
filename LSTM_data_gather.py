import numpy as np

class Blink:
    def __init__(self, si, sv, ei, ev, mi, mv):
        """
        si: start index
        sv: start EAR value
        ei: end index
        ev: end EAR value
        mi: minimum EAR value's index
        mv: minimum EAR value
        """
        self.duration = ei-si+1
        self.amplitude = (sv-2*mv+ev)/2
        self.velocity = (ev-mv)/(ei-mi)


def detect_data(arr, baseline):
    """
    Calculate blink features based on inputted time series data
    arguments:
        arr: an array of Eye Aspect ratios for every frame of the video
        baseline: the baseline aspect ratio calculated during normalization
    output:
        blinks: array containing Blink instances that contain information about blinks
    """
    arr = np.array(arr)
    blinks = []
    started_count = False
    start_index, end_index = None, None
    for index, EAR in enumerate(arr):
        if EAR < baseline and not started_count:
            started_count = True
            start_index = index
        if started_count:
            if EAR > baseline:
                started_count = False
                end_index = index
                minimum = np.min(np.arr(arr[start_index, end_index+1]))
                min_loc = np.where(arr[start_index, end_index+1] == minimum)
                instance = Blink(start_index, arr[start_index],
                                 end_index, arr[end_index], min_loc, minimum)
                blinks.append(instance)
                start_index = None
                end_index = None
    return blinks
