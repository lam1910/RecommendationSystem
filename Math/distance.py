from math import pow
def eucledianDistance(pointA, pointB):
    if isinstance(pointA, list) and isinstance(pointB, list):
        if len(pointA) != len(pointB):
            errorMes = 'Two points are in different dimension. ' \
                       'Point 1 has {0} while point 2 has {1}'.format(len(pointA), len(pointB))
            raise ValueError(errorMes)
        elif False in [(item != True and item != False) and (isinstance(item, float) or isinstance(item, int))
                 for item in pointA]:
            errorMes = 'Point 1 has invalid dimension value. Should be a real number.'
            raise ValueError(errorMes)
        elif False in [(item != True and item != False) and (isinstance(item, float) or isinstance(item, int))
                 for item in pointB]:
            errorMes = 'Point 2 has invalid dimension value. Should be a real number.'
            raise ValueError(errorMes)
        else:
            tmp = []
            for i in range(len(pointA)):
                tmp.append(pow(pointA[i] - pointB[i], 2))

            result = pow(sum(tmp), 0.5)
            return result
    else:
        raise TypeError('Both points has to be represented as a list of coordinates.')