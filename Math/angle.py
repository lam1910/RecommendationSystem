def cos(vecA, vecB):
    import numpy as np
    if isinstance(vecA, list) and isinstance(vecB, list):
        if len(vecA) != len(vecB):
            errorMes = 'Two vectors are in different dimension. ' \
                       'Vector 1 has {0} while vector 2 has {1}'.format(len(vecA), len(vecB))
            raise ValueError(errorMes)
        elif False in [(item != True and item != False) and (isinstance(item, float) or isinstance(item, int))
                 for item in vecA]:
            errorMes = 'Vector 1 has invalid dimension value. Should be a real number.'
            raise ValueError(errorMes)
        elif False in [(item != True and item != False) and (isinstance(item, float) or isinstance(item, int))
                 for item in vecB]:
            errorMes = 'Vector 2 has invalid dimension value. Should be a real number.'
            raise ValueError(errorMes)
        else:
            lenA = np.linalg.norm(x = vecA)
            lenB = np.linalg.norm(x = vecB)
            result = np.dot(vecA, vecB) / (lenA * lenB)
            return result
    else:
        raise TypeError('Both vectors has to be represented as a list of coordinates.')

def sin(vecA, vecB):
    from math import pow
    return pow(1 - pow(cos(vecA, vecB), 2), 0.5)