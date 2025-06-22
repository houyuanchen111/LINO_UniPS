import scipy.io

def write_depth_image(data, filename): # kernel = 1st or 2nd
    scipy.io.savemat(filename, {'name':data})
    return 1
