import __init__
import sys
print sys.path
import caffe
print caffe.__file__

from caffe import NetSpec
from caffe import NetBuilder

if __name__ == "__main__":
    builder = caffe.NetBuilder("dcgan")
