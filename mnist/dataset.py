import h5py
import cPickle
import gzip
from os.path import dirname, join, isfile
from helpers import helpers
import numpy as np


# _default_name = join(dirname(__file__), "mnist.h5")
# _default_block_name = join(dirname(__file__), "mnist_spatial.h5")

_default_name = join("mnist.h5")
_default_block_name = join("mnist_spatial.h5")

def get_store(fname=_default_name):
    fname = join(dirname(__file__), fname)
    print "Loading from store", fname
    return h5py.File(fname, 'r')


def build_store(store="mnist.h5", mnist="mnist.pkl.gz"):
    """Build a hdf5 data store for MNIST.
    """
    print "Reading", mnist
    mnist_f = gzip.open(mnist,'rb')
    train_set, valid_set, test_set = cPickle.load(mnist_f)
    mnist_f.close()

    print "Writing to", store
    h5file = h5py.File(store, "w")

    print "Creating train set."
    grp = h5file.create_group("train")
    dset = grp.create_dataset("inputs", data = train_set[0])
    dset = grp.create_dataset("targets", data = train_set[1])

    print "Creating validation set."
    grp = h5file.create_group("validation")
    dset = grp.create_dataset("inputs", data = valid_set[0])
    dset = grp.create_dataset("targets", data = valid_set[1])

    print "Creating test set."
    grp = h5file.create_group("test")
    dset = grp.create_dataset("inputs", data = test_set[0])
    dset = grp.create_dataset("targets", data = test_set[1])

    print "Closing", store
    h5file.close()


def build_block_store(block, block_store, store="mnist.h5"):
    '''
    Build a hdf5 data store with block view form for MNIST
    '''
    xs = (28, 28)
    helpers.blockify(h5py.File(store, 'r'),
                     h5py.File(block_store, 'w'),
                     xs, block, exclude=['targets'])


def visualize():
    if isfile(_default_name):
        f = get_store()
        data = f['train']['inputs']
        mat = data[:128]
        rmat = np.ravel(mat.astype(float))
        im = helpers.visualize(rmat, 28*28)
        im.save('mnist.png')


def visualize_bv(block, block_file):
    '''
    visualize the block view data
    '''
    if isfile(block_file):
        f = get_store(fname=block_file)
        data = f['train']['inputs']
        block_mat = data[:128]
        mat = helpers._batch_unblock_view(block_mat, (28, 28), block)
        rmat = np.ravel(mat.astype(float))
        im = helpers.visualize(rmat, 28*28)
        im.save('mnist_block.png')

if __name__=="__main__":
    if isfile(_default_name):
        print _default_name, 'exists'
    else:
        build_store()
    # visualize()
    bs = (4, 1)
    b_store = 'mnist_spatial_{}x{}.h5'.format(bs[0], bs[1])
    build_block_store(bs, b_store)
    # visualize_bv(bs, b_store)
