import os
import shutil

base_dir = './dataset/casia_mtcnnpy_224/'


def delIpCk(parent_path):
    for fname in os.listdir(parent_path):
        fpath = os.path.join(parent_path, fname)
        if os.path.isfile(fpath):
            continue
        if fname == '.ipynb_checkpoints':
            print('Deleting ', fpath)
            shutil.rmtree(fpath)
        else:
            delIpCk(fpath)

if __name__ == '__main__':
    delIpCk(base_dir)