#! /bin/python3

from model import MaskRCNNWrapper, Config

if __name__ == "__main__":
    config = Config()
    wrapper = MaskRCNNWrapper(config)
    wrapper.compile_model()
