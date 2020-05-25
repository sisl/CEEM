#
# File: n4sid.py
#


def compute_n4sid():
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.run_n4sid(nargout=0)


if __name__ == '__main__':
    compute_n4sid()
