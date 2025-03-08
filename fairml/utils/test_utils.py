# coding: utf-8

import pdb


def test_saver():
    from fairml.utils.utils_saver import (
        get_elogger, rm_ehandler, console_out,
        elegant_print)

    logger, fm, fh = get_elogger('logtest', 'filtest.txt')
    elegant_print('test print')
    elegant_print('test print log', logger)
    rm_ehandler(logger, fm, fh)

    import os
    os.remove('filtest.txt')
    # pdb.set_trace()
    return


def test_timer():
    from fairml.utils.utils_timer import (
        fantasy_durat, elegant_durat, elegant_dated,
        fantasy_durat_major)
    import time
    pi = 3.141592653589793

    for verb in [True, False]:
        print(fantasy_durat(pi - 3, verb))

        # print(fantasy_durat(pi, verb, False))
        # print(fantasy_durat(pi, verb, True))
        print(fantasy_durat_major(pi, verb, True))

        print(elegant_durat(pi, verb))  # same
        print(fantasy_durat_major(pi, verb, False))

    tim = time.time()
    print(elegant_dated(tim, 'num'))
    print(elegant_dated(tim, 'txt'))
    print(elegant_dated(tim, 'day'))
    print(elegant_dated(tim, 'wks'))

    # pdb.set_trace()
    return
