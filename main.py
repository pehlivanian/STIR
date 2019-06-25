from __future__ import absolute_import
import logging
import os
from argparse import ArgumentParser
from datetime import datetime

def build_parser():
    parser = ArgumentParser(prog="Driver for STIR library")
    parser.add_argument("--mode", dest="mode",
                        help="Spreads or GSCI",
                        metavar="MODE", default="GSCI")
    parser.add_argument("--action", dest="action",
                        help="simulate, clear_summ_tables, plot",
                        metavar="ACTION", default="plot")
    parser.add_argument("--plot_mode",dest="plot_mode",
                        help="SERIAL or COMBINED",
                        default="COMBINED")
    parser.add_argument("--filename", dest="filename",
                        help="filename for plotting",
                        metavar="FILENAME", default="./fig_plot.pdf")
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if options.mode == 'GSCI':
        
        if options.action == 'simulate':
            import src
            src.backtest()
        elif options.action == 'clear_summ_tables':
            pass
        elif options.action == 'plot':
            import src
            if options.plot_mode == 'COMBINED':
                src.plot_GSCI_book_combined(filename=options.filename)
            else:
                src.plot_GSCI_book(filename=options.filename)
    else:
        pass
    

if __name__ == '__main__':
    main()
