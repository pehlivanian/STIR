from __future__ import absolute_import
import logging
import os
from data import products
from argparse import ArgumentParser
from datetime import datetime

def build_parser():
    parser = ArgumentParser(prog="Driver for STIR library")
    parser.add_argument("-m","--mode", dest="mode",
                        help="Spreads or GSCI",
                        metavar="MODE", default="GSCI")
    parser.add_argument("-a", "--action", dest="action",
                        help="simulate, clear_summ_tables, plot, plot_by_product, populate_data",
                        metavar="ACTION", default="plot")
    parser.add_argument("-p", "--product", dest="product",
                        help="product for by-product actions",
                        metavar="PRODUCT", default=None)
    parser.add_argument("-l", "--plot_mode",dest="plot_mode",
                        help="SERIAL or COMBINED",
                        default="COMBINED")
    parser.add_argument("-f", "--filename", dest="filename",
                        help="filename for plotting",
                        metavar="FILENAME", default="./fig_plot.pdf")
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if options.action == 'populate_data':
        import src
        src.populate_data()
    if options.mode == 'GSCI':      
        if options.action == 'simulate':
            import src
            src.GSCI_backtest()
        elif options.action == 'clear_summ_tables':
            pass
        elif options.action == 'plot':
            import src
            if options.plot_mode == 'COMBINED':
                src.plot_GSCI_book_combined(filename=options.filename)
            else:
                src.plot_GSCI_book(filename=options.filename)
    else:
        if options.action == 'plot':
            import src
            src.plot_book()
        if options.action == 'plot_by_product':
            import src
            filename = options.filename or './figs/all_{}_spreads.pdf'.format(product)
            src.plot_book_by_product(options.product, filename=filename)
        if options.action == 'simulate_slice':
            import src
            src.seasonal_slice_backtest()
        if options.action == 'simulate_strategy':
            import src
            src.seasonal_strategy_backtest()
    
if __name__ == '__main__':
    main()
