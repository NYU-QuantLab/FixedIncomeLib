from apis import *
from product import *

if __name__ == '__main__':

    # test product cashflow
    maturity_date = '2025-12-31'
    currency = 'USD'
    notional = 1e4
    longOrShort = 'Long'
    this_prod = ProductBulletCashflow(maturity_date, currency, notional, longOrShort)
    print(displayProduct(this_prod))