from product import *
from apis.product import displayProduct

if __name__ == '__main__':

# ------------------------
# Version 1: different visitors
# ------------------------
    # # test product cashflow
    # maturity_date = '2025-12-31'
    # currency = 'USD'
    # notional = 1e4
    # longOrShort = 'Long'
    # this_prod = ProductBulletCashflow(maturity_date, currency, notional, longOrShort)
    # this_displayer = CashflowVisitor()
    # print(this_prod.accept(this_displayer))

    # # test product cashflow
    # effective_date = '2025-12-31'
    # index = 'USD-LIBOR-BBA-3M'
    # strike = 98.
    # this_prod = ProductFuture(
    #     effective_date, 
    #     index , 
    #     strike,
    #     notional,
    #     longOrShort)
    # this_displayer = FutureVisitor()
    # print(this_prod.accept(this_displayer))

# ------------------------
# Version 2: new class based on different visitors
# ------------------------
    # cf = ProductBulletCashflow("2025-12-31", "USD", 1e4, "LONG")
    # print(cf.display()) 

# ------------------------
# Version 3: based on new product/product_display.py & apis/product
# ------------------------
    # single bullet cashflow
    cf = ProductBulletCashflow("2025-12-31", "USD", 1_000_000, "LONG")
    print(displayProduct(cf))

    # IborCashFlow
    # cf = ProductIborCashflow(
    #     "2025-01-01",            # startDate
    #     "2025-07-01",            # endDate
    #     "USD-LIBOR-BBA-3M",      # index
    #     0.0025,                  # spread
    #     5_000_000,               # notional
    #     "LONG"                   # longOrShort
    # )

    # print(displayProduct(cf))

    # Future
    fut = ProductFuture(
        effectiveDate="2025-03-20",
        index="USD-LIBOR-BBA-3M",
        strike=98.75,
        notional=1_000_000,
        longOrShort="SHORT",
        contractualSize=0.25
    )

    print(displayProduct(fut))
