import pandas as pd
# native
from product import Product
from fixedincomelib.product.product_display_visitor import ProductDisplayVisitor

def displayProduct(product : Product):
    this_displayer = ProductDisplayVisitor()
    product.accept(this_displayer)
    return this_displayer.display()
