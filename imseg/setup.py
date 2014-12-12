from distutils.core import setup, Extension
 
phi = Extension('phi', 
        sources = ['phi.c'],
        )

setup (name = 'phi',
        version = '0.01',
        description = 'optimized phi',
        ext_modules = [phi])
