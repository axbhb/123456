from common import *
import runpy
runpy.run_path(str(W/('attack_advdiff_dd2.py' if G=='dd2' else 'attack_advdiff_native.py')),run_name='__main__')
