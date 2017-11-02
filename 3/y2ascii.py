import re
import numpy as np

# __author__ = "Andreas S Strauman"

def y2ascii(Y, switch=False):
	""" Takes in Y in 1 and -1 labels
		>> Y.shape
		(N,)
		Switch will switch which label represents a 0 and which represents a 1
	 """
	if switch:
		lbls=["1" if y<0 else "0" for y in Y]
	else:
		lbls=["1" if y>0 else "0" for y in Y]
	string=""
	for i in np.arange(0,int(len(lbls)/7)):
		binstring=''.join(lbls[(i*7):(i*7)+7])[::-1]
		ascii=chr(int(binstring,2))
		string+=ascii if re.match(r'[a-z\.!]', ascii) else "#"
	return string
