# -*- coding: utf-8 -*-


"""
Author:	Samuel Garcia
Laboratoire de Neurosciences Sensorielles, Comportement, Cognition.
CNRS - UMR5020 - Universite Claude Bernard LYON 1
Equipe logistique et technique
50, avenue Tony Garnier
69366 LYON Cedex 07
FRANCE
sgarcia@olfac.univ-lyon1.fr

License: CeCILL v2 (GPL v2 compatible)

"""

import os

#------------------------------------------------------------------------------
def make_ressoureces_icons():
	fid = open('icons.qrc','w')
	fid.write("""<!DOCTYPE RCC><RCC version="1.0">
	<qresource>
"""			)
	for p, d, files in os.walk('./'):
		for filename in files:

			if filename[-3:] == 'png':
				fid.write('			<file alias="%s">%s/%s</file>\r\n' % (filename, p[2:],filename) )
				
	fid.write("""	</qresource>
</RCC>
"""			)

	fid.close()
	os.popen('pyrcc4 icons.qrc > icons.py')
	

#------------------------------------------------------------------------------
if __name__ == '__main__' :
	make_ressoureces_icons()

