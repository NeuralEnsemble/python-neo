# -*- coding: utf-8 -*-
"""

NeuroConvert is a short GUI to illustrate neo.io module.

@author: sgarcia
"""







import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

if __name__ == '__main__' :
	app = QApplication(sys.argv)


import os


from icons import icons

class MainWindow(QMainWindow) :
	def __init__(self, parent = None,) :
		QMainWindow.__init__(self, parent)
        
		# Layout
		self.setWindowTitle(self.tr('NeuroConvert'))
		self.setWindowIcon(QIcon(':/NeuroConvert.png'))



		self.createActions()
		self.createMenus()
		#self.createToolBars()
	#------------------------------------------------------------------------------
	def createActions(self):

		self.aboutAct = QAction(self.tr("&About"), self)
        self.quitAct.setShortcut(self.tr("Ctrl+Q"))
		self.aboutAct.setStatusTip(self.tr("Show the application's About box"))
		self.aboutAct.setIcon(QIcon(':/help-about.png'))
		self.connect(self.aboutAct,SIGNAL("triggered()"), self.about)

	#------------------------------------------------------------------------------
	def createMenus(self):
		self.fileMenu = self.menuBar().addMenu(self.tr("&File"))
#		self.fileMenu.addAction(self.changeSetupAct)
#		self.fileMenu.addSeparator()
#		self.fileMenu.addAction(self.quitAct)

		self.menuBar().addSeparator()

		self.helpMenu = self.menuBar().addMenu(self.tr("&Help"))
		self.helpMenu.addAction(self.aboutAct)

#------------------------------------------------------------------------------
	def about(self):
		QMessageBox.about(self, self.tr("About Dock Widgets"),
				self.tr("""<b>NeuroConvert</b> : 
                
				a modulable scope for monitoring electrophysiological signal. 
				Software : <b>Samuel GARCIA</b>
				Hardware : <b>Belkacem Messaoudi</b>
				Neurosciences Sensorielles, Comportement, Cognition. CNRS
				Lyon, France
				"""
				))




if __name__ == '__main__' :
	MainWindow
	
	

	mw =MainWindow()
	mw.show()
	
	
	sys.exit(app.exec_())



