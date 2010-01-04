# -*- coding: utf-8 -*-
"""

NeuroConvert is a short GUI to illustrate neo.io module.

@author: sgarcia
"""



__version__ = "0.1"



import sys, os
from PyQt4.QtCore import *
from PyQt4.QtGui import *

if __name__ == '__main__' :
    app = QApplication(sys.argv)

# personnal GUI utilitliies
from icons import icons
from paramwidget import ParamWidget , ChooseFilesWidget

# Change this of course if neo is somewhere else
sys.path.append(os.path.abspath('../..'))
import neo
print neo.io.all_format

# constructing possibles input and output
possibleInput = [ ]
possibleOutput = [ ]
dict_format = { }
for name,format in neo.io.all_format :
    if format['class'].is_readable :
        possibleInput.append(name)
    if format['class'].is_writable :
        possibleOutput.append(name)
    dict_format[name] = format


class MainWindow(QMainWindow) :
    def __init__(self, parent = None,) :
        QMainWindow.__init__(self, parent)

        self.setWindowTitle(self.tr('NeuroConvert'))
        self.setWindowIcon(QIcon(':/NeuroConvert.png'))

        self.createActions()
        self.createMenus()
        
        w = QWidget(self)
        mainlayout = QVBoxLayout()
        w.setLayout(mainlayout)
        self.setCentralWidget(w)
        
        but = QPushButton(QIcon(':/list-add.png') , self.tr("Add files to convert"))
        self.connect(but,SIGNAL("clicked()"), self.addNewFile)
        mainlayout.addWidget(but)
        
        self.table = QTableWidget( 0 , 4 )
        mainlayout.addWidget(self.table)
        self.table.setHorizontalHeaderLabels([  'Input file',
                                                'Input format',
                                                'Outpout format',
                                                'State',
                                                ])
        self.table.resizeColumnsToContents()

        but = QPushButton(QIcon(':/NeuroConvert.png') , self.tr("Start convertion")) 
        self.connect(but,SIGNAL("clicked()"), self.startConvertion)
        mainlayout.addWidget(but)


    def createActions(self):

        self.aboutAct = QAction(self.tr("&About"), self)
        self.aboutAct.setShortcut(self.tr("Ctrl+A"))
        self.aboutAct.setStatusTip(self.tr("Show the application's About box"))
        self.aboutAct.setIcon(QIcon(':/help-about.png'))
        self.connect(self.aboutAct,SIGNAL("triggered()"), self.about)
        
        self.addAct = QAction(self.tr("Add Files"), self)
        self.addAct.setShortcut(self.tr("Ctrl+F"))
        self.addAct.setIcon(QIcon(':/list-add.png'))
        self.addAct.setStatusTip(self.tr("Add files to convert"))
        self.addAct.connect(self.addAct,SIGNAL("triggered()"), self.addNewFile)
    
        self.startAct = QAction(self.tr("Start convertion"), self)
        self.startAct.setShortcut(self.tr("Ctrl+S"))
        self.startAct.setIcon(QIcon(':/NeuroConvert.png'))
        self.startAct.setStatusTip(self.tr("Start convertion"))
        self.startAct.connect(self.startAct,SIGNAL("triggered()"), self.startConvertion)
        
    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu(self.tr("&File"))
        self.fileMenu.addAction(self.addAct)
        self.fileMenu.addAction(self.startAct)

        self.menuBar().addSeparator()

        self.helpMenu = self.menuBar().addMenu(self.tr("&Help"))
        self.helpMenu.addAction(self.aboutAct)

    def about(self):
        QMessageBox.about(self, self.tr("About Dock Widgets"),
                self.tr("""<b>NeuroConvert %s</b>
                <p> A short GUI to test neo.io module
                
                <p>Copyright &copy; 2009 Samuel Garcia
                <p>Neurosciences Sensorielles, Comportement, Cognition. CNRS
                Lyon, France
                """
                %
                (__version__)
                ))

    def addNewFile(self):
        print 'yep'
        
        dia = AddFileDialog()
        if dia.exec_() :
            print 'ok'
            
            
    
    def startConvertion(self):
        pass



class AddFileDialog(QDialog) :
    def __init__(self, parent = None,) :
        QDialog.__init__(self, parent)
        
        mainlayout = QVBoxLayout()
        self.setLayout(mainlayout)
        self.tab = QTabWidget(self)
        mainlayout.addWidget(self.tab)
        
        but = QPushButton(QIcon(':/dialog-ok-apply.png') , self.tr("OK"))
        self.connect(but,SIGNAL("clicked()"), self , SLOT('accept()'))
        mainlayout.addWidget(but)
        
        
        # input format tab
        param = [
                    ('inputFormat' , { 'value' :  possibleInput[0] , 
                                        'possible' : possibleInput  } ),
                    ('outputFormat' , { 'value' :  possibleOutput[0] ,
                                       'possible' : possibleOutput  } ),
                ]
        self.formats = ParamWidget(param)
        self.tab.addTab(self.formats,'Files format')
        self.connect(self.formats , SIGNAL('paramChanged( QString )'), self.changeIOFormat )
        
        
        # files selector tab
        param = [('fileList' , { 'value' :  '~' ,  'widgettype' : ChooseFilesWidget }     ), ]
        self.files = ParamWidget(param)
        self.tab.addTab(self.files,'Input files')
        
        
        self.frameInput = QFrame() # HERE
        self.tab.addTab(self.frameInput,'Input options')
        
        self.frameOutput = QFrame() # HERE
        self.tab.addTab(self.frameOutput,'Output options')
        
        # convertions options
        param = [ ('blockToMultipleSegment' ,
                       { 'value' :  True ,
                        'label' : ' If input is block convert to many segment' 
                        } ),
                 
                 ]
        self.convertOptions = ParamWidget(param)
        self.tab.addTab(self.convertOptions,'Convertion options')
        
        self.changeIOFormat('')
        
    def changeIOFormat(self , name) :
            
        formatname = self.formats['inputFormat']
        cl = dict_format[formatname]['class']
        param = cl.read_params[cl.supported_types[0]]
        self.inputOption = ParamWidget( param )
        self.frameInput.setWidget( self.inputOption ) # HERE
            
        formatname = self.formats['outputFormat']
        cl = dict_format[formatname]['class']
        param = cl.read_params[cl.supported_types[0]]
        self.outputOption = ParamWidget( param )
        self.frameOutput.setWidget( self.outputOption ) # HERE


if __name__ == '__main__' :
    mw =MainWindow()
    mw.show()
    sys.exit(app.exec_())



