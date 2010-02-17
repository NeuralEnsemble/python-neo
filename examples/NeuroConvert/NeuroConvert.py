# -*- coding: utf-8 -*-
"""

NeuroConvert is a short GUI to illustrate neo.io module.

@author: sgarcia
"""



__version__ = "0.1"


import sys, os, time
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
        self.setMinimumSize( 500,400)
        
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
        self.table.setSortingEnabled(False)
        self.list_convert = [ ]

        but = QPushButton(QIcon(':/NeuroConvert.png') , self.tr("Start convertion")) 
        self.connect(but,SIGNAL("clicked()"), self.startConvertion)
        mainlayout.addWidget(but)
        
        self.convertionStarted = False


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
        dia = AddFileDialog()
        if dia.exec_() :
            for f in dia.files.get_dict()['fileList'] :
                self.list_convert.append({
                                        'filename' : f,
                                        'inputFormat' : dia.formats.get_dict()['inputFormat'],
                                        'outputFormat' : dia.formats.get_dict()['outputFormat'],
                                        'inputOptions' : dia.inputOptions.get_dict() ,
                                        'outputOptions' : dia.outputOptions.get_dict(),
                                        'convertOptions' : dia.convertOptions.get_dict(),
                                        'state' : 'not done',
                                        })
            self.refreshTable()
            
    def refreshTable(self):
        self.table.setRowCount( len(self.list_convert) )
        for r,convert in enumerate(self.list_convert) :
            self.table.setItem(r,0,QTableWidgetItem(os.path.basename(convert['filename']), 0 ))
            self.table.setItem(r,1,QTableWidgetItem(convert['inputFormat'], 0 ))
            self.table.setItem(r,2,QTableWidgetItem(convert['outputFormat'], 0 ))
            self.table.setItem(r,3,QTableWidgetItem(convert['state'], 0 ))
        self.table.resizeColumnsToContents()
            
    
    def startConvertion(self):
        if self.convertionStarted : return
        self.convertionStarted = True
        self.thread = ThreadConvertion(self)
        self.thread.start()
        self.connect(self.thread , SIGNAL('one more') , self.refreshTable)


class ThreadConvertion(QThread):
    def __init__(self  , parent):
        QThread.__init__(self , parent)
        
    def run(self):
        
        for convert in self.parent().list_convert :
            if convert['state'] != 'not done' : continue
            
            convert['state'] = 'in progress'
            self.emit(SIGNAL('one more'))
            print 'start convertion' , convert['filename']
            
            # read instance
            input = dict_format[convert['inputFormat']]['class'](filename = convert['filename']  )
            try :
                readed = input.read( **convert['inputOptions'] )
            except :
                convert['state'] = 'read error : check input options'
                self.emit(SIGNAL('one more'))
                continue
            
            # write instance : 4 case supported for the moment
            output = dict_format[convert['outputFormat']]['class']()
            filename = os.path.splitext(convert['filename'])[0]+'.'+output.extensions[0]
            if not(convert['convertOptions']['overwrite']):
                if os.path.exists(filename) :
                    convert['state'] = 'file exits : force overwrite'
                    self.emit(SIGNAL('one more'))
                    continue
            output.filename = filename
            
            print 'out',filename
            try :
                if type(readed)== neo.Segment :
                    if neo.Segment in output.writeable_objects :
                        print 'segment to segment'
                        output.write_segment( readed,
                            filename = filename , **convert['outputOptions'] )
                        convert['state'] = 'OK'
                        self.emit(SIGNAL('one more'))
                        
                    elif neo.Block in output.writeable_objects :
                        print 'segment to block'
                        block = Block()
                        block._segments.append( readed )
                        output.write_block( block,
                            filename = filename , **convert['outputOptions'] )
                        convert['state'] = 'OK'
                        self.emit(SIGNAL('one more'))
                    
                elif type(readed)== neo.Block :
                    if neo.Segment in output.writeable_objects :
                        print 'block to segment'
                        
                        if convert['convertOptions']['blockToMultipleSegment'] :
                            for s,seg in enumerate(readed.get_segments()):
                                filename = os.path.splitext(convert['filename'])[0]+'%3d'%s+'.'+output.extensions[0]
                                if not(convert['convertOptions']['overwrite']):
                                    if os.path.exists(filename) :
                                        convert['state'] = 'file exits : force overwrite '
                                        self.emit(SIGNAL('one more'))
                                        continue
                                output.write_segment( seg,
                                                    **convert['outputOptions'] )
                                convert['state'] = 'OK'
                                self.emit(SIGNAL('one more'))
                        else :
                            # write only first segment
                            output.write_segment( readed.get_segments[0],
                                    filename = filename , **convert['outputOptions'] )
                            convert['state'] = 'OK'
                            self.emit(SIGNAL('one more'))
                            
                                
                    elif neo.Block in output.writeable_objects :
                        print 'block to block'
                        output.write_block( readed,
                            filename = filename , **convert['outputOptions'] )
                        convert['state'] = 'OK'
                        self.emit(SIGNAL('one more'))
            
            except :
                convert['state'] = 'write error : check output options'
                self.emit(SIGNAL('one more'))
                continue
            
        self.parent().convertionStarted = False


class AddFileDialog(QDialog) :
    def __init__(self, parent = None,) :
        QDialog.__init__(self, parent)
        
        self.setMinimumSize( 600,800)
        self.setWindowIcon(QIcon(':/NeuroConvert.png'))
        
        mainlayout = QVBoxLayout()
        self.setLayout(mainlayout)
        
        
        # input format
        g = QGroupBox()
        g.setTitle('Files format')
        mainlayout.addWidget(g)
        v = QVBoxLayout()
        g.setLayout(v)
        param = [
                    ('inputFormat' , { 'value' :  possibleInput[0] , 
                                        'possible' : possibleInput  } ),
                    ('outputFormat' , { 'value' :  possibleOutput[0] ,
                                       'possible' : possibleOutput  } ),
                ]
        self.formats = ParamWidget(param)
        v.addWidget( self.formats )
        self.connect(self.formats , SIGNAL('paramChanged( QString )'), self.changeIOFormat )
        
        
        # files selector
        g = QGroupBox()
        g.setTitle('Files selection')
        mainlayout.addWidget(g)
        v = QVBoxLayout()
        g.setLayout(v)        
        param = [('fileList' , { 'value' :  '~' ,  'widgettype' : ChooseFilesWidget }     ), ]
        self.files = ParamWidget(param)
        v.addWidget( self.files )
        
        
        #inputOptions
        g = QGroupBox()
        g.setTitle('Input options')
        mainlayout.addWidget(g)
        v = QVBoxLayout()
        g.setLayout(v)  
        self.widgetInput = QWidget()
        self.widgetInput.setLayout(QHBoxLayout())
        v.addWidget( self.widgetInput )
        self.inputOptions = None
        
        #output options 
        g = QGroupBox()
        g.setTitle('Output options')
        mainlayout.addWidget(g)
        v = QVBoxLayout()
        g.setLayout(v)          
        self.widgetOutput = QWidget()
        self.widgetOutput.setLayout(QHBoxLayout())
        v.addWidget( self.widgetOutput )
        self.outputOptions = None

        
        # convertions options
        g = QGroupBox()
        g.setTitle('Convertion options')
        mainlayout.addWidget(g)
        v = QVBoxLayout()
        g.setLayout(v)           
        param = [ ('blockToMultipleSegment' ,
                       { 'value' :  True ,
                        'label' : ' If input is block convert to many segment (otherwise take first one)' 
                        } ),
                ('overwrite' ,
                       { 'value' :  False ,
                        'label' : ' Overwrite if file destination already exsits' 
                        } ),
                 
                 ]
        self.convertOptions = ParamWidget(param)
        v.addWidget( self.convertOptions )

        but = QPushButton(QIcon(':/dialog-ok-apply.png') , self.tr("OK"))
        self.connect(but,SIGNAL("clicked()"), self , SLOT('accept()'))
        mainlayout.addWidget(but)


        self.changeIOFormat('')
        
    def changeIOFormat(self , name) :
        
        if self.inputOptions is not None :
            self.inputOptions.hide()
            self.widgetInput.layout().removeWidget( self.inputOptions )
        formatname = self.formats['inputFormat']
        cl = dict_format[formatname]['class']
        param = cl.read_params[cl.readable_objects[0]]
        self.inputOptions = ParamWidget( param )
        self.widgetInput.layout().addWidget( self.inputOptions )

        if self.outputOptions is not None :
            self.outputOptions.hide()
            self.widgetOutput.layout().removeWidget( self.outputOptions )
        formatname = self.formats['outputFormat']
        cl = dict_format[formatname]['class']
        param = cl.write_params[cl.writeable_objects[0]]
        self.outputOptions = ParamWidget( param )
        self.widgetOutput.layout().addWidget( self.outputOptions )


if __name__ == '__main__' :
    mw =MainWindow()
    mw.show()
    sys.exit(app.exec_())



