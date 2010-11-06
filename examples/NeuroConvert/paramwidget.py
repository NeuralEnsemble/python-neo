# -*- coding: utf-8 -*-


from PyQt4.QtCore import *
from PyQt4.QtGui import *


import numpy
import datetime
import os
from copy import deepcopy

from icons import icons

#------------------------------------------------------------------------------
class ParamWidget(QFrame) :
	"""
	params is a tuple of all parameters with this king
	
	[
		(paramname1  ,  dictParam1  ),
		(paramname2  ,  dictParam2  ),
		(paramname3  ,  dictParam3  ),
		
	]
	
	dictParam can contains this keys:
	
		- default : the default param
		- type : out type
		- label : name to display
		- widgettype
		- allownone : None is lineedit give a NoneType
	
	"""
	
	def __init__(self, params  , parent = None ,
				keyformemory = None ,
				applicationdict = None,
				):
		QFrame.__init__(self, parent)
		
		
		self.keyformemory = keyformemory
		self.applicationdict = applicationdict
		
		self.setFrameStyle(QFrame.Raised | QFrame.StyledPanel)
		v1 = QVBoxLayout()
		self.setLayout(v1)
		
		if self.keyformemory is not None :
			stored_list = self.applicationdict['storedParameters/'+self.keyformemory]
			if stored_list is None :
				self.applicationdict['storedParameters/'+self.keyformemory] = [ ]
				stored_list = [ ]
			
			list_name = [ l[0] for l in stored_list ]
			
			h1 = QHBoxLayout()
			v1.addLayout(h1)
			self.comboParam = QComboBox()
			h1.addWidget(self.comboParam)
			self.comboParam.addItems( ['Default' , ]+list_name  )
			
			self.connect(self.comboParam,SIGNAL('currentIndexChanged( int  )') , self.comboParam_changed )
			buttonSave = QPushButton(QIcon(':/list-add.png') ,'')
			buttonSave.setMaximumSize(25,25)
			h1.addWidget(buttonSave)
			self.connect(buttonSave,SIGNAL('clicked()') ,self.saveNewParam )
			buttonDel = QPushButton(QIcon(':/list-remove.png') ,'')
			buttonDel.setMaximumSize(25,25)
			h1.addWidget(buttonDel)
			self.connect(buttonDel,SIGNAL('clicked()') ,self.delSavedParam )
		
		qg = QGridLayout()
		v1.addLayout(qg)
		
		self.list_widget = []
		self.list_name = [ p[0] for p in params ]
		self.params = dict(deepcopy(params))
		
		for i in range(len(self.list_name)) :
			name = self.list_name[i]
			
			
			if self.params[name].has_key('label'):
				qg.addWidget(QLabel( self.params[name]['label']),i,0)
			else :
				qg.addWidget(QLabel(name),i,0)
			
			if not(self.params[name].has_key('type')) and not(self.params[name].has_key('value')) :
				self.params[name]['type'] = str
				self.params[name]['value'] = ''
			elif self.params[name].has_key('type') and not(self.params[name].has_key('value'))  :
				self.params[name]['value'] = self.params[name]['type']()
			elif self.params[name].has_key('value')  and not(self.params[name].has_key('type')) :
				self.params[name]['type'] = type(self.params[name]['value'])
			
			if not( self.params[name].has_key('default') ):
				self.params[name]['default'] = self.params[name]['value']
			
			if self.params[name].has_key('widgettype'):
				pass
				# TODO : control compatibility widget-type
			else :
				if self.params[name]['type'] == bool :
					self.params[name]['widgettype'] = QCheckBox
				elif self.params[name].has_key('possible') :
					self.params[name]['widgettype'] = QComboBox
				else :
					self.params[name]['widgettype'] = QLineEdit
			
			#widget creation
			wid = self.params[name]['widgettype']()
			self.params[name]['widget'] = wid
			qg.addWidget(wid,i,1)
			
			
			# update dict
			self[name] = self.params[name]['value']
			
			# widget connection and specific custumisation
			if  self.params[name]['widgettype'] ==  QLineEdit :
				self.connect(wid, SIGNAL('textEdited( QString )'), self.oneParamChanged)
			elif self.params[name]['widgettype'] ==  QCheckBox:
				self.connect(wid, SIGNAL('stateChanged( int )'), self.oneParamChanged)
			elif self.params[name]['widgettype'] == QComboBox:
				self.connect(wid, SIGNAL('currentIndexChanged( int )'), self.oneParamChanged)
			elif self.params[name]['widgettype'] == LimitWidget:
				self.connect(wid, SIGNAL('limitChanged( )'), self.oneParamChanged)
				if self.params[name].has_key('decimals') :
					wid.setDecimals(self.params[name]['decimals'])
				else :
					wid.setDecimals(3)
				if self.params[name].has_key('singleStep') :
					wid.setSingleStep(self.params[name]['singleStep'])
				else :
					wid.setSingleStep(.1)
				
				
			elif self.params[name]['widgettype'] == MultiSpinBox	 :
				self.connect(wid, SIGNAL('oneValueChanged( )'), self.oneParamChanged)
			elif self.params[name]['widgettype'] == MultiComboBox  :
				pass
			elif self.params[name]['widgettype'] == ChooseColorWidget:
				self.connect(wid, SIGNAL('colorChanged( )'), self.oneParamChanged)				
			elif self.params[name]['widgettype'] == QDoubleSpinBox:
				self.connect(wid, SIGNAL('valueChanged ( double )'), self.oneParamChanged)
				wid.setMaximum(numpy.inf)
				wid.setMinimum(-numpy.inf)
				
				if self.params[name].has_key('decimals') :
					wid.setDecimals(self.params[name]['decimals'])
				else :
					wid.setDecimals(3)
				if self.params[name].has_key('singleStep') :
					wid.setSingleStep(self.params[name]['singleStep'])
				else :
					wid.setSingleStep(.1)
				
			elif (self.params[name]['widgettype'] == ChooseDirWidget) or (self.params[name]['widgettype'] == ChooseFileWidget):
				pass
			elif self.params[name]['widgettype'] == QTextEdit:
				pass
				
		
		#~ v1.addStretch(0)
		
	#------------------------------------------------------------------------------
	def __setitem__(self,key,val):
		if key in self.params.keys() :
			wid = self.params[key]['widget']
			
			if val is None:
				if  self.params[key]['widgettype'] ==  QLineEdit :
					wid.setText(unicode('None'))
				return
			#~ elif type(val) != self.params[key]['type']:
				#~ return
			
			if self.params[key]['widgettype'] ==  QCheckBox:
				wid.setChecked(val)
			
			elif self.params[key]['widgettype'] == QComboBox:
				if val in self.params[key]['possible'] :
					items = [ str(v) for v in self.params[key]['possible'] ]
					wid.clear()
					wid.addItems(items)
					i = self.params[key]['possible'].index(val)
					wid.setCurrentIndex (i)
			
			elif self.params[key]['widgettype'] == LimitWidget:
				wid.set_limit(val)
				
			elif self.params[key]['widgettype'] == MultiSpinBox :
				wid.set_listValue(val)
			
			elif self.params[key]['widgettype'] == MultiComboBox :
				wid.set_listValue(val , possible = self.params[key]['possible'])
			
			elif self.params[key]['widgettype'] == ChooseColorWidget:
				wid.setColor(val)
				
			elif self.params[key]['widgettype'] == QDoubleSpinBox:
				wid.setValue(val)
			
			elif (self.params[key]['widgettype'] == ChooseDirWidget) or (self.params[key]['widgettype'] == ChooseFileWidget):
				wid.set_choose(val)
			
			elif self.params[key]['widgettype'] == QTextEdit:
				wid.setPlainText(val)
			
			elif self.params[key]['widgettype'] ==  QLineEdit:
				
				if (self.params[key]['type'] == float) or (self.params[key]['type'] == numpy.float) :
					# patch for win32 for float(numpy.inf) -> '1.#IND'
					if numpy.isfinite(val) :
						wid.setText(unicode(val))
					else :
						list_infinite = [numpy.nan, numpy.inf, -numpy.inf]
						list_str_infinite = ['nan' , 'inf' , '-inf' ]
						if val in list_infinite :
							wid.setText(list_str_infinite[ list_infinite.index(val) ])
				else :
					wid.setText(unicode(val))
					
					
					
	#------------------------------------------------------------------------------
	def __getitem__(self,key):
		if key in self.params.keys() :
			
			wid = self.params[key]['widget']
			
			if self.params[key]['widgettype'] ==  QCheckBox:
				return self.params[key]['type'](wid.isChecked())
			
			elif self.params[key]['widgettype'] == QComboBox:

				i = wid.currentIndex()
				return self.params[key]['possible'][i]
			
			elif self.params[key]['widgettype'] == LimitWidget:
				return  wid.get_limit()
				
			elif self.params[key]['widgettype'] == MultiSpinBox :
				return  wid.get_listValue()
			
			elif self.params[key]['widgettype'] == MultiComboBox :
				return  wid.get_listValue()			
			
			elif self.params[key]['widgettype'] == ChooseColorWidget:
				return wid.getColor()
				
			elif self.params[key]['widgettype'] == QDoubleSpinBox:
				return wid.value()
			
			elif (self.params[key]['widgettype'] == ChooseDirWidget) or \
					(self.params[key]['widgettype'] == ChooseFileWidget) :
				return wid.get_choose()
			
			elif (self.params[key]['widgettype'] == ChooseFilesWidget) :
				return wid.get_files()
				
			elif self.params[key]['widgettype'] == QTextEdit:
				return wid.toPlainText()
				
			elif self.params[key]['widgettype'] ==  QLineEdit:
				text = wid.text()
				
				if text == 'None':
					return None
				
				elif (self.params[key]['type'] == float) or (self.params[key]['type'] == numpy.float) :
					list_infinite = [numpy.nan, numpy.inf, -numpy.inf]
					list_str_infinite = ['nan' , 'inf' , '-inf' ]
					if text in list_str_infinite :
						return list_infinite[list_str_infinite.index(text)]
					else :
						return self.params[key]['type']( text )
				
				elif self.params[key]['type'] == datetime.date :
					r = re.findall('(\d+)\-(\d+)\-(\d+)',str(text) )
					if len(r) <1 :
						return self.params[key]['value']
					else :
						YY , MM , DD =r[0]
						return datetime.date( int(YY) , int(MM) , int(DD) )
				elif self.params[key]['type'] == datetime.datetime :
					r = re.findall('(\d+)\-(\d+)\-(\d+) (\d+):(\d+):(\d+)(.\d*)?',str(text) )
					if len(r) <1 :
						return self.params[key]['value']
					else :
						YY , MM , DD , hh, mm , ss , ms =r[0]
						if ms =='' :
							return datetime.datetime(int(YY) , int(MM) , int(DD) , int(hh), int(mm) , int(ss) )
						else :
							return datetime.datetime(int(YY) , int(MM) , int(DD) , int(hh), int(mm) , int(ss) , int(ms[1:]))
				elif self.params[key]['type'] is type(None):
					return None
				else :
					return self.params[key]['type']( text )
		
	#------------------------------------------------------------------------------
	def update(self,dict_param):
		"""
		"""
		for k,v in dict_param.iteritems() :
			self[k] = v
		
	
	#------------------------------------------------------------------------------
	def get_dict(self):
		d = { }
		for k,v in self.params.iteritems() :
			d[k] = self[k]
		return d

	#------------------------------------------------------------------------------
	def oneParamChanged(self , *args , **kargs ):
		name = None
		for k,v in self.params.iteritems() :
			if v['widget'] == self.sender() :
				name = k
		if name is None : return
		self.emit(SIGNAL('paramChanged( QString )'), name)

	#------------------------------------------------------------------------------
	def comboParam_changed(self , pos) :
		if pos <= 0 :
			new_param = {}
			for name in self.list_name :
				new_param[name] = self.params[name]['default']
		else :
			stored_list = self.applicationdict['storedParameters/'+self.keyformemory]
			new_param = stored_list[pos-1][1]
		self.update(new_param)
		
	#------------------------------------------------------------------------------
	def saveNewParam( self ) :
		dia = ParamDialog( [ ('name' , { 'value'  : 'name' } )])
		ok = dia.exec_()
		if  ok !=  QDialog.Accepted: return

		name = dia.param_widget['name']
		stored_list = self.applicationdict['storedParameters/'+self.keyformemory]
		stored_list += [ [ name , self.get_dict() ] ]
		self.applicationdict['storedParameters/'+self.keyformemory] = stored_list

		self.comboParam.clear()
		list_name = [ l[0] for l in stored_list ]
		self.comboParam.addItems(['Default' , ]+list_name  )
		self.comboParam.setCurrentIndex(len(list_name))
		
	#------------------------------------------------------------------------------
	def delSavedParam( self) :
		pos = self.comboParam.currentIndex()
		if pos == 0: return
		
		stored_list = self.applicationdict['storedParameters/'+self.keyformemory]
		del stored_list[pos-1]
		self.applicationdict['storedParameters/'+self.keyformemory] = stored_list
			
		self.comboParam.clear()
		list_name = [ l[0] for l in stored_list ]
		self.comboParam.addItems(['Default' , ]+list_name  )
		self.comboParam.setCurrentIndex(0)
		


#------------------------------------------------------------------------------
class OldParamWidget( ParamWidget ) :
	"""
	Old ParamWidget API
	
	list_param : list of parameters name
	default_param : list of default value
	list_label : list of name
	
	"""
	def __init__(self, list_param , default_param , list_label = None  , parent = None ,
					 family=None, list_type=None ):
		
		params = [ ]
		
		for i,name in enumerate(list_param) :
			
			dict_param = {}
			dict_param['value'] = default_param[i]
			if list_label is not None:
				dict_param['label'] =  list_label[i]
			if list_type is not None :
				dict_param['type'] =  list_type[i]
			params.append( (name,dict_param) )
		
		ParamWidget.__init__(self, params ,parent = None , family=None)
	
	#------------------------------------------------------------------------------
	def set_one_param(self, param , val ) :
		self[param] = val
	#------------------------------------------------------------------------------
	def get_one_param(self, param ) :
		return self[param]
		
	#------------------------------------------------------------------------------
	def get_dict(self) :
		d ={}
		for i,key in enumerate(self.list_param) :
			d[key] = self.get_param()[i]
		return d





#------------------------------------------------------------------------------
class ParamDialog(QDialog):
	def __init__(self,  params  , parent = None ,
						keyformemory = None , 
						applicationdict = None,
						title=''):
		QDialog.__init__(self, parent)
		self.setWindowTitle (title)
		mainLayout = QVBoxLayout()
		self.param_widget = ParamWidget(params, keyformemory = keyformemory , applicationdict = applicationdict)
		mainLayout.addWidget(self.param_widget)
		h = QHBoxLayout()
		self.buttonOK = QPushButton(QIcon(':/dialog-ok-apply.png') ,'OK')
		h.addWidget(self.buttonOK)
		self.connect(self.buttonOK , SIGNAL('clicked()') , self , SLOT('accept()'))
		#self.buttonCancel = QPushButton('Cancel')
		#self.connect(self.buttonOK , SIGNAL('clicked()') , self , SLOT('reject()'))
		#h.addWidget(self.buttonCancel)
		mainLayout.addLayout(h)
		
		#~ self.connect(self , SIGNAL('accepted()') , self.ok)
		#~ self.connect(self , SIGNAL('rejected()') , self.cancel)
		
		self.setLayout(mainLayout)
		self.setModal(True)
	
	#~ def ok(self) :
		#~ pass
	#~ def cancel(self) :
		#~ pass


#------------------------------------------------------------------------------
# Widget for ajusting limits
#------------------------------------------------------------------------------
class LimitWidget(QWidget) :
	def __init__( self , parent = None , limit = [-10. , 10.] , linked = False) :
		QWidget.__init__(self , parent)
		mainLayout = QHBoxLayout()
		self.setLayout(mainLayout)
		
		mainLayout.addWidget(QLabel('Lower'))
		self.spinBoxLower = QDoubleSpinBox()
		mainLayout.addWidget(self.spinBoxLower)
		mainLayout.addWidget(QLabel('Upper'))
		self.spinBoxUpper = QDoubleSpinBox()
		mainLayout.addWidget(self.spinBoxUpper)
		self.checkBoxLink = QCheckBox('Link')
		mainLayout.addWidget(self.checkBoxLink)
		
		self.checkBoxLink.setChecked(linked)
		
		self.spinBoxLower.setMaximum(numpy.inf)
		self.spinBoxLower.setMinimum(-numpy.inf)
		self.spinBoxUpper.setMaximum(numpy.inf)
		self.spinBoxUpper.setMinimum(-numpy.inf)
		
		self.connect(self.spinBoxUpper, SIGNAL('valueChanged( double  )') , self.spinBoxUpperChanged )
		self.connect(self.spinBoxLower, SIGNAL('valueChanged( double  )') , self.spinBoxLowerChanged )
		
		self.set_limit(limit)
	
	#------------------------------------------------------------------------------
	def get_limit(self):
		return [ self.spinBoxLower.value(), self.spinBoxUpper.value() ]

	#------------------------------------------------------------------------------
	def set_limit(self, lim):
		self.spinBoxLower.setValue(lim[0])
		self.spinBoxUpper.setValue(lim[1])

	#------------------------------------------------------------------------------
	def setSingleStep(self, singleStep):
		self.spinBoxLower.setSingleStep( singleStep )
		self.spinBoxUpper.setSingleStep( singleStep )
		
	#------------------------------------------------------------------------------
	def setDecimals(self, decimals):
		self.spinBoxLower.setDecimals( decimals )
		self.spinBoxUpper.setDecimals( decimals )

	
	#------------------------------------------------------------------------------
	def spinBoxUpperChanged(self, val) : 
		if self.checkBoxLink.isChecked():
			self.spinBoxLower.setValue(-val)
		
		self.emit(SIGNAL('limitChanged( )'))
	
	#------------------------------------------------------------------------------
	def spinBoxLowerChanged(self, val) : 
		if self.checkBoxLink.isChecked():
			self.spinBoxUpper.setValue(-val)
		self.emit(SIGNAL('limitChanged( )'))




#------------------------------------------------------------------------------
# Widget multiple spinbox
#------------------------------------------------------------------------------
class MultiSpinBox(QWidget) :
	def __init__( self , parent = None , list_value = [ ], labels = None, linked = False) :
		QWidget.__init__(self , parent)
		mainLayout = QHBoxLayout()
		
		self.setLayout(mainLayout)
		
		
		w = QWidget()
		scrollArea = QScrollArea()
		scrollArea.setWidget(w)
		scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		#~ scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		scrollArea.setWidgetResizable(True)
		self.gl= QGridLayout()
		w.setLayout(self.gl)
		
		scrollArea.setMinimumHeight(90)
		
		self.list_spin = [ ]
		self.list_label = [ ]
		
		#~ mainLayout.addLayout(self.gl)
		mainLayout.addWidget(scrollArea)
		
		self.checkBoxLink = QCheckBox('Linked')
		mainLayout.addWidget(self.checkBoxLink)
		self.checkBoxLink.setChecked(linked)
		
		self.set_listValue(list_value , labels = labels)
	
	#------------------------------------------------------------------------------
	def oneSpinChanged(self, val):
		spin = self.sender()
		i = self.list_spin.index(spin)
		if self.checkBoxLink.isChecked():
			for sp in self.list_spin :
				sp.setValue(val)
		self.emit(SIGNAL('oneValueChanged( )'))
	
	#------------------------------------------------------------------------------
	def set_listValue(self, list_value , labels = None):

		# delete old one
		for spin in self.list_spin :
			spin.setVisible(False)
		for label in self.list_label :
			label.setVisible(False)
		self.list_spin = [ ]
		self.list_label = [ ]

		for i,val in enumerate(list_value):
			if labels is None:
				label = QLabel(str(i))
			else :
				label = QLabel(labels[i])
			self.list_label.append(label)
			self.gl.addWidget(label, 0, i)
			spin = QDoubleSpinBox()
			spin.setMaximum(numpy.inf)
			spin.setMinimum(-numpy.inf)
			spin.setValue(val)
			self.gl.addWidget(spin, 1, i)
			self.list_spin.append(spin)
			self.connect(spin, SIGNAL('valueChanged( double  )') , self.oneSpinChanged )
		
		#~ for i,val in enumerate(list_value):
			#~ self.list_spin[i].setValue(val)

	#------------------------------------------------------------------------------
	def get_listValue(self):
		list_value = [ ]
		for i,spin in enumerate(self.list_spin):
			list_value.append(spin.value())
		return list_value



#------------------------------------------------------------------------------
# Widget multiple comboobx
#------------------------------------------------------------------------------
class MultiComboBox(QWidget) :
	def __init__( self , parent = None , list_value = [ ],
						possible = [ ], labels = None ) :
		QWidget.__init__(self , parent)
		mainLayout = QHBoxLayout()
		
		self.setLayout(mainLayout)
		
		
		w = QWidget()
		scrollArea = QScrollArea()
		scrollArea.setWidget(w)
		scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		#~ scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		scrollArea.setWidgetResizable(True)
		self.gl= QGridLayout()
		w.setLayout(self.gl)
		
		scrollArea.setMinimumHeight(90)
		
		self.list_combo = [ ]
		self.list_label = [ ]
		
		mainLayout.addWidget(scrollArea)
		
		self.set_listValue(list_value , possible = possible, labels = labels)
		
	#------------------------------------------------------------------------------
	def set_listValue(self, list_value , possible = [], labels = None):

		# delete old one
		for combo in self.list_combo :
			combo.setVisible(False)
		for label in self.list_label :
			label.setVisible(False)
		self.list_combo = [ ]
		self.list_label = [ ]
		self.possible = possible

		for i,val in enumerate(list_value):
			if labels is None:
				label = QLabel(str(i))
			else :
				label = QLabel(labels[i])
			self.list_label.append(label)
			self.gl.addWidget(label, 0, i)
			
			combo = QComboBox()
			items = [ str(v) for v in possible ]
			combo.clear()
			combo.addItems(items)
			ind = possible.index(val)
			combo.setCurrentIndex(ind)
			self.gl.addWidget(combo, 1, i)
			self.list_combo.append(combo)
			#~ self.connect(spin, SIGNAL('valueChanged( double  )') , self.oneSpinChanged )
		

	#------------------------------------------------------------------------------
	def get_listValue(self):
		list_value = [ ]
		for i, combo in enumerate(self.list_combo):
			i = combo.currentIndex()
			list_value.append(self.possible[i])
			
		return list_value


#------------------------------------------------------------------------------
# Widget for choosing a color
#------------------------------------------------------------------------------
class ChooseColorWidget(QWidget) :
	def __init__( self , parent = None, color = QColor('green') ) :
		QWidget.__init__(self , parent)
		mainLayout = QHBoxLayout()
		self.setLayout(mainLayout)
		
		self.label = QLabel('')
		
		mainLayout.addWidget(self.label)
		bt_change = QPushButton(QIcon(':/applications-graphics.png') , '')
		bt_change.setMaximumSize(20,20)
		mainLayout.addWidget(bt_change)
		self.connect(bt_change,SIGNAL("clicked()"), self.change)
		
		self.color = QColor(color)
		self.refreshColor()
		
		self.label.setSizePolicy(QSizePolicy.Expanding , QSizePolicy.Minimum )
	
	#------------------------------------------------------------------------------
	def refreshColor(self):
		if self.color.isValid():
			self.label.setText(self.color.name())
			self.label.setPalette(QPalette(self.color))
			self.label.setAutoFillBackground(True)

	#------------------------------------------------------------------------------
	def change(self) :
		
		self.color = QColorDialog.getColor(QColor('red'))
		self.refreshColor()
		self.emit(SIGNAL('colorChanged( )'))
	
	#------------------------------------------------------------------------------
	def getColor(self):
		return self.color

	#------------------------------------------------------------------------------
	def setColor(self , color):
		self.color = color
		self.refreshColor()


#------------------------------------------------------------------------------
# Widget for choosing a File or a dir
#------------------------------------------------------------------------------
class ChooseFileDirWidget(QWidget) :
	def __init__( self , parent = None, family = None , type_choose = QFileDialog.Directory ) :
		QWidget.__init__(self , parent)
		self.family  = family
		self.type_choose = type_choose
		mainLayout = QHBoxLayout()
		self.setLayout(mainLayout)
		
		self.lineEdit = QLineEdit()
		mainLayout.addWidget(self.lineEdit)
		bt_change = QPushButton(QIcon(':/document-open-folder.png') , '')
		bt_change.setMaximumSize(20,20)
		mainLayout.addWidget(bt_change)
		self.connect(bt_change,SIGNAL("clicked()"), self.change)
		
		if self.family is not None :
			usersetting.add_family( self.family, { 'path' : ''} )
			path  = usersetting.load_last(self.family)['path']
			self.lineEdit.setText(path)
	
	#------------------------------------------------------------------------------
	def change(self) :
		fd = QFileDialog()
		fd.setAcceptMode(QFileDialog.AcceptOpen)
		fd.setFileMode(self.type_choose)
		fd.setDirectory(os.path.split(unicode(self.lineEdit.text()))[0])
		if (fd.exec_()) :
			fileNames = fd.selectedFiles()
			l = u''
			for f in  fileNames :
				 l += unicode(f)+u';'
			self.lineEdit.setText(l)
			#self.lineEdit.setText(str(fd.selectedFiles()[0]))
			
			if self.family is not None :
				usersetting.save_last(self.family , {'path' : unicode(fd.selectedFiles()[0]) } )
	#------------------------------------------------------------------------------
	def set_choose(self, path):
		self.lineEdit.setText(path)
	
	#------------------------------------------------------------------------------
	def get_choose(self) :
		return self.list_file()[0]
	
	
	#------------------------------------------------------------------------------
	def list_file(self) :
		l = unicode(self.lineEdit.text()).split(';')
		while '' in l :
			l.remove('')
		return l


#------------------------------------------------------------------------------
# Widget for choosing a directory
#------------------------------------------------------------------------------
class ChooseDirWidget(ChooseFileDirWidget) :
	def __init__( self , parent = None, family = None ) :
		ChooseFileDirWidget.__init__(self , parent = parent , family = family , type_choose = QFileDialog.Directory )
		
	#------------------------------------------------------------------------------
	def get_dir(self) :
		return self.get_choose()

#------------------------------------------------------------------------------
# Widget for choosing one file
#------------------------------------------------------------------------------
class ChooseFileWidget(ChooseFileDirWidget) :
	def __init__( self , parent = None, family = None ) :
		ChooseFileDirWidget.__init__(self , parent = parent , family = family , type_choose = QFileDialog.AnyFile )
		
	#------------------------------------------------------------------------------
	def get_file(self) :
		return self.get_choose()

#------------------------------------------------------------------------------
# Widget for choosing one file
#------------------------------------------------------------------------------
class ChooseFilesWidget(ChooseFileDirWidget) :
	def __init__( self , parent = None, family = None ) :
		ChooseFileDirWidget.__init__(self , parent = parent , family = family , type_choose = QFileDialog.ExistingFiles )
		
	#------------------------------------------------------------------------------
	def get_files(self) :
		return self.list_file()


