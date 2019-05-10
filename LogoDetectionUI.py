import sys
import os
import time
import cv2
import glob
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow
from object_detection import *

global dem
dem=0

class Team_Champion(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'NHAN DIEN THUONG HIEU'
        self.model_path = 'fineturning/final_model.ckpt'
        self.left = 240
        self.top = 200
        self.width = 800
        self.height = 450
        self.initUI()
        self.loadModel()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # 4 BUTTON ben trai
        #BUTTON OPEN FOLDER
        self.btn1= QPushButton("OPEN FOLDER",self)
        self.btn1.clicked.connect(self.Open_Folder)
        self.btn1.setGeometry(20,20,120,60)

        #BUTTON OPEN FILE
        self.btn2 = QPushButton("OPEN FILE", self)
        self.btn2.clicked.connect(self.Open_File)
        self.btn2.setGeometry(20, 90, 120, 60)

        # BUTTON EXIT
        self.btn3=QPushButton("EXIT", self)
        self.btn3.clicked.connect(self.EXIT)
        self.btn3.setGeometry(20, 160, 120, 60)

        # BUTTON DETECT:
        self.btn4=QPushButton('DETECT!',self)
        self.btn4.clicked.connect(self.Detect)
        self.btn4.setGeometry(20, 280, 120, 60)
        # PHAN DAT ANH: 4X4 MOI ANH CO kICH THUOC 80x80 Pixel.
        #Anh 1

        self.anh1 = QPushButton(self)

        self.anh1.setGeometry(190, 20, 80, 80)
        self.anh1.setIcon(QIcon('sami/1.png'))
        self.anh1.setIconSize(QSize(80, 80))
        #Anh 2
        self.anh2=QPushButton(self)
        self.anh2.setGeometry(280, 20, 80, 80)
        self.anh2.setIcon(QIcon('sami/2.png'))
        self.anh2.setIconSize(QSize(80, 80))
        #Anh 3
        self.anh3 = QPushButton(self)
        self.anh3.setGeometry(370, 20, 80, 80)
        self.anh3.setIcon(QIcon('sami/3.png'))
        self.anh3.setIconSize(QSize(80, 80))
        #Anh 4
        self.anh4 = QPushButton(self)
        self.anh4.setGeometry(460, 20, 80, 80)
        self.anh4.setIcon(QIcon('sami/4.png'))
        self.anh4.setIconSize(QSize(80, 80))
        #Anh 5
        self.anh5 = QPushButton(self)
        self.anh5.setGeometry(190, 110, 80, 80)
        self.anh5.setIcon(QIcon('sami/5.png'))
        self.anh5.setIconSize(QSize(80, 80))
        #Anh 6
        self.anh6 = QPushButton(self)
        self.anh6.setGeometry(280, 110, 80, 80)
        self.anh6.setIcon(QIcon('sami/6.png'))
        self.anh6.setIconSize(QSize(80, 80))
        # Anh 7
        self.anh7 = QPushButton(self)
        self.anh7.setGeometry(370, 110, 80, 80)
        self.anh7.setIcon(QIcon('sami/7.png'))
        self.anh7.setIconSize(QSize(80, 80))
        # Anh 8
        self.anh8 = QPushButton(self)
        self.anh8.setGeometry(460, 110, 80, 80)
        self.anh8.setIcon(QIcon('sami/8.png'))
        self.anh8.setIconSize(QSize(80, 80))
        #Anh 9
        self.anh9 = QPushButton(self)
        self.anh9.setGeometry(190, 200, 80, 80)
        self.anh9.setIcon(QIcon('sami/9.png'))
        self.anh9.setIconSize(QSize(80, 80))
        # Anh 10
        self.anh10 = QPushButton(self)
        self.anh10.setGeometry(280, 200, 80, 80)
        self.anh10.setIcon(QIcon('sami/10.png'))
        self.anh10.setIconSize(QSize(80, 80))
        # Anh 11
        self.anh11 = QPushButton(self)
        self.anh11.setGeometry(370, 200, 80, 80)
        self.anh11.setIcon(QIcon('sami/11.png'))
        self.anh11.setIconSize(QSize(80, 80))
        # Anh 12
        self.anh12 = QPushButton(self)
        self.anh12.setGeometry(460, 200, 80, 80)
        self.anh12.setIcon(QIcon('sami/12.png'))
        self.anh12.setIconSize(QSize(80, 80))
        #Anh 13
        self.anh13 = QPushButton(self)
        self.anh13.setGeometry(190, 290, 80, 80)
        self.anh13.setIcon(QIcon('sami/13.png'))
        self.anh13.setIconSize(QSize(80, 80))
        # Anh 14
        self.anh14 = QPushButton(self)
        self.anh14.setGeometry(280, 290, 80, 80)
        self.anh14.setIcon(QIcon('sami/14.png'))
        self.anh14.setIconSize(QSize(80, 80))
        # Anh 15
        self.anh15 = QPushButton(self)
        self.anh15.setGeometry(370, 290, 80, 80)
        self.anh15.setIcon(QIcon('sami/15.png'))
        self.anh15.setIconSize(QSize(80, 80))
        # Anh 16
        self.anh16 = QPushButton(self)
        self.anh16.setGeometry(460, 290, 80, 80)
        self.anh16.setIcon(QIcon('sami/16.png'))
        self.anh16.setIconSize(QSize(80, 80))
        #Truoc
        self.truoc = QPushButton(self)
        self.truoc.clicked.connect(self.Truoc)
        self.truoc.setGeometry(200, 400, 100, 30)
        self.truoc.setText('Truoc')
        # Ke tiep:
        self.tiep=QPushButton(self)
        self.tiep.clicked.connect(self.Tiep)
        self.tiep.setGeometry(450, 400, 100, 30)
        self.tiep.setText('Tiep')
        # Phan Load More

        # Phan 3 ngan hang
        # Viettin bank
        self.Viettinbank = QPushButton(self)
        self.Viettinbank.clicked.connect(self.viettinbank_sukien)
        self.Viettinbank.setGeometry(600, 30, 100, 30)
        self.Viettinbank.setText('Viettinbank')
        # Vietcombank
        self.Vietcombank = QPushButton(self)
        self.Vietcombank.clicked.connect(self.vietcombank_sukien)
        self.Vietcombank.setGeometry(600, 80, 100, 30)
        self.Vietcombank.setText('Vietcombank')
        #BIDV
        self.BIDV = QPushButton(self)
        self.BIDV.clicked.connect(self.bidv_sukien)
        self.BIDV.setGeometry(600, 130, 100, 30)
        self.BIDV.setText('BIDV')
        self.show()


    def loadModel(self):
        print('Loading model parameters ...')
        self.detector = RPN()
        self.detector.build_net()
        self.detector.session.run([tf.global_variables_initializer()])
        self.detector.saver.restore(self.detector.session, self.model_path)
        print('Loading model complete !')

    @pyqtSlot()
    def Open_Folder(self):

        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        print(self.folder)
        self.Tap_anh=sorted(glob.glob(self.folder + '/*.jpg') + glob.glob(self.folder + '/*.png'))
        #time.sleep(0.4)

        print(self.Tap_anh)
        if len(self.Tap_anh)%16==0:
            self.thuong_16 = len(self.Tap_anh) // 16
        else:
            self.thuong_16 = len(self.Tap_anh) // 16+1
        print(self.thuong_16)
        self.thuong_16=0
        self._vietin = []
        self._vietcom = []
        self._bidv = []
        self.Reset_anh()


    def Open_File(self):
        pass

    def EXIT(self):
        QApplication.quit()
    def Detect(self):
        self.detected_vietin = []
        self.detected_vietcom = []
        self.detected_bidv = []
        self._vietin, self._vietcom, self._bidv = self.detector.predict_logo_trademark(self.Tap_anh)   

    def Truoc(self):
        self.thuong_16-=1
        self.Reset_anh()
    def Tiep(self):
        self.thuong_16+=1
        self.Reset_anh()
    def Reset_anh_sami(self):
        try:
            self.anh1.setIcon(QIcon('sami/1.png'))
            self.anh2.setIcon(QIcon('sami/2.png'))
            self.anh3.setIcon(QIcon('sami/3.png'))
            self.anh4.setIcon(QIcon('sami/4.png'))
            self.anh5.setIcon(QIcon('sami/5.png'))
            self.anh6.setIcon(QIcon('sami/6.png'))
            self.anh7.setIcon(QIcon('sami/7.png'))
            self.anh8.setIcon(QIcon('sami/8.png'))
            self.anh9.setIcon(QIcon('sami/9.png'))
            self.anh10.setIcon(QIcon('sami/10.png'))
            self.anh11.setIcon(QIcon('sami/11.png'))
            self.anh12.setIcon(QIcon('sami/12.png'))
            self.anh13.setIcon(QIcon('sami/13.png'))
            self.anh14.setIcon(QIcon('sami/14.png'))
            self.anh15.setIcon(QIcon('sami/15.png'))
            self.anh16.setIcon(QIcon('sami/16.png'))
        except:
            pass
    def Reset_anh(self):
        try:
            self.anh1.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+0]))
            self.anh2.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+1]))
            self.anh3.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+2]))
            self.anh4.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+3]))
            self.anh5.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+4]))
            self.anh6.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+5]))
            self.anh7.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+6]))
            self.anh8.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+7]))
            self.anh9.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+8]))
            self.anh10.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+9]))
            self.anh11.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+10]))
            self.anh12.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+11]))
            self.anh13.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+12]))
            self.anh14.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+13]))
            self.anh15.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+14]))
            self.anh16.setIcon(QIcon(self.Tap_anh[self.thuong_16*16+15]))
        except:
            pass
        #------------------------------------------------------------------
        try:
            self.anh1.clicked.connect(self.Dis_play_anh1)
            self.anh2.clicked.connect(self.Dis_play_anh2)
            self.anh3.clicked.connect(self.Dis_play_anh3)
            self.anh4.clicked.connect(self.Dis_play_anh4)
            self.anh5.clicked.connect(self.Dis_play_anh5)
            self.anh6.clicked.connect(self.Dis_play_anh6)
            self.anh7.clicked.connect(self.Dis_play_anh7)
            self.anh8.clicked.connect(self.Dis_play_anh8)
            self.anh9.clicked.connect(self.Dis_play_anh9)
            self.anh10.clicked.connect(self.Dis_play_anh10)
            self.anh11.clicked.connect(self.Dis_play_anh11)
            self.anh12.clicked.connect(self.Dis_play_anh12)
            self.anh13.clicked.connect(self.Dis_play_anh13)
            self.anh14.clicked.connect(self.Dis_play_anh14)
            self.anh15.clicked.connect(self.Dis_play_anh15)
            self.anh16.clicked.connect(self.Dis_play_anh16)
        except:
            pass

    def Dis_play_anh1(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+0])
        cv2.imshow('GK61',f)
    def Dis_play_anh2(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+1])
        cv2.imshow('GK61', f)
    def Dis_play_anh3(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+2])
        cv2.imshow('GK61', f)
    def Dis_play_anh4(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+3])
        cv2.imshow('GK61', f)
    def Dis_play_anh5(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+4])
        cv2.imshow('GK61', f)
    def Dis_play_anh6(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+5])
        cv2.imshow('GK61', f)
    def Dis_play_anh7(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+6])
        cv2.imshow('GK61', f)
    def Dis_play_anh8(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+7])
        cv2.imshow('GK61', f)
    def Dis_play_anh9(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+8])
        cv2.imshow('GK61', f)
    def Dis_play_anh10(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+9])
        cv2.imshow('GK61', f)
    def Dis_play_anh11(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+10])
        cv2.imshow('GK61', f)
    def Dis_play_anh12(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+11])
        cv2.imshow('GK61', f)
    def Dis_play_anh13(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+12])
        cv2.imshow('GK61', f)
    def Dis_play_anh14(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+13])
        cv2.imshow('GK61', f)
    def Dis_play_anh15(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+14])
        cv2.imshow('GK61', f)
    def Dis_play_anh16(self):
        f = cv2.imread(self.Tap_anh[self.thuong_16*16+15])
        cv2.imshow('GK61', f)
    def viettinbank_sukien(self):
        self.Reset_anh_sami()
        self.thuong_16 = 0
        self.Tap_anh = self._vietin
        self.Reset_anh()
    def vietcombank_sukien(self):
        self.Reset_anh_sami()
        self.thuong_16 = 0
        self.Tap_anh = self._vietcom
        self.Reset_anh()
    def bidv_sukien(self):
        self.Reset_anh_sami()
        self.thuong_16 = 0
        self.Tap_anh = self._bidv
        self.Reset_anh()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Team_Champion()
    sys.exit(app.exec_())
