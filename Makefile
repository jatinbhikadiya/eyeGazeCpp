CC = g++
CFLAGS = -c 

WFLAGS = -Wall 

OPTS = -O0
OFLAGS =  -g2 

#LFLAGS = -Lgui -lcpptk -ltcl8.5 -ltk8.5
LFLAGS = -llapack -lm

#Boost Libs Go Here
BOOST_LIBS = -lboost_python  -lboost_filesystem  -lboost_program_options 
PYTHON_LIBS = -lpython2.7
EXECUTABLE = brivasmodule.so

#Opencv Includes Go here
OPENCV_PATH = /usr/local/include
OPENCV_LIB = /usr/local/lib
OPENCV_LIBS = -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ts -lopencv_video
OPENCV_INCPATH = -I$(OPENCV_PATH)/opencv
OPENCV2_INCPATH = -I$(OPENCV_PATH)
OPENCV_LIBPATH = -L$(OPENCV_LIB)
PYTHON_PATH = /usr/include/python2.7
PYTHON_INCPATH = -I$(PYTHON_PATH)
CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))


SUBMIT_DIR = $(shell whoami)
BACKUP_DIR = $(shell date "+%b_%d_%Y_%I_%M")
BACKUP_REPO	= ./Backups
BACKUP_PATH = $(BACKUP_REPO)/$(BACKUP_DIR)

all:$(EXECUTABLE)

$(EXECUTABLE): $(OBJ_FILES)
	$(CC) $(WFLAGS) $(OPTS) $(LFLAGS)  $(OPENCV_INCPATH)  $(OPENCV2_INCPATH) $(OPENCV_LIBPATH) $(PYTHON_LIBS) $(OPENCV_LIBS) $(BOOST_LIBS) $^ -o $@   $(LFLAGS)

obj/%.o: src/%.cpp
	mkdir -p ./obj
	$(CC) $(CFLAGS) $(OPENCV_INCPATH) $(PYTHON_INCPATH)  $(OPENCV2_INCPATH) $(WFLAGS) $(OPTS) $(OFLAGS)  -c -o $@ $<  



clean:
	rm -f $(OBJ_FILES)
	rm -f *.out
	rm -f *~
	rm -f $(EXECUTABLE) 
	
	
#Create a Backup directory with <Month>_<Date>_<Year>_<Hr>_<Min>_<Sec>.tar
backup: 
	mkdir -p $(BACKUP_REPO)
	mkdir -p $(BACKUP_PATH)
	mkdir -p $(BACKUP_PATH)/src
	cp -r ./src/*.h ./$(BACKUP_PATH)/src
	cp -r ./src/*.cpp ./$(BACKUP_PATH)/src
	cp Makefile $(BACKUP_PATH)/
	cp TestScript.sh $(BACKUP_PATH)/
	tar -zcvf $(BACKUP_REPO)/$(BACKUP_DIR).tar $(BACKUP_PATH)/
	rm -rf $(BACKUP_PATH)
