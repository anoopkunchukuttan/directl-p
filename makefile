
## STLport is the location of STLport ##
## the library can be download from 
## http://www.stlport.org/
STLport=./STLport-5.2.1

## SVMsrc is the SVMlight source code 
## which can be downloaded from
## http://svmlight.joachims.org/
## svm_common.c svm_hideo.c svm_learn.c
## speficied here but you need to put these files and also with
## kernel.h svm_common.h svm_learn.h 
## in the same location
SVMsrc=svm_common.c svm_hideo.c svm_learn.c

## tclap library
TCLAP=./tclap-1.2.0

## g++ options for code optimization
GNUREC=-O3 -ffast-math -funroll-all-loops -fpeel-loops -ftracer -funswitch-loops -funit-at-a-time -pthread
GO=$(GNUREC)

## g++ 
CC=g++ $(GO) 

INCLUDES=-I$(TCLAP)/include/ -I$(STLport)/stlport
LIBS=-L$(STLport)/lib
CFLAGS=-c $(INCLUDES) 
LDFLAGS=$(LIBS) 
INLIBS=-lstlport -lgcc_s -lpthread -lc -lm
L2Psrc=allPhonemeSet.cpp weightWF.cpp miraPhrase.cpp phraseModel.cpp
SOURCES=$(SVMsrc) $(L2Psrc)
SVMobj=$(SVMsrc:.c=.o)
L2Pobj=$(L2Psrc:.cpp=.o)
OBJECTS=$(SVMobj) $(L2Pobj)
EXECUTABLE=directlp

all: $($SOURCES) $(EXECUTABLE)

$(EXECUTABLE):	$(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@ $(INLIBS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:	
	rm -f $(EXECUTABLE) $(OBJECTS)
