CC = icc
CFLAGS = -O3 -openmp -mmic -parallel

PROGRAMS= stream

default: $(PROGRAMS)

clean:
	rm -f $(PROGRAMS) *.o

run: $(PROGRAMS)
	export SINK_LD_LIBRARY_PATH=/opt/intel/composer_xe_2013.2.146/compiler/lib/mic/
	#/opt/intel/mic/bin/micnativeloadex memcpy
	/opt/intel/mic/bin/micnativeloadex stream
