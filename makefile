EXECS=kmeans
MPICC?=mpicc

all: ${EXECS}

kmeans: kmeans.c
	${MPICC} -o kmeans kmeans.c

run: kmeans
	mpirun -n 100 ./kmeans 1000 4 2

clean:
	rm -f ${EXECS}
