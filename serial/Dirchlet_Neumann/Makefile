run:main.c
	mpicc -O0 -g -Wall -fbounds-check -I$(PETSC_DIR)/include -L$(PETSC_DIR)/lib -o $@ $< -lpetsc -lm

clean:
	@rm -vf run
