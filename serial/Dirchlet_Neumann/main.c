
static char help[] = "Nonlinear driven cavity with multigrid in 2d.\n \
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\n";
/* in HTML, '&lt' = '<' and '&gt' = '>' */

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DMDA^using distributed arrays;
   Concepts: multicomponent
   Processors: 1
T*/


/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar x,y;
} Field;

PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field**,Field**,void*);  //init Formfunc

typedef struct {
  PetscScalar 	p, p_delta, p_max;            /*plaplacian-param*/

	PetscScalar 	uul, uvl, uur, uvr;
  PetscScalar 	uut, uvt, uub, uvb;

  PetscScalar 	duut, duvt, duub, duvb;
  PetscScalar 	duul, duvl, duur, duvr;

  PetscInt	  nx;                      /*Nr Nodes x-y */

  PetscBool   draw_contours, save_solution;               /* Flags*/
} AppCtx;

typedef struct {
  PetscViewer viewer;
} MonitorCtx;

extern PetscErrorCode FormInitialGuess(AppCtx*,DM,Vec); //init Inital
extern PetscErrorCode SaveSolution(AppCtx*,DM,Vec); //init Inital

extern double interpol(double,double);
extern PetscScalar dot(const PetscScalar [][2] , const PetscScalar [][2]);
extern PetscScalar abssum(const PetscScalar [][2]);


int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da;
  Vec            u;
  PetscOptions   options;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsCreate(&options);
  ierr = PetscOptionsInsertString(options,"snes_monitor_lg_residualnorm 1");


  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;


	/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	Create Snes OBJ
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	ierr = SNESCreate(comm, &snes); CHKERRQ(ierr);
	ierr = SNESSetType(snes, SNESQN ); CHKERRQ(ierr);  //SET SNESQN SNESQN)

  /*
      Create distributed array object to manage parallel grid and vectors
      for principal unknowns (u) and governing residuals (f)
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,50,50,PETSC_DECIDE,PETSC_DECIDE,2,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

  ierr = SNESSetDM(snes,(DM)da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  /*
     Problem parameters 
  */

  ierr = PetscPrintf(comm,"mx = %f, my # = %f\n",(double)mx,(double)my);CHKERRQ(ierr);

  user.nx = mx;
	user.p_max = 2.0;
	user.p_delta = 0.1;
	user.p = 2.0;

//TOP BOTTOM
  user.duut = 0.0; //Neumann boundary Cond
	user.duvt = 0.0;  //Neumann boundary Cond
  user.duub = 0.0; //Neumann boundary Cond
	user.duvb = 0.0;  //Neumann boundary Cond

  user.uut = 0.0; //Dirich - boundary Cond
	user.uvt = 0.0;  //Dirich - boundary Cond
  user.uub = 0.0; //Dirich - boundary Cond
	user.uvb = 0.0;  //Dirich - boundary Cond


//LEFT RIGHT
  user.uul = 1.0; //Dirich - boundary Cond
	user.uvl = 0.0;  //Dirich - boundary Cond
  user.uur = 0.0; //Dirich - boundary Cond
	user.uvr = 0.0;  //Dirich - boundary Cond

  user.duul = 0.0; //Neumann boundary Cond
	user.duvl = 0.0;  //Neumann boundary Cond
  user.duur = -1.0; //Neumann boundary Cond
	user.duvr = 0.0;  //Neumann boundary Cond

	user.save_solution = 0;
	user.draw_contours = 0;

  ierr = DMDASetFieldName(da,0,"x_displacement");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"y_displacement");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create user context, set problem data, create vector data structures.
  Also, compute the initial guess.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	PetscInt count = 0;
	ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = FormInitialGuess(&user,da,u);CHKERRQ(ierr);  //Init field with 0

  ierr = PetscPrintf(comm,"p_max = %f, delta_p # = %f\n",(double)user.p_max,(double)user.p_delta); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "#################\n"); CHKERRQ(ierr);


	while (1)	
	{	
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Create nonlinear solver context

      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Solve the nonlinear system
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "#########\n"); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\tp = %lf\n", count, user.p); CHKERRQ(ierr);

    ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);


    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    ierr = SaveSolution(&user,da,u);

    user.p = user.p + user.p_delta;
    if (user.p > user.p_max + 1.0e-12) break;
    count++;
	}

  /*
     Visualize solution
  */
  if (user.draw_contours) {
    ierr = VecView(u,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }


	ierr = VecViewFromOptions(u, NULL, "-solution_view"); CHKERRQ(ierr);



  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */

/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
*/
PetscErrorCode FormInitialGuess(AppCtx *user,DM da,Vec U)
{
  PetscInt       i,j,mx,my, xs,ys,xm,ym;
  PetscErrorCode ierr;
  Field          **u;

  PetscFunctionBeginUser;

  ierr = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArrayWrite(da,U,&u);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      u[j][i].x     = 0.0;
      u[j][i].y     = 0.0;
    }
  }
  /*
     Restore vector
  */
  ierr = DMDAVecRestoreArrayWrite(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
*/
PetscErrorCode SaveSolution(AppCtx *user,DM da,Vec U)
{
  PetscInt       i,j,mx,my, xs,ys,xm,ym;
  FILE			      *fp;
  PetscErrorCode ierr;
  Field          **u;

  PetscFunctionBeginUser;

  ierr = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArrayWrite(da,U,&u);CHKERRQ(ierr);

  /*
     write data
  */
  ierr = PetscFOpen(PETSC_COMM_WORLD, "x_solution.dat", "w", &fp); CHKERRQ(ierr);
  // plot 1 line
	ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ",(PetscReal) mx+1); CHKERRQ(ierr);
  for (i = 0; i < mx; i++){
		ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ",(PetscReal) ((i+0.5) * 1.0/mx)); CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "\n"); CHKERRQ(ierr);

  // plot lines
	for (j = 0; j < my; j++){
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ", (PetscReal) ((j+0.5) *  1.0/my)); CHKERRQ(ierr);
    for (i = 0; i < my; i++){
      ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ", u[j][i].x); CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "\n"); CHKERRQ(ierr);
  }
	ierr = PetscFClose(PETSC_COMM_WORLD, fp); CHKERRQ(ierr);

  ierr = PetscFOpen(PETSC_COMM_WORLD, "y_solution.dat", "w", &fp); CHKERRQ(ierr);
  // plot 1 line
	ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ",(PetscReal) mx+1); CHKERRQ(ierr);
  for (i = 0; i < mx; i++){
		ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ",(PetscReal) ((i+0.5) * 1.0/mx)); CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "\n"); CHKERRQ(ierr);

  // plot lines
	for (j = 0; j < my; j++){
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ", (PetscReal) ((j+0.5) *  1.0/my)); CHKERRQ(ierr);
    for (i = 0; i < my; i++){
      ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "%2.2lf  ", u[j][i].y); CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fp, "\n"); CHKERRQ(ierr);
  }
	ierr = PetscFClose(PETSC_COMM_WORLD, fp); CHKERRQ(ierr);

  /*
     Restore vector
  */
  ierr = DMDAVecRestoreArrayWrite(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   FormFunctionLocac - Forms local form approx

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field **u,Field **f,void *ptr)
{
  AppCtx           *user = (AppCtx*)ptr;
  PetscInt         dhx, dhy, xints,xinte,yints,yinte,i,j,k,m,iface;
  PetscScalar      hx, hy, one_over_volume, volume; 
  PetscScalar  		 un, ue, us, uw, up;
  PetscScalar      fx,fy;
  PetscScalar	  	 exponent = (user->p - 2.0)*0.5;
  PetscErrorCode   ierr;

	PetscFunctionBeginUser;
  /*
     Define mesh intervals ratios for uniform grid.
  */

  dhx   = info->mx;  dhy = info->my;
  hx    = 1.0/dhx;    hy = 1.0/dhy;

  volume = hx*hy*1.0;
  one_over_volume = 1.0/volume;

  //PetscPrintf(PETSC_COMM_WORLD,"dhx = %i\n",dhx);
  //PetscPrintf(PETSC_COMM_WORLD,"hx = %f\n",hx);
  //PetscPrintf(PETSC_COMM_WORLD,"volume = %f\n",volume);
  //PetscPrintf(PETSC_COMM_WORLD,"one_over_volume = %f\n",one_over_volume);



  xints = info->xs; xinte = info->xs+info->xm; 
  yints = info->ys; yinte = info->ys+info->ym;

  PetscScalar n_f[4][2] = {{0,1},{1, 0},{0, -1},{-1, 0}}; //face Normals //n 0 -1 //e -1 0 //s 0 1 //w 1 0

  PetscScalar u_f[4] = {0,0,0,0}; 

  /*Init Gradient*/
  PetscScalar Du [info->ym][info->xm][2][2];
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      for (k=0; k<2; k++){
        for (m=0; m<2; m++){
          Du[j][i][k][m]= 0.0;//init grad with 0 .  inital state
        }
      }
    }
  }
  
  /*compute Gradient*/
  // DU = {{du1 /dx1 , du2/dx1}, {du1/dx2, du2,dx2}}
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      //PetscPrintf(PETSC_COMM_WORLD,"j = %i i = %i\n",j,i);


      u_f[0] = 0.0;
      u_f[1] = 0.0;
      u_f[2] = 0.0;
      u_f[3] = 0.0;
    
      /// Imply Boundary Conditions Dirichlet and Neumann
      if (j == yints && i == xints ) {  /* x=0 y=0 corner*/
        //DIS X
                              un = u[j+1][i].x;
        uw = u[j][i].x ;  		up = u[j][i].x; 	ue = u[j][i+1].x;  //uw not avail 
                              us = u[j][i].x; //not avail   //using cell centered one

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = user->uub; //interpol(up,us)
        u_f[3] = user->uul; //interpol(up,up);


        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

        //DIS Y
                          un = u[j+1][i].y;
        uw = u[j][i].y ;  up = u[j][i].y; 	ue = u[j][i+1].y;
                          us = u[j][i].y; //not avail //using cell centered one

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = user->uvb; //interpol(up,us)
        u_f[3] = user->uvl; // interpol(up,up); 

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

      } else if (j == yints && i == xinte-1 ) {  /* x=max y=0 corner*/
        // DIS X
                              un = u[j+1][i].x;
        uw = u[j][i-1].x ;		up = u[j][i].x; 
                              us = u[j][i].x; //not avail  //using cell centered one

        u_f[0] = interpol(up,un);
        u_f[1] = up-(user->duur*(hx/2)); //Neumann //interpol(up,up); //user->uur; //
        u_f[2] = user->uub; //interpol(up,us)
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }
  
        // DIS Y
                              un = u[j+1][i].y;
        uw = u[j][i-1].y ;		up = u[j][i].y; 	
                              us = u[j][i].y; //not avail  //using cell centered one

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,up); // user->uvr; //
        u_f[2] = user->uvb; //interpol(up,us)
        u_f[3] = interpol(up,uw);


        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

      } else if (j == yinte-1 && i == xints) {         /* x=0 y=max corner*/
        //DIS X
                        un = u[j][i].x; //not avail
			                	up = u[j][i].x; 	ue = u[j][i+1].x;
									      us = u[j-1][i].x;

        u_f[0] = user->uut; //interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = user->uul; // interpol(up,up); //

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }
      
        //DIS Y
                        un = u[j][i].y; //not avail
			                  up = u[j][i].y; 	ue = u[j][i+1].y;
									      us = u[j-1][i].y;

        u_f[0] = user->uvt; //interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = user->uvl; // interpol(up,up); //

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

      } else if (j == yinte-1 && i == xinte-1) {        /* x=max y=max corner*/
        //DIS X
                              un = u[j][i].x; //not avail
        uw = u[j][i-1].x ;		up = u[j][i].x; 	
                              us = u[j-1][i].x;

        u_f[0] = user->uut; //interpol(up,un);
        u_f[1] = up-(user->duur*(hx/2)); //interpol(up,up); //user->uur; //
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

        //DIS Y
                              un = u[j][i].y; //not avail
        uw = u[j][i-1].y ;		up = u[j][i].y; 
                              us = u[j-1][i].y;

        u_f[0] = user->uvt; //interpol(up,un);
        u_f[1] = interpol(up,up); //user->uvr; //
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }
        
      } else if (j == yints) {         /* bottom edge*/
        //DIS X
                            un = u[j+1][i].x;
        uw = u[j][i-1].x;		up = u[j][i].x; 	ue = u[j][i+1].x;
                            us = u[j][i].x; //not avail

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = user->uub; //interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

        //DIS Y
                              un = u[j+1][i].y;
        uw = u[j][i-1].y ;		up = u[j][i].y; 	ue = u[j][i+1].y;
                              us = u[j][i].y; //not avail

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = user->uvb; //interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

      } else if (j == yinte-1) {         /* top edge */
        // DIS X
                              un = u[j][i].x; //not avail
        uw = u[j][i-1].x ;	  up = u[j][i].x; 	ue = u[j][i+1].x;
                              us = u[j-1][i].x;

        u_f[0] = user->uut; //interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

        // DIS Y
                              un = u[j][i].y; //not avail
        uw = u[j][i-1].y ;		up = u[j][i].y; 	ue = u[j][i+1].y;
                              us = u[j-1][i].y;

        u_f[0] = user->uvt; //interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

      } else if (i == xints) { /* left edge */

        //DIS X
                              un = u[j+1][i].x;
                          		up = u[j][i].x; 	ue = u[j][i+1].x;
                              us = u[j-1][i].x;

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = user->uul; // interpol(up,up); //

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

        //DIS Y
                              un = u[j+1][i].y;
                          		up = u[j][i].y; 	ue = u[j][i+1].y;
                              us = u[j-1][i].y;

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = user->uvl; // interpol(up,up); //

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

      } else if (i == xinte-1) {        /* right edge */

        //DIS X
                              un = u[j+1][i].x;
        uw = u[j][i-1].x ;		up = u[j][i].x; 
                              us = u[j-1][i].x;

        u_f[0] = interpol(up,un);
        u_f[1] = up-(user->duur*(hx/2)); //interpol(up,up); //user->uur; //
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

        //DIS Y
                              un = u[j+1][i].y;
        uw = u[j][i-1].y ;		up = u[j][i].y; 	
                              us = u[j-1][i].y;

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,up); //user->uvr; //
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

      } else {
        /* Compute over the interior points */
        //DIS X
                              un = u[j+1][i].x;
        uw = u[j][i-1].x ;		up = u[j][i].x; 	ue = u[j][i+1].x;
                              us = u[j-1][i].x;

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][0]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][0]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }

        //DIS Y
                              un = u[j+1][i].y;
        uw = u[j][i-1].y ;		up = u[j][i].y; 	ue = u[j][i+1].y;
                              us = u[j-1][i].y;

        u_f[0] = interpol(up,un);
        u_f[1] = interpol(up,ue);
        u_f[2] = interpol(up,us);
        u_f[3] = interpol(up,uw);

        for(iface=0;iface<4;iface++){
          Du[j][i][0][1]  +=  (u_f[iface] * n_f[iface][0] * hx)*one_over_volume;
          Du[j][i][1][1]  +=  (u_f[iface] * n_f[iface][1] * hx)*one_over_volume;
        }
      }
      //PetscPrintf(PETSC_COMM_WORLD,"D%i %i 00 = %e\n",j,i, Du[j][i][0][0]);
      //PetscPrintf(PETSC_COMM_WORLD,"D%i %i 01 = %e\n",j,i, Du[j][i][0][1]);
      //PetscPrintf(PETSC_COMM_WORLD,"D%i %i 10 = %e\n",j,i, Du[j][i][1][0]);
      //PetscPrintf(PETSC_COMM_WORLD,"D%i %i 11 = %e\n",j,i, Du[j][i][1][1]);

    }
  }



  PetscScalar  	du_f[4][2][2];  //Face Deriv [n,e,s,w]
  PetscScalar   eta_f;  //Face eta

  //Calc FormFunction
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      //PetscPrintf(PETSC_COMM_WORLD,"j = %i i = %i\n",j,i);


      fx = 0.0;
      fy = 0.0;

      for(iface=0;iface<4;iface++){
        for (k=0; k<2; k++){
          for (m=0; m<2; m++){
            du_f[iface][k][m]= 0.0; 
          }
        }
      }

      //imply boundary Cond neumann
      if (j == yints && i == xints ) {         /* x=0 y=0 corner*/
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j+1][i][k][m]);//North
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i+1][k][m]);//East
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]);//South  //Take Cell Grad //is this the right assumption ?  What walls do we have ? 
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]);//West //Take Cell Grad
          }
        }

        //SouthFace
        //du_f[2][1][0] = user->duub; //Modify due to Neumann  /normal dir
        //du_f[2][1][1] = user->duvb;  //Modify due to Neumann /normal dir

        //WestFace
        //du_f[3][0][0] = user->duul; //Modify due to Neumann  /normal dir
        //du_f[3][0][1] = user->duvl;  //Modify due to Neumann /normal dir

      } else if (j == yints && i == xinte-1 ) {         /* x=max y=0 corner*/
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j+1][i][k][m]);
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]); //Take Cell Grad
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]);//Take Cell Grad
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i-1][k][m]);
          }
        }
        //SouthFace
        //du_f[2][1][0] = user->duub; //Modify due to Neumann  /normal dir
        //du_f[2][1][1] = user->duvb;  //Modify due to Neumann /normal dir

        //EastFace
        du_f[1][0][0] = user->duur; //Modify due to Neumann  /normal dir
        du_f[1][0][1] = user->duvr;  //Modify due to Neumann /normal dir

      } else if (j == yinte-1 && i == xints) {         /* x=0 y=max corner*/
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]); //Take Cell Grad
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i+1][k][m]);
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j-1][i][k][m]);
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]); //Take Cell Grad
          }
        }
        // Northface
        //du_f[0][1][0] = user->duut; //Modify due to Neumann  /normal dir
        //du_f[0][1][1] = user->duvt;  //Modify due to Neumann /normal dir

        //WestFace
        //du_f[3][0][0] = user->duul; //Modify due to Neumann  /normal dir
        //du_f[3][0][1] = user->duvl;  //Modify due to Neumann /normal dir


      } else if (j == yinte-1 && i == xinte-1) {         /* x=max y=max corner*/              
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]);  //Take Cell Grad
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]);  //Take Cell Grad
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j-1][i][k][m]);
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i-1][k][m]);
          }
        }
        //Northface
        //du_f[0][1][0] = user->duut; //Modify due to Neumann  /normal dir
        //du_f[0][1][1] = user->duvt;  //Modify due to Neumann /normal dir


        //EastFace
        du_f[1][0][0] = user->duur; //Modify due to Neumann  /normal dir
        du_f[1][0][1] = user->duvr;  //Modify due to Neumann /normal dir


      } else if (j == yints) {        /* bottom edge*/    
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j+1][i][k][m]);
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i+1][k][m]);
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]); //Take Cell Grad
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i-1][k][m]);
          }
        }

        //du_f[2][1][0] = user->duub; //Modify due to Neumann  /normal dir
        //du_f[2][1][1] = user->duvb;  //Modify due to Neumann /normal dir

      } else if (j == yinte-1) {         /* top edge */
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]); //Take Cell Grad
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i+1][k][m]);
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j-1][i][k][m]);
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i-1][k][m]);
          }
        }
        //du_f[0][1][0] = user->duut; //Modify due to Neumann  /normal dir
        //du_f[0][1][1] = user->duvt;  //Modify due to Neumann /normal dir

      } else if (i == xints) {         /* left edge */
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j+1][i][k][m]);
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i+1][k][m]);
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j-1][i][k][m]);
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]); //Take Cell Grad
          }
        }
        //WestFace
        //du_f[3][0][0] = user->duur; //Modify due to Neumann  /normal dir
        //du_f[3][0][1] = user->duvr;  //Modify due to Neumann /normal dir
        
      } else if (i == xinte-1) {         /* right edge */
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j+1][i][k][m]);
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i][k][m]);  //Take Cell Grad
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j-1][i][k][m]);
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i-1][k][m]);
          }
        }

        //EastFace
        du_f[1][0][0] = user->duul; //Modify due to Neumann  /normal dir
        du_f[1][0][1] = user->duvl;  //Modify due to Neumann /normal dir

 
      } else {
        /* Compute over the interior points */
        for (k = 0; k<2; k++){
          for(m = 0; m<2; m++){
            du_f[0][k][m] = interpol(Du[j][i][k][m],Du[j+1][i][k][m]);
            du_f[1][k][m] = interpol(Du[j][i][k][m],Du[j][i+1][k][m]);
            du_f[2][k][m] = interpol(Du[j][i][k][m],Du[j-1][i][k][m]);
            du_f[3][k][m] = interpol(Du[j][i][k][m],Du[j][i-1][k][m]);
          }
        }
      }

      for (iface=0; iface<4; iface++){
        //PetscPrintf(PETSC_COMM_WORLD,"du_f %i 00 = %e \n",iface, du_f[iface][0][0]);
        //PetscPrintf(PETSC_COMM_WORLD,"du_f %i 01 = %e \n",iface, du_f[iface][0][1]);
        //PetscPrintf(PETSC_COMM_WORLD,"du_f %i 10 = %e \n",iface, du_f[iface][1][0]);
        //PetscPrintf(PETSC_COMM_WORLD,"du_f %i 11 = %e \n",iface, du_f[iface][1][1]);

        eta_f = pow(dot(du_f[iface],du_f[iface]),exponent);
        
        /* X Displacement */
        fx -= eta_f * (du_f[iface][0][0] * n_f[iface][0] + du_f[iface][1][0] * n_f[iface][1]);

        /* Y Displacement */
        fy -= eta_f * (du_f[iface][0][1] * n_f[iface][0] + du_f[iface][1][1] * n_f[iface][1]);
      }

      /* Displacements */
      f[j][i].x = hx*fx;
  
      //f[j][i].y = hy*fy;  

      f[j][i].y = u[j][i].y-0;
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

  /*
     Helper
  */

PetscScalar interpol(const PetscScalar p, const PetscScalar fn) {
    PetscScalar  vec_f;
    vec_f = fn * 0.5 + p * 0.5;
    //PetscPrintf(PETSC_COMM_WORLD,"vec_f %f = %f * 0.5 + %f * 0.5\n",vec_f,fn,p);
    return vec_f;
}

PetscScalar dot(const PetscScalar u[][2] , const PetscScalar v[][2]) { //v[][2]
    PetscScalar sum = 0.0;
    for (int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++ ){
        sum += u[i][j] * v[i][j];
      }
    }
    //PetscPrintf(PETSC_COMM_WORLD,"dof  = %f\n",sum);

    return sum;
}

PetscScalar abssum(const PetscScalar u[][2]) { //v[][2]
    PetscScalar sum = 0.0;
    for (int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++ ){
        sum += abs(u[i][j]);
      }
    }
    return sum;
}