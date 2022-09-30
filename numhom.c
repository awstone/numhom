#include <petscsys.h>
#include <petsdmcplex.h>
#include <petscds.h>


/* exact solution to the oscillatory problem */
static PetscErrorCode oscillatory_u
(PetscInt dim, PetscReal time, const PetscReal x[],
 PetscInt Nc, PetscScalar *u, void *ctx)
{
  Parameter      *param = (Parameter *) ctx;
  const PetscReal eps   = param->epsilon;
  
  u[0] = x[0] - x[0]*x[0]
    + (eps / (2.*PETSC_PI))*(0.5 - x[0])*PetscSinReal(2.*PETSC_PI*x[0]/eps)
    + PetscSqr(eps / (2.*PETSC_PI))*(1. - PetscCosReal(2.*PETSC_PI*x[0]/eps));
  return 0;
}

/* integrand for the test function term */
static void f0_oscillatory_u
(PetscInt dim, PetscInt Nf, PetscInt NfAux,
 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
 const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
 const PetscScalar a_x[], PetscReal t, const PetscReal x[],
 PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -1.;
}

/* integrand for the test function gradient term */
static void f1_oscillatory_u
(PetscInt dim, PetscInt Nf, PetscInt NfAux,
 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
 const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
 const PetscScalar a_x[], PetscReal t, const PetscReal x[],
 PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal eps = PetscRealPart(constants[EPSILON]);

  f1[0] = u_x[0] / (2. + PetscCosReal(2.*PETSC_PI*x[0]/eps));
}

/* integrand for the test function gradient and basis function gradient term */
static void g3_oscillatory_uu
(PetscInt dim, PetscInt Nf, PetscInt NfAux,
 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
 const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
 const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
 PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal eps = PetscRealPart(constants[EPSILON]);

  g3[0] = 1. / (2. + PetscCosReal(2.*PETSC_PI*x[0]/eps));
}

int main(int argc; char **argv) {
  DM                    dmc, dmf; /* coarse and fine problem specification */
  SNES                  snes;     /* nonlinear solver */
  Vec                   u;        /* solution */
  PetscBool             simplex;  /* is the first cell in dmplex a simplex */
  PetscInt              dim;      /* dimension of dm */
  const PetscInt        id = 1;   /* values for constrained points */
  PetscFE               fe;       /* manages finite element space */

  /* initialize petsc */
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  
  /* setup mesh coarse */
  PetscCall(DMCreate(MPI_COMM_WORLD, dmc));
  PetscCall(DMSetType(*dmc, DMPLEX));
  PetscCall(DMSetFromOptions(*dmc));
  PetscCall(DMViewFromOptions(*dmc, NULL, "-dmc_view"));

  /* setup mesh fine */
  PetscCall(DMCreate(MPI_COMM_WORLD, dmf));
  PetscCall(DMSetType(*dmf, DMPLEX));
  PetscCall(DMSetFromOptions(*dmf));
  PetscCall(DMViewFromOptions(*dmf, NULL, "-dmf_view"));
  
  /* setup snes */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));

  /* setup discretization */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  /* create finite element */
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  /* setup discretization and boundary condition for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));

  /* setup problem */
  PetscCall(PetscDSSetResidual(ds, 0, f0_oscillatory_u, f1_oscillatory_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_oscillatory_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, oscillatory_u, NULL));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0,
			  NULL, (void, (*)(void)) oscillatory_u, NULL, NULL, NULL));
  
  
  /* finalize petsc */
  PetscCall(PetscFinalize());
  
  return 0;
}


