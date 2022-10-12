#include <petscsys.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>

/* exact solution to the oscillatory problem */
static PetscErrorCode oscillatory_u
(PetscInt dim, PetscReal time, const PetscReal x[],
 PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal eps = 0.03125;
  
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
  const PetscReal eps = 0.03125;

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
  const PetscReal eps = 0.03125;

  g3[0] = 1. / (2. + PetscCosReal(2.*PETSC_PI*x[0]/eps));
}

int main(int argc, char **argv) {
  DM                    dmc, dmf;       /* problem specification coarse, fine */
  PetscDS               dsc, dsf;       /* discrete system coarse, fine */
  SNES                  snesc, snesf;   /* nonlinear solver coarse, fine */
  Vec                   uc, uf;         /* solution coarse, fine */
  PetscBool             simplex;        /* is the first cell in dmplex a simplex */
  PetscInt              dimc, dimf;     /* dimension of dm coarse, fine */
  const PetscInt        id = 1;         /* values for constrained points */
  PetscFE               fe;             /* manages finite element space */
  DMLabel               label;          /* label */

  /* initialize petsc */
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  
  /* setup mesh coarse */
  PetscCall(DMCreate(MPI_COMM_WORLD, &dmc));
  PetscCall(DMSetOptionsPrefix(dmc, "coarse_"));
  PetscCall(DMSetType(dmc, DMPLEX));
  PetscCall(DMSetFromOptions(dmc));
  PetscCall(DMViewFromOptions(dmc, NULL, "-dmc_view"));
  
  /* setup mesh fine */
  PetscCall(DMCreate(MPI_COMM_WORLD, &dmf));
  PetscCall(DMSetOptionsPrefix(dmf, "fine_"));
  PetscCall(DMSetType(dmf, DMPLEX));
  PetscCall(DMSetFromOptions(dmf));
  PetscCall(DMViewFromOptions(dmf, NULL, "-dmf_view"));
  
  /* setup snes coarse */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snesc));
  PetscCall(SNESSetDM(snesc, dmc));

  /* setup sens fine */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snesf));
  PetscCall(SNESSetDM(snesf, dmf));

  /* setup discretization coarse */
  PetscCall(DMGetDimension(dmc, &dimc));
  PetscCall(DMPlexIsSimplex(dmc, &simplex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dimc, 1, simplex, NULL, -1, &fe));
  PetscCall(DMSetField(dmc, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dmc));
  PetscCall(PetscFEDestroy(&fe));

  /* setup discretization fine */
  PetscCall(DMGetDimension(dmf, &dimf));
  PetscCall(DMPlexIsSimplex(dmf, &simplex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dimf, 1, simplex, NULL, -1, &fe));
  PetscCall(DMSetField(dmf, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dmf));
  PetscCall(PetscFEDestroy(&fe));

  /* setup problem coarse */
  PetscCall(DMGetDS(dmc, &dsc));
  PetscCall(PetscDSSetResidual(dsc, 0, f0_oscillatory_u, f1_oscillatory_u));
  PetscCall(PetscDSSetJacobian(dsc, 0, 0, NULL, NULL, NULL, g3_oscillatory_uu));
  PetscCall(PetscDSSetExactSolution(dsc, 0, oscillatory_u, NULL));
  PetscCall(DMGetLabel(dmc, "marker", &label));
  PetscCall(DMAddBoundary(dmc, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0,
			  NULL, (void (*)(void)) oscillatory_u, NULL, NULL, NULL));

  /* setup problem fine */
  PetscCall(DMGetDS(dmf, &dsf));
  PetscCall(PetscDSSetResidual(dsf, 0, f0_oscillatory_u, f1_oscillatory_u));
  PetscCall(PetscDSSetJacobian(dsf, 0, 0, NULL, NULL, NULL, g3_oscillatory_uu));
  PetscCall(PetscDSSetExactSolution(dsf, 0, oscillatory_u, NULL));
  PetscCall(DMGetLabel(dmf, "marker", &label));
  PetscCall(DMAddBoundary(dmf, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0,
			  NULL, (void (*)(void)) oscillatory_u, NULL, NULL, NULL));

  /* solve problem coarse */
  PetscCall(DMCreateGlobalVector(dmc, &uc));
  PetscCall(VecSet(uc, 0.0));
  PetscCall(DMPlexSetSNESLocalFEM(dmc, NULL, NULL, NULL));
  PetscCall(SNESSolve(snesc, NULL, uc));
  PetscCall(SNESGetSolution(snesc, &uc));

  /* solve problem fine */
  PetscCall(DMCreateGlobalVector(dmf, &uf));
  PetscCall(VecSet(uf, 0.0));
  PetscCall(DMPlexSetSNESLocalFEM(dmf, NULL, NULL, NULL));
  PetscCall(SNESSolve(snesf, NULL, uf));
  PetscCall(SNESGetSolution(snesf, &uf));
  /* show solutions */
  PetscCall(VecView(uc, NULL));
  PetscCall(VecView(uf, NULL));

  /* finalize petsc */
  PetscCall(PetscFinalize());
  
  return 0;
}


